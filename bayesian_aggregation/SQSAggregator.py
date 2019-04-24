from .SQSClient import SQSClient
from .SQSMessageParser import SQSMessageParser

import importlib
import signal
import sys
import os

if importlib.util.find_spec("crowdsourcing") is not None:
    from crowdsourcing.annotations.detection.bbox import CrowdDatasetBBox
else:
    error = 'The required module "crowdsourcing" is missing. It is available from "https://github.com/hughdickinson/crowdsourcing".'
    raise ModuleNotFoundError(error)


class SQSAggregator:
    def __init__(
        self,
        queueUrl,
        maxRisk=0.5,
        messageBatchSize=200,
        savePath=".",
        savePrefix="aggregator_output",
        saveInputAnnotations=False,
        purgeOldBBoxSetFile=False,
        **kwargs
    ):
        self.sqsClient = SQSClient(queueUrl=queueUrl)
        self.sqsMessageParser = SQSMessageParser(**kwargs)
        self.savePath = savePath
        self.savePrefix = savePrefix
        self.fullSavePrefix = os.path.join(self.savePath, self.savePrefix)
        self.saveInputAnnotations = saveInputAnnotations

        self.maxRisk = maxRisk
        self.aggregator = CrowdDatasetBBox(
            debug=0,
            learn_worker_params=True,
            learn_image_params=True,
            estimate_priors_automatically=True,
            computer_vision_predictor=None,
            naive_computer_vision=False,
            min_risk=self.maxRisk,
        )
        self.aggregator.fname = self.fullSavePrefix

        self.messageBatchSize = messageBatchSize
        self.allUniqueMessages = []
        self.inputAnnotations = [] if self.saveInputAnnotations else None

        self.verbose = kwargs.get("verbose", False)

        # Register interrupt handler.
        signal.signal(signal.SIGINT, self.stopLoopHandler)

        if purgeOldBBoxSetFile:
            self.purgeBBoxSetFile()

    def purgeBBoxSetFile(self):
        bboxSetFilePath = self.fullSavePrefix + ".big_bbox_set.pkl"
        if os.path.exists(bboxSetFilePath):
            os.remove(bboxSetFilePath)

    def accumulateMessages(self):
        while len(self.allUniqueMessages) < self.messageBatchSize:
            uniqueMessages, allMessages, messageIds = self.sqsClient.getMessages()
            if not len(uniqueMessages):
                return False
            self.allUniqueMessages.extend(uniqueMessages)

        return True

    def aggregate(self):
        if not self.accumulateMessages() and len(self.allUniqueMessages) == 0:
            # If no messages are available for processing
            return False

        self.sqsMessageParser.processMessages(self.allUniqueMessages)

        if self.saveInputAnnotations:
            self.inputAnnotations.extend(
                self.sqsMessageParser.aggregatorInputData["annos"]
            )

        self.aggregator.load(
            data=self.sqsMessageParser.getAggregatorInputData(), overwrite_workers=False
        )
        self.aggregator.get_big_bbox_set()
        self.aggregator.estimate_parameters(
            avoid_if_finished=True, max_iters=10, refine=True
        )
        self.allUniqueMessages = []
        return True

    def checkNumFinished(self):
        image_id_to_finished = self.aggregator.check_finished_annotations(
            set_finished=True
        )
        num_finished = sum(image_id_to_finished.values())
        print(
            "{:d} / ({:d}) ({:.2f}%) images are finished".format(
                num_finished,
                len(image_id_to_finished),
                100.0 * float(num_finished) / len(image_id_to_finished),
            )
        )

    def save(self):
        self.aggregator.save(
            self.fullSavePrefix + "_aggregated.json",
            save_dataset=True,
            save_images=True,
            save_workers=True,
            save_annos=True,
            save_combined_labels=True,
        )

    def getInputAnnotations(self):
        return self.inputAnnotations

    def stopLoopHandler(self, sig, frame):
        while True:
            quitResponse = input("Are you sure you want to stop? (y/n)")
            if quitResponse == "y":
                while True:
                    saveResponse = input("Save aggregation results? (y/n)")
                    if saveResponse == "y":
                        self.save()
                    elif saveResponse == "n":
                        sys.exit(0)
                    else:
                        print("Unrecognized response.")
            elif quitResponse == "n":
                self.loop()
            else:
                print("Unrecognized response.")

    def loop(self, verbose=True, stopOnExhaustion=False):
        while True:
            if self.aggregate():
                if verbose:
                    self.checkNumFinished()
            elif not stopOnExhaustion:
                print("No messages recieved. Waiting...")
                continue
            else:
                print("No messages recieved. Stopping.")
                self.checkNumFinished()
                break
