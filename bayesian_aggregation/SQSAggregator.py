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
        saveInputMessages=False,
        purgeOldBBoxSetFile=False,
        **kwargs
    ):

        self.savePath = savePath
        self.savePrefix = savePrefix

        self.sqsClient = SQSClient(queueUrl=queueUrl, **kwargs)

        self.saveInputAnnotations = saveInputAnnotations
        self.saveInputMessages = saveInputMessages

        self.maxRisk = maxRisk

        self.messageBatchSize = messageBatchSize

        # Support aggregation for multiple tasks using sub-aggregators
        self.sqsMessageParsers = []
        self.subAggregators = []
        self.fullSavePrefixes = []
        self.taskLabels = kwargs.get("taskLabels", ["T0"])
        try:
            for taskCounter, taskLabel in enumerate(self.taskLabels):
                self.sqsMessageParsers.append(
                    SQSMessageParser(taskLabel=taskLabel, **kwargs)
                )
                self.subAggregators.append(
                    CrowdDatasetBBox(
                        debug=0,
                        learn_worker_params=True,
                        learn_image_params=True,
                        estimate_priors_automatically=True,
                        computer_vision_predictor=None,
                        naive_computer_vision=False,
                        min_risk=self.maxRisk,
                    )
                )
                self.fullSavePrefixes.append(
                    os.path.join(
                        self.savePath, "{}_T{}".format(self.savePrefix, taskCounter)
                    )
                )
                self.subAggregators[-1].fname = self.fullSavePrefixes[-1]

        except TypeError as e:
            print(
                "The 'taskLabels' argument must be an iterable containing task label strings e.g. ['T0', 'T1']"
            )
            raise

        self.allUniqueMessages = []
        self.inputAnnotations = (
            {taskLabel: [] for taskLabel in self.taskLabels}
            if self.saveInputAnnotations
            else None
        )
        self.inputMessages = [] if self.saveInputMessages else None
        self.deleteMessagesFromQueue = kwargs.get("deleteMessagesFromQueue", True)

        self.verbose = kwargs.get("verbose", False)

        # Register interrupt handler.
        signal.signal(signal.SIGINT, self.stopLoopHandler)

        if purgeOldBBoxSetFile:
            self.purgeBBoxSetFile()

    def purgeBBoxSetFile(self):
        for fullSavePrefix in self.fullSavePrefixes:
            bboxSetFilePath = fullSavePrefix + ".big_bbox_set.pkl"
            if os.path.exists(bboxSetFilePath):
                os.remove(bboxSetFilePath)

    def accumulateMessages(self):
        while len(self.allUniqueMessages) < self.messageBatchSize:
            uniqueMessages, allMessages, messageIds = self.sqsClient.getMessages(
                self.deleteMessagesFromQueue
            )
            if not len(uniqueMessages):
                print(
                    "Aggregator: accumulateMessages: No messages extracted from queue. Accumulation stops"
                )
                return False
            self.allUniqueMessages.extend(uniqueMessages)

        self.allUniqueMessages = self.sqsClient.deduplicate(self.allUniqueMessages)

        return True

    def aggregate(self):
        if not self.accumulateMessages() and len(self.allUniqueMessages) == 0:
            # If no messages are available for processing
            return False
        else:
            print(
                "Aggregator: aggregate: Accumulated",
                len(self.allUniqueMessages),
                " messages.",
            )

        for taskLabel, aggregator, sqsMessageParser in zip(
            self.taskLabels, self.subAggregators, self.sqsMessageParsers
        ):
            if not sqsMessageParser.processMessages(self.allUniqueMessages):
                return False

            if self.saveInputAnnotations:
                self.inputAnnotations[taskLabel].extend(
                    sqsMessageParser.aggregatorInputData["annos"]
                )

            aggregator.load(
                data=sqsMessageParser.getAggregatorInputData(), overwrite_workers=False
            )
            aggregator.get_big_bbox_set()
            aggregator.estimate_parameters(
                avoid_if_finished=True, max_iters=10, refine=True
            )

        if self.saveInputMessages:
            self.inputMessages.extend(self.allUniqueMessages)
        self.allUniqueMessages = []
        return True

    def checkNumFinished(self):
        for taskLabel, aggregator in zip(self.taskLabels, self.subAggregators):
            image_id_to_finished = aggregator.check_finished_annotations(
                set_finished=True
            )
            num_finished = sum(image_id_to_finished.values())
            if len(image_id_to_finished) > 0:
                print(
                    "Task {}: {:d} / ({:d}) ({:.2f}%) images are finished".format(
                        taskLabel,
                        num_finished,
                        len(image_id_to_finished),
                        100.0 * float(num_finished) / len(image_id_to_finished),
                    )
                )

    def save(self):
        for aggregator, fullSavePrefix in zip(
            self.subAggregators, self.fullSavePrefixes
        ):
            aggregator.save(
                fullSavePrefix + "_aggregated.json",
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
        retries = 0
        while True:
            if self.aggregate():
                if verbose:
                    self.checkNumFinished()
            elif not stopOnExhaustion:
                print("No messages received. Waiting...")
                continue
            elif retries < 3:
                print(
                    "No messages received after {} retries. Retrying...".format(retries)
                )
                retries += 1
                self.checkNumFinished()
            else:
                print(
                    "No messages received after {} retries. Stopping.".format(retries)
                )
                self.checkNumFinished()
                break
