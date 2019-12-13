from .SQSClient import SQSClient, SQSOfflineClient
from .SQSMessageParser import SQSMessageParser
from .BBoxResultsPlotter import BBoxResultsPlotter

import importlib
import signal
import sys
import os
import time
import pickle

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
        postIterateCallback=None,
        offlineMode=False,
        offlineMessageDump=None,
        saveIntermittently=True,
        removeAnonUsers=False,
        crowdsourcing_kwargs={},
        **kwargs
    ):

        self.savePath = savePath
        self.savePrefix = savePrefix

        self.crowdsourcing_kwargs = crowdsourcing_kwargs

        self.removeAnonUsers = removeAnonUsers
        self.saveIntermittently = saveIntermittently
        self.offlineMode = offlineMode
        self.offlineMessageDump = offlineMessageDump

        if self.offlineMode:
            self.sqsClient = SQSOfflineClient(
                filename=self.offlineMessageDump,
                sizeMetaDatumName=kwargs.get("sizeMetaDatumName", "#fwhmImagePix"),
                trainingMessagesOnly=kwargs.get("trainingMessagesOnly", False),
                removeAnonUsers=self.removeAnonUsers,
            )
        else:
            self.sqsClient = SQSClient(queueUrl=queueUrl, **kwargs)

        self.saveInputAnnotations = saveInputAnnotations
        self.saveInputMessages = saveInputMessages

        self.maxRisk = maxRisk

        self.messageBatchSize = messageBatchSize
        self.postIterateCallback = postIterateCallback

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
                        estimate_priors_automatically=False,
                        computer_vision_predictor=None,
                        naive_computer_vision=False,
                        min_risk=self.maxRisk,
                    )
                )
                self.fullSavePrefixes.append(
                    os.path.join(
                        self.savePath, "{}_{}".format(self.savePrefix, taskLabel)
                    )
                )
                self.subAggregators[-1].fname = self.fullSavePrefixes[-1]

                for k, v in self.crowdsourcing_kwargs.items():
                    if hasattr(self.subAggregators[-1], k):
                        setattr(self.subAggregators[-1], k, v)
                    else:
                        print(
                            "Warning: No {} attribute found for CrowdDatasetBBox. Ignoring.".format(
                                k
                            )
                        )
        except TypeError as e:
            print(
                "TypeError raised while initialising sub-aggregators.",
                e,
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
                delete=self.deleteMessagesFromQueue
            )
            if not len(uniqueMessages):
                print(
                    "Aggregator: accumulateMessages: No messages extracted from queue. Accumulation stops"
                )
                return False
            self.allUniqueMessages.extend(uniqueMessages)

        if not self.offlineMode:
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
            if not sqsMessageParser.processMessages(
                uniqueMessages=self.allUniqueMessages
            ):
                return False

            if self.saveInputAnnotations:
                self.inputAnnotations[taskLabel].extend(
                    sqsMessageParser.aggregatorInputData["annos"]
                )

            aggInput = sqsMessageParser.getAggregatorInputData()
            print(
                "Agg input for batch: Num annos = {}, Num images = {}".format(
                    len(aggInput["annos"]), len(aggInput["images"])
                ),
                flush=True,
            )
            aggregator.load(
                data=aggInput,
                overwrite_workers=False,
                load_workers=False,
                load_images=False,
                load_dataset=False,
                clear_previous_image_annos=False,
            )
            aggregator.get_big_bbox_set()
            aggregator.estimate_parameters(
                avoid_if_finished=False, max_iters=25, refine=True
            )
            sqsMessageParser.clearProcessedClassifications()

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

        if self.saveInputMessages:
            self.dumpInputMessages()

    def dumpInputMessages(self):
        inputMessageFileName = "{}_inputMessages.pkl".format(self.savePrefix)
        with open(inputMessageFileName, mode="wb") as inputMessageFile:
            pickle.dump(obj=self.inputMessages, file=inputMessageFile)

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
                        sys.exit(0)
                    elif saveResponse == "n":
                        sys.exit(0)
                    else:
                        print("Unrecognized response.")
            elif quitResponse == "n":
                self.loop()
            else:
                print("Unrecognized response.")

    def loop(
        self,
        verbose=True,
        stopOnExhaustion=False,
        plotInterrimResults=False,
        interrimPlotDir=None,
    ):
        retries = 0
        n_loop = 0
        while True:
            if self.aggregate():
                if plotInterrimResults:
                    for taskLabel, aggregator in zip(
                        self.taskLabels, self.subAggregators
                    ):
                        BBoxResultsPlotter.plotUserData(
                            aggregator,
                            interrimPlotDir,
                            "userSkills_{}_{}".format(taskLabel, n_loop),
                        )
                        # with open(os.path.join(self.savePath, "userSkillData", "userSkills_{}_{}.pkl".format(taskLabel, n_loop)), mode="wb") as skillFile:
                        #     pickle.dump(obj=aggregator.workers, file=skillFile)
                if verbose:
                    self.checkNumFinished()
                    if self.postIterateCallback is not None:
                        self.postIterateCallback(
                            {
                                taskLabel: {
                                    "data": subAgg.save(fname=None),
                                    "finished_id_map": subAgg.check_finished_annotations(
                                        set_finished=True
                                    ),
                                }
                                for taskLabel, subAgg in zip(
                                    self.taskLabels, self.subAggregators
                                )
                            }
                        )
                if self.saveIntermittently and not (n_loop % 10):
                    self.save()
                self.purgeBBoxSetFile()
                n_loop += 1
            elif not stopOnExhaustion:
                print("No messages received. Waiting...")
                if self.offlineMode:
                    time.sleep(60)
                    self.sqsClient.update()
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
