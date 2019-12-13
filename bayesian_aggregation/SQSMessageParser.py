import numpy as np
import pandas as pd
import itertools


class SQSMessageParser:

    defaultMarkWidth = 35
    defaultMarkHeight = 35

    def __init__(self, **kwargs):

        self.markScaleFactor = kwargs.get("markScaleFactor", 1.0)
        print("SQSMessageParser.markScaleFactor", self.markScaleFactor)
        self.setMarkDimensions(**kwargs)

        self.filterSubjectList = kwargs.get("filterSubjectList", [])

        self.taskLabel = kwargs.get("taskLabel", "T1")
        self.allClassificationIds = set()
        self.processedClassifications = None
        self.aggregatorInputData = dict()
        self.subjectMetadataFilter = kwargs.get("subjectMetadataFilter", lambda x: True)

    def setMarkDimensions(self, **kwargs):
        self.markWidth = None
        self.markHeight = None

        self.sizeMetaDatumName = kwargs.get("sizeMetaDatumName", None)
        self.widthMetaDatumName = kwargs.get("widthMetaDatumName", None)
        self.heightMetaDatumName = kwargs.get("heightMetaDatumName", None)

        if not self.widthMetaDatumName:
            self.widthMetaDatumName = self.sizeMetaDatumName
        if not self.heightMetaDatumName:
            self.heightMetaDatumName = self.sizeMetaDatumName

        if not (
            self.markWidth
            or self.markHeight
            or self.sizeMetaDatumName
            or (self.widthMetaDatumName and self.heightMetaDatumName)
        ):
            print(
                "No marking dimension metadata specified. Using explicit values with ({}, {}) pixel default".format(
                    SQSMessageParser.defaultMarkWidth,
                    SQSMessageParser.defaultMarkHeight,
                )
            )
            self.markWidth = kwargs.get("markWidth", SQSMessageParser.defaultMarkWidth)
            self.markHeight = kwargs.get(
                "markHeight", SQSMessageParser.defaultMarkHeight
            )

    def extractSubjectDimensions(self, metadata):
        try:
            return metadata["subject_dimensions"][0]
        except IndexError as e:
            # print("IndexError parsing subject dimensions.", e)
            return None

    def extractBoxWidths(self, subject):
        try:
            if self.markWidth is not None:
                return float(self.markWidth) * self.markScaleFactor
            return (
                float(subject["metadata"][self.widthMetaDatumName])
                * self.markScaleFactor
            )
        except KeyError as e:
            print(
                "SQSMessageParser.extractBoxWidths: KeyError. Could not find {}. Using default box size ({} pixels)".format(
                    self.widthMetaDatumName, SQSMessageParser.defaultMarkWidth
                )
            )
            return SQSMessageParser.defaultMarkWidth * self.markScaleFactor
        except TypeError as e:
            print("SQSMessageParser.extractBoxWidths: TypeError")
            return SQSMessageParser.defaultMarkWidth * self.markScaleFactor

    def extractBoxHeights(self, subject):
        try:
            if self.markHeight is not None:
                return float(self.markHeight) * self.markScaleFactor
            return (
                float(subject["metadata"][self.heightMetaDatumName])
                * self.markScaleFactor
            )
        except KeyError as e:
            print("SQSMessageParser.extractBoxHeights: KeyError", e)
            return SQSMessageParser.defaultMarkHeight * self.markScaleFactor
        except TypeError as e:
            print("SQSMessageParser.extractBoxHeights: TypeError", e)
            return SQSMessageParser.defaultMarkHeight * self.markScaleFactor

    def imageDimsToTuple(self, imageDims):
        try:
            return (imageDims["naturalWidth"], imageDims["naturalHeight"])
        except TypeError as e:
            print(e)
            return (400, 400)  ### VM-edit; fixing the image dimensions

    def extractClassification(self, message):
        try:
            return message["data"]["classification"]
        except TypeError:
            return None
        except IndexError:
            return None

    def processMessages(self, uniqueMessages):
        classifications = self.extractClassifications(uniqueMessages)
        if classifications is not None:
            self.processClassifications(classifications)
            self.genAggregatorInput()
            return True
        else:
            return False

    def extractClassifications(self, uniqueMessages):

        classificationData = list(
            filter(
                lambda x: x,
                [
                    self.extractClassification(message)
                    for message in uniqueMessages
                    if message["classification_id"] not in self.allClassificationIds
                    and self.subjectMetadataFilter(
                        message["data"]["classification"]["subject"]["metadata"]
                    )
                ],
            )
        )

        if self.allClassificationIds.intersection(
            set([message["classification_id"] for message in uniqueMessages])
        ):
            duplicateMessageIds = [
                message["classification_id"]
                for message in uniqueMessages
                if message["classification_id"] in self.allClassificationIds
            ]
            print(
                "Message Parser: Duplicate classification IDs: {}".format(
                    duplicateMessageIds
                )
            )

        self.allClassificationIds.update(
            [message["classification_id"] for message in uniqueMessages]
        )

        if len(classificationData):
            classificationsFrame = pd.DataFrame(classificationData)
            classificationsFrame.loc[
                ~np.isfinite(classificationsFrame.user_id), "user_id"
            ] = -99
            classificationsFrame = classificationsFrame.astype({"user_id": int})
        else:
            print("All messages already seen.")
            return None

        print(
            "Task {}: Processed {} classifications after deduplication ({} duplicate classification IDs). The following fields were extracted:".format(
                self.taskLabel,
                classificationsFrame.id.unique().size,
                classificationsFrame.id.size - classificationsFrame.id.unique().size,
            )
        )
        print("[", ", ".join(classificationsFrame.columns), end="]\n")

        return classificationsFrame.drop_duplicates(subset="id")

    def processClassifications(self, classificationsFrame):
        print("Unique subject count", classificationsFrame.subject_id.unique().size)
        classificationsFrame["annotation_counts"] = classificationsFrame.loc[
            :, "annotations"
        ].apply(len)

        classificationsFrame["have_markings"] = (
            classificationsFrame.annotation_counts > 0
        )

        markedClassificationsFrame = classificationsFrame[
            classificationsFrame.have_markings
        ].reindex()

        markedClassificationsFrame = markedClassificationsFrame[
            markedClassificationsFrame.annotations.apply(lambda x: self.taskLabel in x)
        ]

        markedClassificationsFrame[
            "markings"
        ] = markedClassificationsFrame.annotations.apply(
            lambda x: [(mark["x"], mark["y"]) for mark in x[self.taskLabel][0]["value"]]
        )

        markedClassificationsFrame[
            "tool"
        ] = markedClassificationsFrame.annotations.apply(
            lambda x: [mark["tool"] for mark in x[self.taskLabel][0]["value"]]
        )

        markedClassificationsFrame[
            "num_markings"
        ] = markedClassificationsFrame.markings.apply(lambda x: len(x))

        markedClassificationsFrame[
            "image_dimensions"
        ] = markedClassificationsFrame.metadata.apply(self.extractSubjectDimensions)

        markedClassificationsFrame[
            "box_widths"
        ] = markedClassificationsFrame.subject.apply(self.extractBoxWidths)

        markedClassificationsFrame[
            "box_heights"
        ] = markedClassificationsFrame.subject.apply(self.extractBoxHeights)

        markedClassificationsFrame[
            "image_dimensions"
        ] = markedClassificationsFrame.image_dimensions.fillna(method="ffill").apply(
            self.imageDimsToTuple
        )

        # On Zooniverse mobile some taps can be registered multiple times.
        # Attempt to filter these taps.
        markedClassificationsFrame[
            "marking_separations"
        ] = markedClassificationsFrame.markings.apply(
            lambda x: [
                (
                    (firstId, secondId),
                    np.hypot(
                        firstMark[0] - secondMark[0], firstMark[1] - secondMark[1]
                    ),
                )
                for (firstId, firstMark), (
                    secondId,
                    secondMark,
                ) in itertools.combinations(enumerate(x), 2)
            ]
        )

        markedClassificationsFrame[
            "original_markings"
        ] = markedClassificationsFrame.markings

        markedClassificationsFrame["unique_mark_indices"] = markedClassificationsFrame[
            [
                "marking_separations",
                "markings",
                "num_markings",
                "box_widths",
                "box_heights",
            ]
        ].apply(
            lambda x: list(range(len(x.markings)))
            if x.num_markings <= 1
            else np.unique(
                [x.marking_separations[0][0][0]]
                + [
                    idPair[1]
                    for (idPair, distance) in x.marking_separations
                    if distance > 0.75 * (x.box_widths + x.box_heights)
                ]
            ).tolist(),
            axis=1,
        )

        markedClassificationsFrame.markings = markedClassificationsFrame[
            ["markings", "unique_mark_indices", "num_markings"]
        ].apply(
            lambda x: x.markings
            if x.num_markings <= 1
            else np.array(x.markings)[x.unique_mark_indices].tolist(),
            axis=1,
        )

        markedClassificationsFrame[
            "num_filtered_markings"
        ] = markedClassificationsFrame.markings.apply(lambda x: len(x))

        if self.processedClassifications is None:
            self.processedClassifications = markedClassificationsFrame
        else:
            self.processedClassifications = pd.concat(
                [self.processedClassifications, markedClassificationsFrame]
            )

        self.processedClassifications.drop_duplicates(subset="id", inplace=True)

        # Spammer alarm:
        # 1) If a single user submits the majority of classifications
        # in a batch, suspect that they are spammers.
        # 2) Check how many marks per subject - if average > 10 strongly
        # suspect spammers!
        if self.taskLabel == "T1":
            userGroupedCF = self.processedClassifications.groupby(by="user_id")
            userClassificationCounts = userGroupedCF.count()
            userMeanMarkCounts = userGroupedCF.mean().num_markings
            userMeanFilteredMarkCounts = userGroupedCF.mean().num_filtered_markings
            userMedianMarkCounts = userGroupedCF.median().num_markings
            print(
                "USER COUNTS:",
                pd.concat(
                    [
                        userClassificationCounts.id,
                        userMeanMarkCounts,
                        userMeanFilteredMarkCounts,
                        userMedianMarkCounts,
                        userClassificationCounts.id
                        > 0.15 * len(self.processedClassifications.index),
                        userMeanMarkCounts > 3,
                        (
                            userClassificationCounts.id
                            > 0.15 * len(self.processedClassifications.index)
                        )
                        & (userMeanMarkCounts > 3),
                    ],
                    axis=1,
                ),
            )

    def genAggregatorInput(self):
        dataset = dict()
        workers = dict()
        images = {
            str(rowData.subject_id): {
                "height": rowData.image_dimensions[1],
                "width": rowData.image_dimensions[0],
                "url": "",
            }
            for _, rowData in self.processedClassifications.drop_duplicates(
                subset="subject_id"
            ).iterrows()
            if rowData.subject_id not in self.filterSubjectList
        }

        annos = [
            {
                "anno": {
                    "bboxes": [
                        {
                            "image_height": rowData.image_dimensions[1],
                            "image_width": rowData.image_dimensions[0],
                            "x": markData[0] - 0.5 * rowData.box_heights,
                            "x2": markData[0] + 0.5 * rowData.box_widths,
                            "y": markData[1] - 0.5 * rowData.box_heights,
                            "y2": markData[1] + 0.5 * rowData.box_widths,
                            "tool" : tool
                        }
                        for markData, tool in zip(rowData.markings, rowData.tool)
                    ]
                },
                "image_id": str(rowData.subject_id),
                "worker_id": str(rowData.user_id),
            }
            for _, rowData in self.processedClassifications.iterrows()
            if rowData.subject_id not in self.filterSubjectList
        ]

        self.aggregatorInputData = dict(
            dataset=dataset, workers=workers, images=images, annos=annos
        )
        # print(self.aggregatorInputData)

    def getAggregatorInputData(self):
        return self.aggregatorInputData

    def getNumProcessedClassifications(self):
        if self.processedClassifications is None:
            return 0
        return len(self.processedClassifications.index)

    def clearProcessedClassifications(self):
        if self.processedClassifications is not None:
            del self.processedClassifications
            self.processedClassifications = None
