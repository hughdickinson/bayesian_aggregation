import pandas as pd


class SQSMessageParser:

    defaultMarkWidth = 35
    defaultMarkHeight = 35

    def __init__(self, **kwargs):

        self.setMarkDimensions(**kwargs)

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
            print("IndexError parsing subject dimensions.", e)
            return None

    def extractBoxWidths(self, subject):
        if self.markWidth is not None:
            return self.markWidth
        try:
            return subject["metadata"][self.widthMetaDatumName]
        except KeyError as e:
            print(
                "KeyError. Could not find {}. Using default box size ({} pixels)".format(
                    self.widthMetaDatumName, SQSMessageParser.defaultMarkWidth
                ),
                subject["metadata"],
            )
            return SQSMessageParser.defaultMarkWidth

    def extractBoxHeights(self, subject):
        if self.markWidth is not None:
            return self.markHeight
        try:
            return subject["metadata"][self.heightMetaDatumName]
        except KeyError as e:
            print("KeyError", e)
            return SQSMessageParser.defaultMarkHeight

    def imageDimsToTuple(self, imageDims):
        try:
            return (imageDims["naturalWidth"], imageDims["naturalHeight"])
        except TypeError as e:
            print(e)
            return (1, 1)

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

        # print(markedClassificationsFrame)
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

        if self.processedClassifications is None:
            self.processedClassifications = markedClassificationsFrame
        else:
            self.processedClassifications = pd.concat(
                [self.processedClassifications, markedClassificationsFrame]
            )

        self.processedClassifications.drop_duplicates(subset="id")

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
                        }
                        for markData in rowData.markings
                    ]
                },
                "image_id": str(rowData.subject_id),
                "worker_id": str(rowData.user_id),
            }
            for _, rowData in self.processedClassifications.iterrows()
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
