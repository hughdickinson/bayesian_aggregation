import pandas as pd


class SQSMessageParser:
    def __init__(self, **kwargs):

        self.markWidth = kwargs.get("markWidth", 35)
        self.markHeight = kwargs.get("markHeight", 35)
        self.allClassificationIds = set()
        self.processedClassifications = None
        self.aggregatorInputData = dict()

    def extractSubjectDimensions(self, metadata):
        try:
            return metadata["subject_dimensions"][0]
        except IndexError as e:
            print(e)
            return None

    def imageDimsToTuple(self, imageDims):
        try:
            return (imageDims["clientWidth"], imageDims["clientHeight"])
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
        self.processClassifications(classifications)
        self.genAggregatorInput()

    def extractClassifications(self, uniqueMessages):
        classificationData = list(
            filter(
                lambda x: x,
                [
                    self.extractClassification(message)
                    for message in uniqueMessages
                    if message["classification_id"] not in self.allClassificationIds
                ],
            )
        )

        for message in uniqueMessages:
            if message["classification_id"] in self.allClassificationIds:
                print("Duplicate classification ID: {}".format(message["classification_id"]))

        self.allClassificationIds.update([message["id"] for message in uniqueMessages])

        classificationsFrame = pd.DataFrame(classificationData)

        print(
            "Processed {} classifications. The following fields were extracted:".format(
                len(classificationsFrame.index)
            ),
            *classificationsFrame.columns,
            sep="\n"
        )

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

        markedClassificationsFrame[
            "markings"
        ] = markedClassificationsFrame.annotations.apply(
            lambda x: [(mark["x"], mark["y"]) for mark in x[0]["value"]]
        )

        markedClassificationsFrame[
            "image_dimensions"
        ] = markedClassificationsFrame.metadata.apply(self.extractSubjectDimensions)

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
                            "x": markData[0] - 0.5 * self.markWidth,
                            "x2": markData[0] + 0.5 * self.markWidth,
                            "y": markData[1] - 0.5 * self.markHeight,
                            "y2": markData[1] + 0.5 * self.markHeight,
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

    def getAggregatorInputData(self):
        return self.aggregatorInputData

    def getNumProcessedClassifications(self):
        if self.processedClassifications is None:
            return 0
        return len(self.processedClassifications.index)
