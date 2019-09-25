import os
import hashlib
import boto3
import json
import pickle
import numpy as np
import astropy.io.fits as fitsio

class UniqueMessage:
    def __init__(self, message):
        self.classification_id = int(message["classification_id"])
        self.message = message

    def __eq__(self, other):
        return self.classification_id == other.classification_id

    def __hash__(self):
        return hash(self.classification_id)


class SQSClient:
    def __init__(self, queueUrl, **kwargs):
        self.sqs = boto3.client("sqs")
        self.queueUrl = queueUrl
        self.subscribers = []
        if kwargs.get("verbose", False):
            print(
                "SQS Queue Attributes",
                self.sqs.get_queue_attributes(
                    QueueUrl=self.queueUrl, AttributeNames=["All"]
                ),
            )

    def getMessages(self, delete=True):
        response = self.sqs.receive_message(
            QueueUrl=self.queueUrl,
            AttributeNames=["SentTimestamp", "MessageDeduplicationId"],
            MaxNumberOfMessages=10,  # Allow up to 10 messages to be received
            MessageAttributeNames=["All"],
            # Allows the message to be retrieved again after 40s
            VisibilityTimeout=40,
            # Wait at most 20 seconds for an extract enables long polling
            WaitTimeSeconds=20,
        )

        receivedMessageIds = []
        receivedMessages = []
        uniqueMessages = set()

        # Loop over messages
        if "Messages" in response:
            for message in response["Messages"]:
                # extract message body expect a JSON formatted string
                # any information required to deduplicate the message should be
                # present in the message body
                messageBody = message["Body"]
                # verify message body integrity
                messageBodyMd5 = hashlib.md5(messageBody.encode()).hexdigest()

                if messageBodyMd5 == message["MD5OfBody"]:
                    receivedMessages.append(json.loads(messageBody))
                    receivedMessageIds.append(receivedMessages[-1]["classification_id"])
                    uniqueMessage = UniqueMessage(receivedMessages[-1])

                    uniqueMessages.add(uniqueMessage)

                    if delete:
                        self.sqs.delete_message(
                            QueueUrl=self.queueUrl,
                            ReceiptHandle=message["ReceiptHandle"],
                        )
                else:
                    print("MD5 mismatch!")

        messages = [m.message for m in uniqueMessages]
        return messages, receivedMessages, receivedMessageIds

    def putMessages(self, messages, purge=False):
        if purge:
            sqsResource = boto3.resource("sqs")
            queue = sqsResource.Queue(self.queueUrl)
            queue.purge()

        for message in messages:
            if type(message) == dict:
                self.sqs.send_message(
                    QueueUrl=self.queueUrl, MessageBody=json.dumps(message)
                )
        print(
            'SQSClient posted {} messages to "{}""'.format(len(messages), self.queueUrl)
        )

    def deduplicate(self, messageList):
        return [
            um.message
            for um in set([UniqueMessage(message) for message in messageList])
        ]

class SQSOfflineClient:
    """
    Added by VM to facilitate parsing offline using downloaded datadump
    """
    def __init__(self, filename, sizeMetaDatumName="#fwhmImagePix", trainingMessagesOnly=False, removeAnonUsers=False, **kwargs):

        self.messagesFilename = filename

        self.mTimes = {}
        self.allMessages = []
        self.parsedCount = 0
        self.removeAnonUsers = removeAnonUsers
        self.trainingMessagesOnly = trainingMessagesOnly
        self.sizeMetaDatumName = sizeMetaDatumName

        if os.path.isfile("datastore/trainingFWHM.fits"):
            self.trainingFWHM = fitsio.getdata("datastore/trainingFWHM.fits")
        else:
            self.trainingFWHM = None

        self.loadMessages()

    def loadMessages(self):

        for filename in np.atleast_1d(self.messagesFilename):
            with open(filename,'rb') as pklfile:
                messages = pickle.load(pklfile)
                if self.removeAnonUsers:
                    messages = [x for x in messages if x["user_id"] is not None]
                if self.trainingMessagesOnly:
                    messages = [x for x in messages if x["data"]["classification"]["subject"]["metadata"]["origin"]=="training"]
                self.allMessages.extend(messages)
            self.mTimes[filename] = os.stat(filename).st_mtime

        self.messageIds = np.arange(len(self.allMessages))
        print("SQSOfflineClient: Loaded {} messages ...".format(len(self.allMessages)))

    def update(self):

        update = []
        for filename in np.atleast_1d(self.messagesFilename):
            update.append(os.stat(filename).st_mtime > self.mTimes[filename])

        if any(update):

            allMessages = []
            for filename in np.atleast_1d(self.messagesFilename):
                with open(filename,'rb') as pklfile:
                    messages = pickle.load(pklfile)
                    if self.removeAnonUsers:
                        messages = [x for x in messages if x["user_id"] is not None]
                    if self.trainingMessagesOnly:
                        messages = [x for x in messages if x["data"]["classification"]["subject"]["metadata"]["origin"]=="training"]
                    allMessages.extend(messages)
                self.mTimes[filename] = os.stat(filename).st_mtime

            cond = np.in1d([_["classification_id"] for _ in allMessages],
                           [_["classification_id"] for _ in self.allMessages])
            cidx = np.where(~cond)[0]
            print("SQSOfflineClient: Updated with {} new messages ...".format(len(cidx)))

            self.allMessages.extend([allMessages[idx] for idx in cidx])
            self.messageIds = np.arange(len(self.allMessages))

    def addTrainingFWHM(self, messages):

        if self.trainingFWHM is not None:

            for message in messages:

                metadata = message["data"]["classification"]["subject"]["metadata"]
                if self.sizeMetaDatumName in metadata:
                    pass
                elif metadata["id"] in self.trainingFWHM["id"].astype(str):
                    idx = np.where(metadata["id"] == self.trainingFWHM["id"].astype(str))[0][0]
                    metadata[self.sizeMetaDatumName] = self.trainingFWHM["fwhmImagePix"][idx]
                else:
                    print("No {} found in message or additional metadata for ID#{}".format(self.sizeMetaDatumName,metadata["id"]))
                    pass

        return messages

    def getMessages(self, batchSize=None, delete=None):

        maxCount = len(self.allMessages)
        if self.parsedCount < maxCount:

            if batchSize is None: batchSize = np.random.randint(40,60)
            batchIds = self.messageIds[self.parsedCount:self.parsedCount+batchSize]

            messages = [self.allMessages[i] for i in batchIds]
            messages = self.addTrainingFWHM(messages)

            receivedMessages = messages
            receivedMessageIds = [m["classification_id"] for m in messages]

            self.parsedCount += batchSize

            print("***** Served {}/{} classifications *****".format(self.parsedCount,maxCount))
            return messages, receivedMessages, receivedMessageIds

        else:

            return [], [], []
