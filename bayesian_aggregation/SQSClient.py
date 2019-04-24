import hashlib
import boto3
import json


class UniqueMessage(object):
    def __init__(self, message):
        self.classification_id = int(message["classification_id"])
        self.message = message

    def __eq__(self, other):
        return self.classification_id == other(self.classification_id)

    def __hash__(self):
        return hash(self.classification_id)


class SQSClient:
    def __init__(self, queueUrl):
        self.sqs = boto3.client("sqs")
        self.queueUrl = queueUrl

    def getMessages(self):
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
                    uniqueMessages.add(UniqueMessage(receivedMessages[-1]))
                    # the message has been retrieved successfully - delete it.
                    self.sqs.delete_message(
                        QueueUrl=self.queueUrl, ReceiptHandle=message["ReceiptHandle"]
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
