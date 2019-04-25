import os

from caesar_external import data as cedata
from caesar_external import utils as ceutils


class CaesarClientWrapper:
    def __init__(
        self, name, projectId, workflowId, caesarName, sqsQueue, isStagingMode, authMode
    ):

        configKwargs = {
            "name": name,
            "project": projectId,
            "workflow": workflowId,
            "sqs_queue": sqsQueue,
            "staging_mode": isStagingMode,
            "last_id":-1
        }

        if authMode == "api_key":
            configKwargs.update(
                {
                    "client_id": os.environ.get("PANOPTES_CLIENT_ID"),
                    "client_secret": os.environ.get("PANOPTES_CLIENT_SECRET"),
                }
            )

        if caesarName is not None:
            configKwargs.update({"caesar_name": caesarName})

        self.caesarConfig = cedata.Config(**configKwargs)
        self.caesarClient = ceutils.caesar_utils.Client()

        def sendReduction(self, subjectId, reduction):
            self.caesarClient.reduce(subjectId, reduction)
