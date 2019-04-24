import boto3
import pandas as pd
import json
import copy


class DummyMessageStructures:

    caesarFullPluckFieldStructure = dummyMessage = {
        "id": None,
        "classification_id": None,
        "classification_at": None,
        "user_id": None,
        "subject_id": None,
        "data": {
            "classification": {
                "id": None,
                "subject": {
                    "id": None,
                    "metadata": {},
                    "created_at": "DUMMY",
                    "updated_at": "DUMMY",
                },
                "user_id": None,
                "metadata": {
                    "source": "api",
                    "session": "DUMMY",
                    "viewport": {"width": 0, "height": 0},
                    "started_at": "2000-01-01T00:00:0.0Z",
                    "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_2) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/12.0.2 Safari/605.1.15",
                    "utc_offset": "18000",
                    "finished_at": "2000-01-01T00:00:0.0Z",
                    "seen_before": False,
                    "live_project": False,
                    "interventions": {"opt_in": True, "message": False},
                    "user_language": "en",
                    "user_group_ids": [],
                    "workflow_version": "19.22",
                    "subject_dimensions": [
                        {
                            "clientWidth": 0,
                            "clientHeight": 0,
                            "naturalWidth": 0,
                            "naturalHeight": 0,
                        }
                    ],
                    "subject_selection_state": {
                        "retired": False,
                        "selected_at": "2000-01-01T00:00:0.0Z",
                        "already_seen": True,
                        "selection_state": "failover_fallback",
                        "finished_workflow": False,
                        "user_has_finished_workflow": True,
                    },
                    "workflow_translation_id": "0000",
                },
                "created_at": None,
                "project_id": 0000,
                "subject_id": None,
                "updated_at": None,
                "annotations": {},
                "workflow_id": None,
                "workflow_version": "19.22",
            }
        },
        "project_id": None,
    }


class SQSMessageGenerator:
    def __init__(
        self,
        panoptesDataExport=None,
        dummyMessageStructure=DummyMessageStructures.caesarFullPluckFieldStructure,
    ):
        self.panoptesDataExport = panoptesDataExport
        self.dummyMessageStructure = dummyMessageStructure
        self.panoptesData = None

    def parsePanoptesExport(self):
        if self.panoptesDataExport is not None:
            self.panoptesData = pd.read_csv(
                self.panoptesDataExport,
                converters={
                    "metadata": json.loads,
                    "subject_data": json.loads,
                    "annotations": json.loads,
                },
            )

    def generateMessages(self, numMessages=None):
        messages = [
            self.generateMessage(row)
            for row in self.panoptesData.loc[slice(0, numMessages), :].iterrows()
        ]
        return messages

    def generateMessage(self, messageData):
        rowIndex, rowData = messageData
        message = copy.deepcopy(self.dummyMessageStructure)
        message.update(
            dict(
                id=rowIndex,
                classification_id=rowData.classification_id,
                classification_at=rowData.created_at,
                user_id=rowData.user_id,
                subject_id=rowData.subject_ids,
            )
        )

        message["data"]["classification"].update(
            dict(
                id=rowIndex,
                classification_id=rowData.classification_id,
                classification_at=rowData.created_at,
                created_at=rowData.created_at,
                user_id=rowData.user_id,
                subject_id=rowData.subject_ids,
                workflow_id=rowData.workflow_id,
            )
        )

        message["data"]["classification"]["metadata"].update(rowData.metadata)
        message["data"]["classification"]["annotations"] = rowData.annotations
        message["data"]["classification"]["subject"]["id"] = rowData.subject_ids

        return message
