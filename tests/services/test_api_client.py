import uuid
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from a4s_eval.service.api_client import get_evaluation

TEST_UUIDS = {
    "project": uuid.UUID("afb49e3f-813d-8888-9919-ee179d1090e6"),
    "evaluation": uuid.UUID("afb49e3f-813d-4260-9919-ee179d1090e6"),
    "test_dataset": uuid.UUID("750ec557-fd8c-4f94-92b9-28796591fd40"),
    "train_dataset": uuid.UUID("d830c991-88a5-4cb3-906a-9aa6e2a8c63f"),
    "model": uuid.UUID("08947de5-bcc2-4fee-8717-deb3f8bc86f9"),
}


@pytest.fixture
def mock_evaluation_data() -> dict[str, any]:
    metadata = pd.read_csv("tests/data/lcld_v2_metadata_api.csv")
    metadata["feature_type"] = metadata["feature_type"].str.capitalize()
    metadata = metadata.to_dict(orient="records")

    for record in metadata:
        record["pid"] = uuid.uuid4()

    data_shape = {
        "features": [
            item
            for item in metadata
            if item.get("Name") not in ["charged_off", "issue_d"]
        ],
        "target": next(rec for rec in metadata if rec.get("name") == "charged_off"),
        "date": next(rec for rec in metadata if rec.get("name") == "issue_d"),
    }

    return {
        "pid": str(TEST_UUIDS["evaluation"]),
        "dataset": {"pid": str(TEST_UUIDS["test_dataset"]), "shape": data_shape},
        "model": {
            "pid": str(TEST_UUIDS["model"]),
            "dataset": {
                "pid": str(TEST_UUIDS["train_dataset"]),
                "shape": data_shape,
            },
        },
        "project": {
            "pid": str(TEST_UUIDS["project"]),
            "name": "LCLD",
            "frequency": "1 day",
            "window_size": "1 month",
            "plugins": [{"name": "ClassificationPerformancePlugin"}],
        },
        "evaluation_plugins": [
            {
                "name": "ClassificationPerformancePlugin",
                "config": {},
                "dataset_filename": None,
                "model_filename": None,
            },
        ],
    }


def test_get_evaluation(mock_evaluation_data: dict[str, any]) -> None:
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_evaluation_data
    with (
        patch("requests.get", return_value=mock_response),
        patch("a4s_eval.service.api_client.log_audit_event"),
    ):
        evaluation = get_evaluation(TEST_UUIDS["evaluation"])

        # Now evaluation should be an EvaluationDto object built from mock_response
        assert evaluation.pid == TEST_UUIDS["evaluation"]
