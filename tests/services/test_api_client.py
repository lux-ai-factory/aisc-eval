import uuid
from typing import Any
from unittest.mock import Mock, patch

import pytest

from aisc_eval.service.api_client import get_evaluation, post_measures
from aisc_plugin_interface import Measure

TEST_UUIDS = {
    "project": uuid.UUID("afb49e3f-813d-8888-9919-ee179d1090e6"),
    "evaluation": uuid.UUID("afb49e3f-813d-4260-9919-ee179d1090e6"),
    "plugin": uuid.UUID("08947de5-bcc2-4fee-8717-deb3f8bc86f9"),
}


@pytest.fixture
def mock_evaluation_data() -> dict[str, Any]:
    return {
        "pid": str(TEST_UUIDS["evaluation"]),
        "project": {"pid": str(TEST_UUIDS["project"]), "name": "LCLD"},
        "evaluation_plugins": [
            {
                "pid": str(TEST_UUIDS["plugin"]),
                "name": "DemoPlugin",
                "package_name": "demo-pkg",
                "version": "0.1.0",
                "evaluation_inputs": [],
            }
        ],
    }


def test_get_evaluation_builds_model_from_api_response(
    mock_evaluation_data: dict[str, Any],
) -> None:
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_evaluation_data

    with patch("requests.get", return_value=mock_response):
        evaluation = get_evaluation(TEST_UUIDS["evaluation"])

    assert evaluation.pid == TEST_UUIDS["evaluation"]
    assert evaluation.project.name == "LCLD"
    assert evaluation.evaluation_plugins[0].name == "DemoPlugin"


def test_post_measures_keys_payload_by_evaluation_plugin() -> None:
    mock_response = Mock()
    mock_response.status_code = 201
    mock_response.headers = {}

    with patch("requests.post", return_value=mock_response) as post:
        post_measures(
            TEST_UUIDS["evaluation"],
            TEST_UUIDS["plugin"],
            [Measure(name="accuracy", score=0.9)],
        )

    _, kwargs = post.call_args
    assert list(kwargs["json"].keys()) == [str(TEST_UUIDS["plugin"])]
    assert kwargs["json"][str(TEST_UUIDS["plugin"])][0]["name"] == "accuracy"


def test_post_measures_raises_on_unexpected_status() -> None:
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.headers = {}
    mock_response.text = "boom"

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(ValueError):
            post_measures(TEST_UUIDS["evaluation"], TEST_UUIDS["plugin"], [])
