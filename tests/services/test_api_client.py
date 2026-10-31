import uuid
from typing import Any
from unittest.mock import Mock, patch

import pytest

from aisc_eval.service.api_client import get_evaluation, mark_failed

TEST_UUIDS = {
    "project": uuid.UUID("afb49e3f-813d-8888-9919-ee179d1090e6"),
    "evaluation": uuid.UUID("afb49e3f-813d-4260-9919-ee179d1090e6"),
}


@pytest.fixture
def mock_evaluation_data() -> dict[str, Any]:
    return {
        "pid": str(TEST_UUIDS["evaluation"]),
        "project": {"pid": str(TEST_UUIDS["project"]), "name": "LCLD"},
        "evaluation_plugins": [],
    }


def test_get_evaluation(mock_evaluation_data: dict[str, Any]) -> None:
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_evaluation_data
    with patch("requests.get", return_value=mock_response):
        evaluation = get_evaluation(TEST_UUIDS["evaluation"])

    assert evaluation.pid == TEST_UUIDS["evaluation"]
    assert evaluation.project.pid == TEST_UUIDS["project"]
    assert evaluation.project.name == "LCLD"
    assert evaluation.evaluation_plugins == []


def test_mark_failed_sends_one_status_update_in_the_query_string() -> None:
    mock_response = Mock()
    mock_response.ok = True
    with patch("requests.put", return_value=mock_response) as put:
        mark_failed(TEST_UUIDS["evaluation"])

    put.assert_called_once()
    assert put.call_args.args[0].endswith(f"/evaluations/{TEST_UUIDS['evaluation']}?status=Failed")
    assert "json" not in put.call_args.kwargs
