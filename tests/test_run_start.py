"""The start of a run: the worker reads the run from the engine, including a helper model input (input_type
"helper", a helper_ref, no file), and a run that cannot start is marked Failed with the reason on each of its
tests, never left Pending."""
import uuid

import pytest

from aisc_eval import celery_tasks
from aisc_eval.data_model.evaluation import Evaluation

EVALUATION = uuid.UUID("11111111-1111-1111-1111-111111111111")
PROJECT = "22222222-2222-2222-2222-222222222222"
EVALUATION_PLUGIN = "33333333-3333-3333-3333-333333333333"
PLUGIN = "44444444-4444-4444-4444-444444444444"


def engine_run(inputs):
    """GET /evaluations/{pid}?include=plugin as the engine answers it (EvaluationInputOutSchema per input)."""
    return {"pid": str(EVALUATION), "project": {"pid": PROJECT, "name": "mijke"},
            "evaluation_plugins": [{"pid": EVALUATION_PLUGIN, "name": "StrongRejectPlugin",
                                    "package_name": "aisc-plugin-strongreject-mijke", "version": "0.1.3",
                                    "plugin_config": None, "evaluation_inputs": inputs}]}


DATASET = {"name": "forbidden-prompts-file", "input_type": "dataset", "helper_ref": None, "value": {},
           "input_file": {"pid": "55555555-5555-5555-5555-555555555555", "name": "probes.csv", "data": "probes.csv"}}
JUDGE = {"name": "helper:judge", "input_type": "helper", "input_file": None,
         "helper_ref": f"helper:{PROJECT}/llm-as-judge", "value": {"model": "gpt-4o-mini"}}


def test_a_run_with_a_helper_model_input_is_read():
    evaluation = Evaluation.model_validate(engine_run([DATASET, JUDGE]))
    judge = evaluation.evaluation_plugins[0].evaluation_inputs[1]
    assert judge.input_type == "helper" and judge.input_file is None
    assert judge.helper_ref == f"helper:{PROJECT}/llm-as-judge"


def test_a_component_input_still_needs_its_file():
    with pytest.raises(ValueError):
        Evaluation.model_validate(engine_run([{**DATASET, "input_file": None}]))


@pytest.fixture
def platform_calls(monkeypatch):
    calls = {"failed": [], "plugin_failed": []}
    monkeypatch.setattr(celery_tasks.plugin_loader, "list_packages", lambda refresh=False: {})
    monkeypatch.setattr(celery_tasks, "mark_failed", lambda pid: calls["failed"].append(pid))
    monkeypatch.setattr(celery_tasks, "mark_plugin_failed",
                        lambda pid, ep, message="": calls["plugin_failed"].append((pid, ep, message)))
    return calls


def test_a_run_that_cannot_start_is_marked_failed_with_its_reason(platform_calls, monkeypatch):
    broken = engine_run([{**DATASET, "input_type": "nonsense"}])
    monkeypatch.setattr(celery_tasks, "get_evaluation", lambda pid: Evaluation.model_validate(broken))
    monkeypatch.setattr(celery_tasks, "get_evaluation_request", lambda pid: broken)
    monkeypatch.setattr(celery_tasks, "get_evaluation_inputs", lambda pid: {})
    celery_tasks.run_evaluation.run(EVALUATION)
    assert platform_calls["failed"] == [EVALUATION]
    [(pid, ep, message)] = platform_calls["plugin_failed"]
    assert (pid, str(ep)) == (EVALUATION, EVALUATION_PLUGIN)
    assert message.startswith("the run could not start: ValidationError")


def test_a_run_whose_details_cannot_be_read_at_all_is_still_marked_failed(platform_calls, monkeypatch):
    def unreachable(pid):
        raise ConnectionError("engine not answering")

    monkeypatch.setattr(celery_tasks, "get_evaluation", unreachable)
    monkeypatch.setattr(celery_tasks, "get_evaluation_request", unreachable)
    celery_tasks.run_evaluation.run(EVALUATION)
    assert platform_calls["failed"] == [EVALUATION]
    assert platform_calls["plugin_failed"] == []
