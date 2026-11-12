"""I1 — a run is self-contained: it is handed everything it needs at dispatch.

``run_evaluation`` (the dispatch path) resolves the dataset/model bytes once and
embeds them in the payload; ``run_plugin`` (the run path) only consumes what it
was given, so a backend or platform outage mid-run cannot fail the run.
"""

import base64

import pytest

from aisc_eval import celery_tasks


def test_dispatch_embeds_dataset_bytes(monkeypatch):
    monkeypatch.setattr(
        celery_tasks, "get_dataset_file_content", lambda name: b"dataset-bytes"
    )
    components = [{"component_type": "dataset", "name": "train", "data": "train.csv"}]

    celery_tasks._embed_input_files(components)

    assert components[0]["content_b64"] == base64.b64encode(b"dataset-bytes").decode(
        "ascii"
    )


def test_dispatch_embeds_model_bytes(monkeypatch):
    monkeypatch.setattr(
        celery_tasks, "get_model_file_content", lambda name: b"model-bytes"
    )
    components = [{"component_type": "model", "name": "model", "data": "model.onnx"}]

    celery_tasks._embed_input_files(components)

    assert components[0]["content_b64"] == base64.b64encode(b"model-bytes").decode(
        "ascii"
    )


def test_dispatch_leaves_non_file_components_alone(monkeypatch):
    components = [{"component_type": "datashape", "name": "shape", "json_value": {}}]

    celery_tasks._embed_input_files(components)

    assert "content_b64" not in components[0]


def test_run_path_decodes_dispatched_bytes():
    component = {"content_b64": base64.b64encode(b"payload").decode("ascii")}
    assert celery_tasks._decode_input_file(component) == b"payload"


def test_run_path_refuses_an_input_without_its_bytes():
    with pytest.raises(ValueError, match="self-contained"):
        celery_tasks._decode_input_file({"name": "train", "data": "train.csv"})
