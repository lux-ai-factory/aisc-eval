import pytest

from a4s_eval.data_model.evaluation import Dataset, DataShape
from a4s_eval.metric_registries.data_metric_registry import (
    DataMetric,
    data_metric_registry,
)


def test_non_empty_registry():
    assert len(data_metric_registry._functions) > 0


@pytest.mark.parametrize("evaluator_function", [e[1] for e in data_metric_registry])
def test_data_metric_registry_contains_evaluator(
    evaluator_function: DataMetric,
    data_shape: DataShape,
    ref_dataset: Dataset,
    test_dataset: Dataset,
):
    metrics = evaluator_function(data_shape, ref_dataset, test_dataset)

    assert len(metrics) > 0
