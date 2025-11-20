import importlib
import pkgutil
from types import ModuleType
from typing import Type

import a4s_eval.metrics
from a4s_eval.metric_registries.abstract import (
    AbstractMetricRegistry,
    MetricInputGenerator,
)
from a4s_eval.metric_registries.data_metric_registry import (
    DataMetricInputGenerator,
    data_metric_registry,
)
from a4s_eval.metric_registries.prediction_metric_registry import (
    PredictionInputGenerator,
    prediction_metric_registry,
)
from a4s_eval.metric_registries.regression_metric_registry import (
    RegressionInputGenerator,
    regression_metric_registry,
)



registries: list[AbstractMetricRegistry] = [
    data_metric_registry,
    prediction_metric_registry,
    regression_metric_registry,
]

# Mapping of registry class names to registry instances
registry_mapping: dict[str, AbstractMetricRegistry] = {
    type(r).__name__: r for r in registries
}

# Mapping of registry class names to related input generator classes
# Manual listing currently; can be automated if needed
input_generator_cls_mapping: dict[str, Type[MetricInputGenerator]] = {
    type(data_metric_registry).__name__: DataMetricInputGenerator,
    type(prediction_metric_registry).__name__: PredictionInputGenerator,
    type(regression_metric_registry).__name__: RegressionInputGenerator,
}


def auto_discover(package: ModuleType) -> None:
    """
    Recursively imports all submodules of a given package.
    This ensures decorators / registries inside those modules get executed.
    """
    for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
        full_name = f"{package.__name__}.{module_name}"
        module = importlib.import_module(full_name)

        if is_pkg:
            auto_discover(module)  # recurse into subpackage


auto_discover(a4s_eval.metrics)


def get_n_evaluation() -> int:
    return sum([len(r.get_functions()) for r in registries])


def get_available_metrics() -> list[str]:
    return [metric_name for r in registries for metric_name in r.get_functions()]


def map_registries_to_supported_metrics(
    metric_names: list[str],
) -> list[tuple[str, list[str]]]:
    """
    Given a list of metric names and available registries,
    return a mapping of registry class names to the subset of metrics they support.

    Args:
        metric_names: A list of metric names to check.
        registries: A list of registry instances that expose `get_functions()`.

    Returns:
        A list of (registry_name, supported_metrics) tuples.
        Each registry_name is the class name of the registry,
        and supported_metrics is a list of metric names that appear
        both in `metric_names` and the registry's registered functions.
    """
    requested_metrics = set(metric_names)
    registry_metric_pairs = []

    for registry in registries:
        registry_name = type(registry).__name__
        registered_metrics = set(registry.get_functions())
        # Find the intersection of requested and registered metrics
        supported_metrics = sorted(requested_metrics & registered_metrics)

        if supported_metrics:
            registry_metric_pairs.append((registry_name, supported_metrics))

    return registry_metric_pairs
