import importlib
import pkgutil
from types import ModuleType

import a4s_eval.metrics
from a4s_eval.metric_registries.abstract import AbstractMetricRegistry
from a4s_eval.metric_registries.data_metric_registry import (
    data_metric_registry,
)
from a4s_eval.metric_registries.prediction_metric_registry import (
    prediction_metric_registry,
)


registries: list[AbstractMetricRegistry] = [
    data_metric_registry,
    prediction_metric_registry,
]

registry_mapping: dict[str, AbstractMetricRegistry] = {
    type(r).__name__: r for r in registries
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
