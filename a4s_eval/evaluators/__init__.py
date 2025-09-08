import importlib
import pkgutil
from types import ModuleType

import a4s_eval.evaluations
from a4s_eval.evaluators.data_evaluator import data_evaluator_registry
from a4s_eval.evaluators.model_evaluator import model_pred_proba_evaluator_registry

registries = [data_evaluator_registry, model_pred_proba_evaluator_registry]


def auto_discover(package: ModuleType):
    """
    Recursively imports all submodules of a given package.
    This ensures decorators / registries inside those modules get executed.
    """
    for _, module_name, is_pkg in pkgutil.iter_modules(package.__path__):
        full_name = f"{package.__name__}.{module_name}"
        module = importlib.import_module(full_name)

        if is_pkg:
            auto_discover(module)  # recurse into subpackage


auto_discover(a4s_eval.evaluations)


def get_n_evaluation() -> int:
    return sum([len(r._functions) for r in registries])
