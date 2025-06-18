from abc import ABC
from typing import Callable, Iterator, List, Protocol

from a4s_eval.data_model.evaluation import Dataset


class Metrics(ABC):
    pass


class DataEvaluator(Protocol):
    def __call__(self, reference: Dataset, evaluated: Dataset) -> List[Metrics]:
        """Run a specific data evaluation.

        Args:
            reference: The reference dataset to run the evaluation.
            evaluated: The evaluaded dataset.

        """
        pass


class DataEvaluatorRegistry:
    def __init__(self):
        self._functions = {}

    def register(self, name: str, func: DataEvaluator):
        self._functions[name] = func

    def __iter__(self) -> Iterator[tuple[str, DataEvaluator]]:
        return iter(self._functions.items())


data_evaluator_registry = DataEvaluatorRegistry()


def data_evaluator() -> Callable[[DataEvaluator], list[Metrics]]:
    """Decorator to register a function as a data evaluator for A4S.


    Returns:
        Callable[[Evaluator], list[Metrics]]: A decorator function that register the evaluation function as data evaluator for A4S
    """

    def func_decorator(func: DataEvaluator) -> None:
        data_evaluator_registry.register("Test", func)
        return func

    return func_decorator
