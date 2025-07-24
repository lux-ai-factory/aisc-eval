from typing import Callable, Iterator, List, Protocol

from a4s_eval.data_model.evaluation import Dataset
from a4s_eval.data_model.metric import Metric


class DataEvaluator(Protocol):
    def __call__(self, reference: Dataset, ewvaluated: Dataset) -> List[Metric]:
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


def data_evaluator(name: str) -> Callable[[DataEvaluator], list[Metric]]:
    """Decorator to register a function as a data evaluator for A4S.

    Args:INSERT INTO model (id, pid, name, data, project_id, dataset_id)
    VALUES (
        id:integer,
        'pid:uuid',
        'name:character varying',
        'data:character varying',
        project_id:integer,
        dataset_id:integer
      );
        name: The name to register the evaluator under.

    Returns:
        Callable[[Evaluator], list[Metrics]]: A decorator function that registers the evaluation function as a data evaluator for A4S.
    """

    def func_decorator(func: DataEvaluator) -> None:
        data_evaluator_registry.register(name, func)
        return func

    return func_decorator
