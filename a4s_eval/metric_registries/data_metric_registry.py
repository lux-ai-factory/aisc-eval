from typing import Callable, Iterator, Protocol

from a4s_eval.data_model.evaluation import Dataset, DataShape
from a4s_eval.data_model.measure import Measure
from a4s_eval.service.api_client import get_dataset_data, get_evaluation, get_project_datashape
from a4s_eval.utils.dates import ProjectDataIterator
from a4s_eval.utils.logging import get_logger

logger = get_logger()


class DataMetric(Protocol):
    def __call__(
        self, datashape: DataShape, reference: Dataset, evaluated: Dataset
    ) -> list[Measure]:
        """Run a specific data evaluation.

        Args:
            datashape: The datashape of the project
            reference: The reference dataset to run the evaluation.
            evaluated: The evaluated dataset.

        """
        raise NotImplementedError


class DataMetricRegistry:
    def __init__(self) -> None:
        self._functions: dict[str, DataMetric] = {}
        logger.debug("DataMetricRegistry initialized")

    def register(self, name: str, func: DataMetric) -> None:
        logger.debug(f"Registering data evaluator: {name}")
        self._functions[name] = func

    def __iter__(self) -> Iterator[tuple[str, DataMetric]]:
        logger.debug(f"Iterating over {len(self._functions)} registered evaluators")
        return iter(self._functions.items())

    def get_functions(self) -> dict[str, DataMetric]:
        return self._functions
    
    @classmethod
    def get_metric_inputs(cls, eval_pid: str) -> tuple[DataShape, Dataset, Dataset]:
        evaluation = get_evaluation(eval_pid)
        evaluation.dataset.data = get_dataset_data(evaluation.dataset.pid)
        evaluation.model.dataset.data = get_dataset_data(evaluation.model.dataset.pid)
        datashape = get_project_datashape(evaluation.project.pid)
        return (datashape, evaluation.model.dataset, evaluation.dataset)

    @classmethod
    def get_metric_inputs_dateiterator(cls, eval_pid: str) -> Iterator[tuple[DataShape, Dataset, Dataset]]:
        evaluation = get_evaluation(eval_pid)
        project_pid = evaluation.project.pid
        date_iterator = ProjectDataIterator(project_pid)
        datashape, ref_dataset, eval_dataset = cls.get_metric_inputs(eval_pid)
        date_iterator.set_dataset(eval_dataset)
        return ((datashape, ref_dataset, eval_data) for _, eval_data in date_iterator)

data_metric_registry = DataMetricRegistry()


def data_metric(name: str) -> Callable[[DataMetric], DataMetric]:
    """Decorator to register a function as a data evaluator for A4S.
        name: The name to register the evaluator under.

    Returns:
        Callable[[DataMetric], DataMetric]: A decorator function that registers the evaluation function as a data evaluator for A4S.
    """
    logger.debug(f"Creating data evaluator decorator for: {name}")

    def func_decorator(func: DataMetric) -> DataMetric:
        data_metric_registry.register(name, func)
        return func

    return func_decorator
