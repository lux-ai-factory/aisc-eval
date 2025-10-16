from abc import ABC, abstractmethod

from a4s_eval.data_model.evaluation import DataShape, Dataset, Evaluation
from a4s_eval.service.api_client import (
    get_dataset_data,
    get_evaluation,
    get_onnx_model,
    get_project_datashape,
)
from a4s_eval.utils.dates import ProjectDateIterator


class MetricInputGenerator(ABC):
    """
    Given an evaluation PID, prepare and yield the inputs required by metrics
    """

    def __init__(self, eval_pid: str):
        self.eval_pid = eval_pid
        self.__evaluation: Evaluation | None = None
        self.__train_dataset: Dataset | None = None
        self.__test_dataset: Dataset | None = None
        self.__expected_datashape: DataShape | None = None

    @property
    def evaluation(self) -> Evaluation:
        """Fetch the evaluation object based on eval_pid."""
        if self.__evaluation is None:
            self.__evaluation = get_evaluation(self.eval_pid)
        return self.__evaluation

    @property
    def expected_datashape(self) -> DataShape:
        """Return the expected data shape for the evaluation."""
        if self.__expected_datashape is None:
            self.__expected_datashape = get_project_datashape(
                self.evaluation.project.pid
            )
        return self.__expected_datashape

    @property
    def test_dataset(self) -> Dataset:
        """Return the dataset associated with the evaluation."""
        if self.__test_dataset is None:
            dataset = self.evaluation.dataset
            dataset.data = get_dataset_data(self.evaluation.dataset.pid)
            self.__test_dataset = dataset
        return self.__test_dataset

    @property
    def train_dataset(self) -> Dataset:
        """Return the dataset associated with the evaluation."""
        if self.__train_dataset is None:
            dataset = self.evaluation.model.dataset
            dataset.data = get_dataset_data(self.evaluation.model.dataset.pid)
            self.__train_dataset = dataset
        return self.__train_dataset

    @property
    def model_onnx_session(self):
        """Return the ONNX session for the model associated with the evaluation."""
        return get_onnx_model(self.evaluation.model.pid)

    @property
    def project_date_iterator(self):
        """Return the date iterator for the project associated with the evaluation."""
        return ProjectDateIterator(self.evaluation.project.pid)

    @abstractmethod
    def get_inputs(self):
        """Return the prepared inputs for metric evaluation."""
        pass

    @abstractmethod
    def get_inputs_dateiterator(self):
        """Return an iterator that yields inputs for each date slice."""
        pass


class AbstractMetricRegistry(ABC):
    InputGenerator: MetricInputGenerator

    @abstractmethod
    def register(self, name: str, func):
        """Register a metric function with a given name."""
        pass

    @abstractmethod
    def __iter__(self):
        """Return an iterator over registered metric functions."""
        pass

    @abstractmethod
    def get_functions(self):
        """Return a dictionary of registered metric functions."""
        pass

    def get_metric_inputs(self, eval_pid: str):
        """Return the inputs required by the metrics for a given evaluation PID."""
        input_generator = self.InputGenerator(eval_pid=eval_pid)
        return input_generator.get_inputs()

    def get_metric_inputs_dateiterator(self, eval_pid: str):
        """Return an iterator that yields inputs for each date slice for a given evaluation PID."""
        input_generator = self.InputGenerator(eval_pid=eval_pid)
        return input_generator.get_inputs_dateiterator()
