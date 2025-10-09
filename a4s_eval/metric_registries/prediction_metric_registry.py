from typing import Callable, Iterator, Protocol

import numpy as np

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.service.api_client import get_dataset_data, get_evaluation, get_onnx_model, get_project_datashape
from a4s_eval.utils.dates import ProjectDataIterator
from a4s_eval.utils.logging import get_logger

logger = get_logger()

class ModelPredProbaEvaluator(Protocol):
    def __call__(
        self,
        datashape: DataShape,
        model: Model,
        dataset: Dataset,
        y_pred_proba: np.ndarray,
    ) -> list[Measure]:
        """Run a specific model evaluation.

        Args:
            model: The model to run the evaluation.
            dataset: The dataset to evaluate.
            y_pred_proba: The predicted probabilities from the model on the dataset.

        """
        raise NotImplementedError


class PredictionMetricRegistry:
    def __init__(self) -> None:
        self._functions: dict[str, ModelPredProbaEvaluator] = {}

    def register(self, name: str, func: ModelPredProbaEvaluator) -> None:
        self._functions[name] = func

    def __iter__(self) -> Iterator[tuple[str, ModelPredProbaEvaluator]]:
        return iter(self._functions.items())

    def get_functions(self) -> dict[str, ModelPredProbaEvaluator]:
        return self._functions

    @classmethod
    def get_metric_inputs(cls, eval_pid: str) -> tuple[DataShape, Model, Dataset, np.ndarray]:
        evaluation = get_evaluation(eval_pid)
        evaluation.dataset.data = get_dataset_data(evaluation.dataset.pid)
        session = get_onnx_model(evaluation.model.pid)
        datashape = get_project_datashape(evaluation.project.pid)

        x_test = evaluation.dataset.data
        x_test_np = x_test[[f.name for f in datashape.features]].to_numpy()

        input_name = session.get_inputs()[0].name
        label_name = session.get_outputs()[1].name
        pred_onx = session.run([label_name], {input_name: x_test_np})[0]
        y_pred_proba = np.array([list(d.values()) for d in pred_onx])
        get_logger().info("Computation finished for Y prediction probability.")
        return (datashape, evaluation.model.dataset, evaluation.dataset, y_pred_proba)

    @classmethod
    def get_metric_inputs_dateiterator(cls, eval_pid: str) -> Iterator[tuple[DataShape, Model, Dataset, np.ndarray]]:
        evaluation = get_evaluation(eval_pid)
        project_pid = evaluation.project.pid
        date_iterator = ProjectDataIterator(project_pid)
        datashape, model, dataset, y_pred_proba = cls.get_metric_inputs(eval_pid)
        date_iterator.set_dataset(dataset)
        return ((datashape, model, dataset, y_pred_proba[list(eval_data.data.index)]) for _, eval_data in date_iterator)


prediction_metric_registry = PredictionMetricRegistry()


def prediction_metric(
    name: str,
) -> Callable[[ModelPredProbaEvaluator], ModelPredProbaEvaluator]:
    """Decorator to register a function as a model evaluator for A4S.

    Returns:
        Callable[[Evaluator], ModelPredProbaEvaluator]: A decorator function that registers the evaluation function as a model evaluator for A4S.
    """

    def func_decorator(func: ModelPredProbaEvaluator) -> ModelPredProbaEvaluator:
        prediction_metric_registry.register(name, func)
        return func

    return func_decorator
