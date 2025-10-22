from typing import Callable, Iterator, Protocol

import numpy as np

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.abstract import (
    AbstractMetricRegistry,
    MetricInputGenerator,
)
from a4s_eval.utils.logging import get_logger

logger = get_logger()


class ModelPredProbaMetric(Protocol):
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


class PredictionInputGenerator(MetricInputGenerator):
    def get_inputs(self) -> tuple[DataShape, Model, Dataset, np.ndarray]:
        session = self.model_onnx_session

        x_test = self.test_dataset.data
        x_test_np = x_test[
            [f.name for f in self.expected_datashape.features]
        ].to_numpy()

        input_name = session.get_inputs()[0].name
        label_name = session.get_outputs()[1].name
        pred_onx = session.run([label_name], {input_name: x_test_np})[0]
        y_pred_proba = np.array([list(d.values()) for d in pred_onx])
        get_logger().info("Computation finished for Y prediction probability.")
        return (
            self.expected_datashape,
            self.train_dataset,
            self.test_dataset,
            y_pred_proba,
        )

    def get_inputs_dateiterator(
        self,
    ) -> Iterator[tuple[DataShape, Model, Dataset, np.ndarray]]:
        date_iterator = self.project_date_iterator
        datashape, model, dataset, y_pred_proba = self.get_inputs()
        date_iterator.set_dataset(dataset)
        return (
            (datashape, model, eval_data, y_pred_proba[list(eval_data.data.index)])
            for _, eval_data in date_iterator
        )


class PredictionMetricRegistry(AbstractMetricRegistry):
    def __init__(self) -> None:
        self._functions: dict[str, ModelPredProbaMetric] = {}

    def register(self, name: str, func: ModelPredProbaMetric) -> None:
        self._functions[name] = func

    def __iter__(self) -> Iterator[tuple[str, ModelPredProbaMetric]]:
        return iter(self._functions.items())

    def get_functions(self) -> dict[str, ModelPredProbaMetric]:
        return self._functions


prediction_metric_registry = PredictionMetricRegistry()


def prediction_metric(
    name: str,
) -> Callable[[ModelPredProbaMetric], ModelPredProbaMetric]:
    """Decorator to register a function as a model evaluator for A4S.

    Returns:
        Callable[[Evaluator], ModelPredProbaMetric]: A decorator function that registers the evaluation function as a model evaluator for A4S.
    """

    def func_decorator(func: ModelPredProbaMetric) -> ModelPredProbaMetric:
        prediction_metric_registry.register(name, func)
        return func

    return func_decorator
