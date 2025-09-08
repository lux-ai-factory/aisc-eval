from typing import Callable, Iterator, Protocol

import numpy as np

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.data_model.metric import Metric


class ModelPredProbaEvaluator(Protocol):
    def __call__(
        self,
        datashape: DataShape,
        model: Model,
        dataset: Dataset,
        y_pred_proba: np.ndarray,
    ) -> list[Metric]:
        """Run a specific model evaluation.

        Args:
            model: The model to run the evaluation.
            dataset: The dataset to evaluate.
            y_pred_proba: The predicted probabilities from the model on the dataset.

        """
        raise NotImplementedError


class ModelPredProbaEvaluatorRegistry:
    def __init__(self):
        self._functions = {}

    def register(self, name: str, func: ModelPredProbaEvaluator):
        self._functions[name] = func

    def __iter__(self) -> Iterator[tuple[str, ModelPredProbaEvaluator]]:
        return iter(self._functions.items())


model_pred_proba_evaluator_registry = ModelPredProbaEvaluatorRegistry()


def model_pred_proba_evaluator(
    name: str,
) -> Callable[[ModelPredProbaEvaluator], ModelPredProbaEvaluator]:
    """Decorator to register a function as a model evaluator for A4S.

    Returns:
        Callable[[Evaluator], ModelPredProbaEvaluator]: A decorator function that registers the evaluation function as a model evaluator for A4S.
    """

    def func_decorator(func: ModelPredProbaEvaluator) -> ModelPredProbaEvaluator:
        model_pred_proba_evaluator_registry.register(name, func)
        return func

    return func_decorator
