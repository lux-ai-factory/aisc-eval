from typing import Callable, Iterator, List, Protocol

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
    ) -> List[Metric]:
        """Run a specific model evaluation.

        Args:
            model: The model to run the evaluation.
            dataset: The dataset to evaluate.
            y_pred_proba: The predicted probabilities from the model on the dataset.

        """
        pass


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
) -> Callable[[ModelPredProbaEvaluator], list[Metric]]:
    """Decorator to register a function as a model evaluator for A4S.

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

    def func_decorator(func: ModelPredProbaEvaluator) -> None:
        model_pred_proba_evaluator_registry.register(name, func)
        return func

    return func_decorator
