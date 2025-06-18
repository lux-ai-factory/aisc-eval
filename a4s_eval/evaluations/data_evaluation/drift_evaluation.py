
from a4s_eval.data_model.evaluation import Dataset
from a4s_eval.evaluations.data_evaluation.registry import Metrics, data_evaluator


@data_evaluator()
def empty_data_evaluator(reference: Dataset, evaluated: Dataset) -> list[Metrics]:
    return []
