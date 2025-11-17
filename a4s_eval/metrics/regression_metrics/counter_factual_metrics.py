import numpy as np
import pandas as pd

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.regression_metric_registry import regression_metric




@regression_metric(name="Empty regression metric")
def empty_regression_metric(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred: np.ndarray
) -> list[Measure]:
    return []
