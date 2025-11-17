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



@regression_metric(name="Simple demo metric")
def simple_demo_metric(
    datashape: DataShape, model: Model, dataset: Dataset, y_pred: np.ndarray
) -> list[Measure]:
    date = pd.to_datetime(dataset.data[datashape.date.name]).max().to_pydatetime()

    # Get df test from NBG data
    df_test = dataset.data

    print(df_test[['col_user', 'col_rating', 'col_item']].head())
    print(y_pred[:5])

    # Do your metric calculations here


    # Instantiate Measure object(s) to return
    measure = Measure(
        name="score_mean_value_metric",
        score=y_pred.mean(),
        time=date
    )
    print(measure)

    return [measure]
