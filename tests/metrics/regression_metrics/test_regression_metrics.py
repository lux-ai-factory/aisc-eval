import datetime
import numpy as np
import pandas as pd
import pytest
import uuid
import onnxruntime as ort

from a4s_eval.data_model.evaluation import Dataset, DataShape, Feature, FeatureType, Model
from a4s_eval.metrics.regression_metrics.counter_factual_metrics import (
    empty_regression_metric,
)


@pytest.fixture(scope="module")
def data_shape() -> DataShape:
    df = pd.read_csv("tests/data/counterfactual/testing_data_2021-11-23 00:00:00.csv")
    type_mapping = {
        "int64": FeatureType.INTEGER,
        "float64": FeatureType.FLOAT,
        # object type mostly comes from textual data
        "object": FeatureType.CATEGORICAL,
        "datetime64[ns]": FeatureType.DATE,
    }
    features = []
    for col in ['past_profitability_21d', 'past_profitability_63d',
       'past_profitability_126d', 'volatility_21d', 'volatility_63d',
       'volatility_126d', 'avg_price_21d', 'avg_price_63d', 'avg_price_126d',
       'sharpe_21d', 'sharpe_63d', 'sharpe_126d', 'm_21d', 'm_63d', 'm_126d',
       'roc_21d', 'roc_63d', 'roc_126d', 'MACD', 'rsi_14', 'dco_22', 'min_21d',
       'min_63d', 'min_126d', 'max_21d', 'max_63d', 'max_126d', 'exp_mean_21d',
       'exp_mean_63d', 'exp_mean_126d']:
        col_type = str(df[col].dtype)
        col_type = type_mapping[col_type]
        _feature = Feature(
            pid=uuid.uuid4(),
            name=col,
            feature_type=col_type,
            min_value=df[col].min(),
            max_value=df[col].max(),
        )

        if col_type in [FeatureType.CATEGORICAL, FeatureType.DATE]:
            _feature.min_value = 0
            _feature.max_value = 0

        features.append(_feature)

    date = Feature(
            pid=uuid.uuid4(),
            name="col_timestamp",
            feature_type=FeatureType.DATE,
            min_value=0,
            max_value=0,
        )

    datashape = DataShape(features=features, date=date, target=None)

    return datashape


@pytest.fixture(scope="module")
def test_dataset(data_shape: DataShape) -> Dataset:
    data = pd.read_csv("./tests/data/counterfactual/testing_data_2021-11-23 00:00:00.csv")
    data["col_timestamp"] = pd.to_datetime(data["col_timestamp"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


@pytest.fixture(scope="module")
def ref_dataset(data_shape: DataShape) -> Dataset:
    data = pd.read_csv("./tests/data/counterfactual/training_data_2021-11-23 00:00:00.csv")
    # Training dataset does not have date column
    # data["col_timestamp"] = pd.to_datetime(data["col_timestamp"])
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=data,
    )


@pytest.fixture(scope="module")
def ref_model(ref_dataset: Dataset) -> Model:
    return Model(
        pid=uuid.uuid4(),
        model=None,
        dataset=ref_dataset,
    )


@pytest.fixture(scope="module")
def y_pred(test_dataset: Dataset) -> np.ndarray:
    session = ort.InferenceSession("./tests/data/counterfactual/profitability_recommendation_2021-11-23 00:00:00.onnx")
    df = test_dataset.data[[f.name for f in test_dataset.shape.features]]
    x_test = df.astype(np.float32).to_numpy()

    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    pred_onx = session.run([label_name], {input_name: x_test})[0]
    y_pred = np.array([d.item() for d in pred_onx])

    return y_pred


def test_smoke(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    y_pred: np.ndarray,
) -> None:
    metrics = empty_regression_metric(data_shape, ref_model, test_dataset, y_pred)
    assert len(metrics) == 0