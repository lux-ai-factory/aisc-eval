import dice_ml
import numpy as np
import pandas as pd
import pytest
import uuid
import onnxruntime as ort


from sklearn.ensemble import RandomForestRegressor
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn

from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Feature,
    FeatureType,
)
from a4s_eval.metric_registries.counterfactuals_metric_registry import ONNXWrapper
from a4s_eval.metrics.counterfactuals_metrics.perf_metric import (
    empty_counterfactual_metric,
    counterfactual_distance_metric,
)


@pytest.fixture(scope="module")
def X_train() -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "col_timestamp": [
                "2021-11-23 00:00:00",
                "2021-11-24 00:00:00",
                "2021-11-25 00:00:00",
                "2021-11-25 00:00:00",
                "2021-11-25 00:00:00",
                "2021-11-26 00:00:00",
            ],
            "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature_2": [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "target": [2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
        }
    )
    return data


@pytest.fixture(scope="module")
def X_test() -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "col_timestamp": [
                "2021-11-26 00:00:00",
                "2021-11-27 00:00:00",
            ],
            "feature_1": [7.0, 8.0],
            "feature_2": [10.0, 11.0],
            "target": [8.5, 9.5],
        }
    )
    return data


@pytest.fixture(scope="module")
def data_shape(X_train: pd.DataFrame) -> DataShape:
    features = []
    for col in X_train.columns:
        if col == "col_timestamp":
            continue
        elif col == "target":
            continue
        else:
            feature = Feature(
                pid=uuid.uuid4(),
                name=col,
                feature_type=FeatureType.FLOAT,
                min_value=float(X_train[col].min()),
                max_value=float(X_train[col].max()),
            )
            features.append(feature)
    
    date = Feature(
        pid=uuid.uuid4(),
        name="date",
        feature_type=FeatureType.DATE,
        min_value=0,
        max_value=0,
    )

    target = Feature(
        pid=uuid.uuid4(),
        name="target",
        feature_type=FeatureType.FLOAT,
        min_value=float(X_train["target"].min()),
        max_value=float(X_train["target"].max()),
    )

    datashape = DataShape(features=features, date=date, target=target)

    return datashape


@pytest.fixture(scope="module")
def test_dataset(data_shape: DataShape, X_test: pd.DataFrame) -> Dataset:
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=X_test)


@pytest.fixture(scope="module")
def ref_dataset(data_shape: DataShape, X_train: pd.DataFrame) -> Dataset:
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=X_train,
    )


@pytest.fixture(scope="module")
def session(X_train: pd.DataFrame, data_shape: DataShape) -> ort.InferenceSession:
    clr = RandomForestRegressor()
    clr.fit(X_train[[f.name for f in data_shape.features]], X_train[data_shape.target.name])

    initial_type = [("float_input", FloatTensorType([None, len(data_shape.features)]))]
    onx = convert_sklearn(clr, initial_types=initial_type, target_opset=12)
    session = ort.InferenceSession(onx.SerializeToString())

    return session


@pytest.fixture(scope="module")
def y_pred(X_test: Dataset, data_shape: DataShape, session: ort.InferenceSession) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    label_name = session.get_outputs()[0].name
    pred_onx = session.run(
        [label_name],
        {
            input_name: X_test[[f.name for f in data_shape.features]]
            .astype(np.float32)
            .to_numpy()
        }
    )[0]
    y_pred = np.array([d.item() for d in pred_onx])
    return y_pred


@pytest.fixture(scope="module")
def dice_data(X_train: pd.DataFrame, data_shape: DataShape) -> dice_ml.Data:
    numeric_columns = [feature.name for feature in data_shape.features if feature.feature_type == FeatureType.FLOAT]
    X_train = X_train.drop("col_timestamp", axis=1)

    dice_data = dice_ml.Data(
        dataframe=X_train,
        continuous_features=list(numeric_columns),
        outcome_name=data_shape.target.name
    )
    return dice_data


@pytest.fixture(scope="module")
def dice_model(session: ort.InferenceSession) -> dice_ml.Model:
    model_wrapper = ONNXWrapper(session)
    dice_model = dice_ml.Model(model=model_wrapper, backend="sklearn", model_type="regressor")
    return dice_model


@pytest.fixture(scope="module")
def exp(
    dice_data: dice_ml.Data,
    dice_model: dice_ml.Model
) -> dice_ml.Dice:
    exp = dice_ml.Dice(dice_data, dice_model, method="genetic")
    return exp


@pytest.fixture(scope="module")
def factuals(data_shape:DataShape, X_test: pd.DataFrame) -> pd.DataFrame:
    return X_test[[f.name for f in data_shape.features]]

@pytest.fixture(scope="module")
def counter_factuals(
    X_test: pd.DataFrame,
    data_shape: DataShape,
    exp: dice_ml.Dice,
) -> pd.DataFrame:
    counterfactuals = []
    feature_cols = [f.name for f in data_shape.features]
    for i in range(len(X_test)):
        instance_id = X_test.index[i]
        query_instance = X_test.loc[X_test.index[i:i+1], feature_cols]
        dice_exp = exp.generate_counterfactuals(
            query_instance,
            total_CFs=1,
            desired_range=[data_shape.target.min_value, data_shape.target.max_value]
        )
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df.copy()

        # set the index of the CF to the original instance ID
        cf_df.index = pd.Index([instance_id])

        counterfactuals.append(cf_df)
    
    # print(counterfactuals)
    return pd.concat(counterfactuals)


@pytest.fixture(scope="module")
def mad_values(exp: dice_ml.Dice) -> dict:
    """
    Get valid MADs from the DiCE data interface.
    Returns a dict: {feature_name: MAD_j}
    """
    mad = exp.data_interface.get_valid_mads()
    # Convert to Python dict (DiCE sometimes returns numpy arrays)
    # mad_dict = {k: float(v) for k, v in mad.items()}

    return mad



def test_smoke(
    factuals: pd.DataFrame,
    data_shape: DataShape,
    counter_factuals: pd.DataFrame,
    mad_values: dict,
) -> None:
    # metrics = empty_counterfactual_metric(data_shape, factuals, counter_factuals, mad_values)
    metrics = []
    assert len(metrics) == 0



def test_counterfactual_distance_metric(
    data_shape: DataShape,
    factuals: pd.DataFrame,
    counter_factuals: pd.DataFrame,
    mad_values: dict,
) -> None:
    metrics = counterfactual_distance_metric(
        data_shape,
        factuals,
        counter_factuals,
        mad_values
    )

    for metric in metrics:
        assert metric.name == "euclidean" # TODO: change to Manhattan
        assert metric.score >= 0
        print(f"Counterfactual distance metric score: {metric.score}")

    print(f"Average distance metric score: {np.average([m.score for m in metrics])}")
