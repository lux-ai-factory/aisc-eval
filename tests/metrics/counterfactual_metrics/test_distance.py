import dice_ml
import numpy as np
import pandas as pd
import pytest
import uuid
import onnxruntime as ort

from sklearn.preprocessing import StandardScaler

from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Feature,
    FeatureType,
)
from a4s_eval.metrics.counterfactuals_metrics.perf_metric import (
    empty_counterfactual_metric,
    counterfactual_distance_metric,
)


@pytest.fixture(scope="module")
def X_train() -> pd.DataFrame:
    df = pd.read_csv(
        "tests/data/counterfactual/training_data_2021-11-23 00:00:00.csv"
    )

    return df


@pytest.fixture(scope="module")
def X_test() -> pd.DataFrame:
    df = pd.read_csv(
        "tests/data/counterfactual/testing_data_2021-11-23 00:00:00.csv"
    )

    return df.iloc[[2,3,5,7,11,13,17,19,23,29]]  # select a few samples for faster testing


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
def session() -> ort.InferenceSession:
    session = ort.InferenceSession(
        "tests/data/counterfactual/profitability_recommendation_2021-11-23 00:00:00.onnx"
    )
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
def scaler(X_test: pd.DataFrame, data_shape: DataShape) -> StandardScaler:
    ss = StandardScaler()
    ss.fit(X_test[[f.name for f in data_shape.features]])
    return ss


@pytest.fixture(scope="module")
def dice_data(X_train: pd.DataFrame, data_shape: DataShape) -> dice_ml.Data:
    numeric_columns = [feature.name for feature in data_shape.features if feature.feature_type == FeatureType.FLOAT]

    dice_data = dice_ml.Data(
        dataframe=X_train,
        continuous_features=list(numeric_columns),
        outcome_name=data_shape.target.name
    )
    return dice_data


@pytest.fixture(scope="module")
def dice_model(session: ort.InferenceSession) -> dice_ml.Model:
    class ONNXRegressorWrapper:
        def __init__(self, session, input_name=None, output_name=None):
            # Load ONNX model
            self.session = session
            # Resolve input/output names automatically if not provided
            self.input_name = input_name or self.session.get_inputs()[0].name
            self.output_name = output_name or self.session.get_outputs()[0].name
        def predict(self, X):
            # Ensure numpy float32 input
            X = np.array(X).astype(np.float32)
            inputs = {self.input_name: X}
            output = self.session.run([self.output_name], inputs)[0]
            # Flatten if necessary (e.g., output shape (N,1))
            return output.reshape(len(output))
        
    model_wrapper = ONNXRegressorWrapper(session)
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
def factual_scaled(X_test, scaler: StandardScaler, data_shape: DataShape) -> pd.DataFrame:
    df = pd.DataFrame(
        scaler.transform(X_test[[f.name for f in data_shape.features]]),
        columns=[f.name for f in data_shape.features],
        index=X_test.index
    )
    return df


@pytest.fixture(scope="module")
def counter_factual(
    factual_scaled: pd.DataFrame,
    scaler: StandardScaler,
    data_shape: DataShape,
    exp: dice_ml.Dice,
) -> pd.DataFrame:
    counterfactual_scaled = []
    for i in range(len(factual_scaled)):
        query_instance = factual_scaled.iloc[i:i+1]
        dice_exp = exp.generate_counterfactuals(
            query_instance,
            total_CFs=1,
            desired_range=[data_shape.target.min_value, data_shape.target.max_value]
        )
        cf_df = dice_exp.cf_examples_list[0].final_cfs_df.copy()
        scaled_cf_df = pd.DataFrame( 
            scaler.transform(cf_df[[f.name for f in data_shape.features]]),
            columns=[f.name for f in data_shape.features],
            index=query_instance.index
        )
        counterfactual_scaled.append(scaled_cf_df)
    return pd.concat(counterfactual_scaled)


def test_smoke(
    factual_scaled: pd.DataFrame,
    data_shape: DataShape,
    counter_factual: pd.DataFrame,
) -> None:
    metrics = empty_counterfactual_metric(data_shape, factual_scaled, counter_factual)
    assert len(metrics) == 0



def test_counterfactual_distance_metric(
    data_shape: DataShape,
    factual_scaled: pd.DataFrame,
    counter_factual: pd.DataFrame,
) -> None:
    metrics = counterfactual_distance_metric(
        data_shape,
        factual_scaled,
        counter_factual
    )

    for metric in metrics:
        assert metric.name == "euclidean"
        assert metric.score >= 0
        print(f"Counterfactual distance metric score: {metric.score}")