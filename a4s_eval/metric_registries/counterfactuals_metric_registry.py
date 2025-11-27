from typing import Callable, Iterator, Protocol
from typing import List

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import dice_ml
from dice_ml import Dice

from a4s_eval.data_model.evaluation import Dataset, DataShape, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.abstract import (
    AbstractMetricRegistry,
    MetricInputGenerator,
)
from a4s_eval.utils.logging import get_logger

logger = get_logger()


class ONNXWrapper:
    def __init__(self, session: Model) -> None:
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.label_name = session.get_outputs()[0].name

    def predict(self, X: Dataset) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.astype(np.float32).values
        else: 
            raise TypeError(f"Expected X to be a pandas DataFrame, but got {type(X).__name__}")

        # Run ONNX inference
        pred = self.session.run([self.label_name], {self.input_name: X})[0]

        # flatten ONNX output so DiCE returns scalars, not lists
        pred = pred.reshape(-1)  

        return pred
    

class CounterfactualMetric(Protocol):
    def __call__(
        self,
        expected_datashape: DataShape,
        factual_scaled: pd.DataFrame,
        counterfactuals: pd.DataFrame,
    ) -> list[Measure]:
        """Run a specific model evaluation.

        Args:
            expected_datashape: The expected datashape for the project
            test_dataset: The test dataset to run the evaluation.
            counterfactuals: The counterfactuals that is generated.

        """
        raise NotImplementedError


class CounterfactualsInputGenerator(MetricInputGenerator):
    def get_inputs(self) -> tuple[DataShape, pd.DataFrame, pd.DataFrame]:
        session = self.model_onnx_session

        x_test = self.test_dataset.data
        if x_test is None:
            raise ValueError("Test dataset data is None.")

        x_train = self.train_dataset.data
        scaler = StandardScaler()
        scaler.fit(x_train.drop(columns=self.expected_datashape.target.name))

        numeric_columns = x_train.select_dtypes(include=["number"]).columns

        # remove target column
        numeric_columns = [col_name for col_name in numeric_columns.to_list() if col_name!= self.expected_datashape.target.name]

        dice_data = dice_ml.Data(
            dataframe=x_train,
            continuous_features=list(numeric_columns),
            outcome_name=self.expected_datashape.target.name
        )
        # Sklearn wrapping
        model_wrapper = ONNXWrapper(session)
        # Comment: It seems that onnx is not supported
        dice_model = dice_ml.Model(model=model_wrapper, backend="sklearn", model_type="regressor")

        # Initiate DiCE
        exp = Dice(dice_data, dice_model, method="genetic")  # "random" ou "genetic" ou "kd"

        counterfactual = []
        n_samples = len(x_test.iloc[:10,:]) # Limit to first 10 samples for efficiency
        
        target_col = self.expected_datashape.target.name
        date_col = self.expected_datashape.date.name
        # Min/max of target for desired_range
        target_min = self.expected_datashape.target.min_value
        target_max = self.expected_datashape.target.max_value
        query_instances_scaled = []

        for i in range(min(n_samples, len(x_test))):
            query_instance = x_test.drop(columns=[target_col, date_col]).iloc[i:i+1].astype(np.float32)

            # Standardize factual
            factual_scaled = pd.DataFrame(
                scaler.transform(query_instance),
                columns=query_instance.columns,
                index=query_instance.index
            )
            query_instances_scaled.append(factual_scaled)

            # Comment: It failed as model is onnx format
            dice_exp = exp.generate_counterfactuals(
                query_instance,
                total_CFs=1,
                desired_range=[target_min, target_max]
            )
            
            cf = dice_exp.cf_examples_list[0].final_cfs_df.copy()
            cf = cf[query_instance.index]  # Ensure same feature order

            # Standardize counterfactual
            counterfactual_scaled = pd.DataFrame(
                scaler.transform(cf),
                columns=cf.columns[:-1],
                index=cf.index
            )

            counterfactual.append(counterfactual_scaled)

        query_instances_scaled = pd.concat(query_instances_scaled)
        counterfactuals_scaled = pd.concat(counterfactual)

        get_logger().info("Computation finished for counterfactuals.")
        return (
            self.expected_datashape,
            query_instances_scaled,
            counterfactuals_scaled
        )

    def get_inputs_dateiterator(
        self,
    ) -> Iterator[tuple[DataShape, Dataset, Dataset]]:
        date_iterator = self.project_date_iterator
        datashape, factuals, counterfactuals = self.get_inputs()
        date_iterator.set_dataset(factuals)
        return ((datashape, factuals, counterfactuals) for _, factuals in date_iterator)


class CounterfactualRegressionMetricRegistry(AbstractMetricRegistry):
    def __init__(self) -> None:
        self._functions: dict[str, CounterfactualMetric] = {}

    def register(self, name: str, func: CounterfactualMetric) -> None:
        self._functions[name] = func

    def __iter__(self) -> Iterator[tuple[str, CounterfactualMetric]]:
        return iter(self._functions.items())

    def get_functions(self) -> dict[str, CounterfactualMetric]:
        return self._functions


counterfactual_metric_registry = CounterfactualRegressionMetricRegistry()


def counterfactual_metric(
    name: str,
) -> Callable[[CounterfactualMetric], CounterfactualMetric]:
    """Decorator to register a function as a model evaluator for A4S.

    Returns:
        Callable[[Evaluator], CounterfactualMetric]: A decorator function that registers the evaluation function as a model evaluator for A4S.
    """

    def func_decorator(func: CounterfactualMetric) -> CounterfactualMetric:
        counterfactual_metric_registry.register(name, func)
        return func

    return func_decorator
