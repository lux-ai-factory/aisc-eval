from typing import Callable, Iterator, Protocol

import pandas as pd
import numpy as np
import onnxruntime as ort
import dice_ml

from a4s_eval.data_model.evaluation import Dataset, DataShape, FeatureType, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.abstract import (
    AbstractMetricRegistry,
    MetricInputGenerator,
)
from a4s_eval.service.api_client import post_artifact_dataset
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
        factuals: pd.DataFrame,
        counter_factuals: pd.DataFrame,
        mad_values: dict,
    ) -> list[Measure]:
        """Run a specific model evaluation.

        Args:
            expected_datashape: The expected datashape for the project
            test_dataset: The test dataset to run the evaluation.
            counterfactuals: The counterfactuals that is generated.

        """
        raise NotImplementedError


class CounterfactualsInputGenerator(MetricInputGenerator):
    # Add cache for counterfactuals
    def __init__(self, *largs, **kwargs) -> None:
        super().__init__(*largs, **kwargs)
        self.__counterfactuals: pd.DataFrame | None = None

    @property
    def factuals(self) -> pd.DataFrame:
        X_test = self.test_dataset.data[[f.name for f in self.expected_datashape.features]]
        return  X_test.iloc[:100]  # limit to first 100 for faster testing (hardcoded)

    @property
    def dice_model(self) -> ort.capi.onnxruntime_inference_collection.InferenceSession:
        model_wrapper = ONNXWrapper(self.model_onnx_session)
        dice_model = dice_ml.Model(model=model_wrapper, backend="sklearn", model_type="regressor")
        return dice_model

    @property
    def dice_data(self) -> dice_ml.Data:
        numeric_columns = [feature.name for feature in self.expected_datashape.features if feature.feature_type == FeatureType.FLOAT]
        X_train = self.train_dataset.data[[feature.name for feature in self.expected_datashape.features] + [self.expected_datashape.target.name]]

        dice_data = dice_ml.Data(
            dataframe=X_train,
            continuous_features=list(numeric_columns),
            outcome_name=self.expected_datashape.target.name
        )
        return dice_data


    @property
    def counter_factuals(self) -> pd.DataFrame:
        if self.__counterfactuals is not None:
            return self.__counterfactuals

        feature_cols = [f.name for f in self.expected_datashape.features]
        X_test = self.factuals
        exp = dice_ml.Dice(self.dice_data, self.dice_model, method="genetic")
        
        # Generate all counterfactuals at once
        dice_exp = exp.generate_counterfactuals(
            X_test[feature_cols],
            total_CFs=1,
            desired_range=[self.expected_datashape.target.min_value, 
                        self.expected_datashape.target.max_value]
        )
        
        # Process results
        counterfactuals = []
        for i, cf_example in enumerate(dice_exp.cf_examples_list):
            cf_df = cf_example.final_cfs_df.copy()
            cf_df.index = pd.Index([X_test.index[i]])
            counterfactuals.append(cf_df)
        
        self.__counterfactuals = pd.concat(counterfactuals)
        
        post_artifact_dataset(
            project_pid=self.evaluation.project.pid,
            evaluation_pid=self.eval_pid,
            artifact_name="counter_factual",
            df=self.__counterfactuals
        )
        return self.__counterfactuals


    def get_inputs(self) -> tuple[DataShape, pd.DataFrame, pd.DataFrame, dict]:
        exp = dice_ml.Dice(self.dice_data, self.dice_model, method="genetic")
        mad_values: dict = exp.data_interface.get_valid_mads()
        return (
            self.expected_datashape,
            self.factuals,
            self.counter_factuals,
            mad_values,
        )

    def get_inputs_dateiterator(
        self,
    ) -> Iterator[tuple[DataShape, pd.DataFrame, pd.DataFrame, dict]]:
        # Not sure it will be used in counterfactual metrics
        date_iterator = self.project_date_iterator
        datashape, factuals, counterfactuals, mad_values = self.get_inputs()
        date_iterator.set_dataset(factuals)
        return ((datashape, factuals, counterfactuals, mad_values) for _, factuals in date_iterator)


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
