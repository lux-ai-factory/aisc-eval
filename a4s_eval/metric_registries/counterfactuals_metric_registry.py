from typing import Callable, Iterator, Protocol

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


class CounterfactualMetric(Protocol):
    def __call__(
        self,
        datashape: DataShape,
        model: Model,
        reference: Dataset,
        evaluated: Dataset,
    ) -> list[Measure]:
        """Run a specific model evaluation.

        Args:
            model: The model to run the evaluation.
            reference: The reference dataset to run the evaluation.
            evaluated: The evaluated dataset.

        """
        raise NotImplementedError


class CounterfactualsInputGenerator(MetricInputGenerator):
    def get_inputs(self) -> tuple[DataShape, Dataset, Dataset]:
        session = self.model_onnx_session

        x_test = self.test_dataset.data
        if x_test is None:
            raise ValueError("Test dataset data is None.")

        x_train = self.train_dataset.data
        scaler = StandardScaler()
        scaler.fit(x_train.drop(columns=self.expected_datashape.target.name))

        numeric_columns = x_train.select_dtypes(include=["number"]).columns

        dice_data = dice_ml.Data(
            dataframe=x_train,
            continuous_features=list(numeric_columns),
            outcome_name=self.expected_datashape.target.name
        )
        # Sklearn wrapping
        dice_model = dice_ml.Model(model=session, backend="sklearn", model_type="regressor")

        # Initiate DiCE
        exp = Dice(dice_data, dice_model, method="genetic")  # "random" ou "genetic" ou "kd"

        counterfactual = []
        n_samples = len(x_test.iloc[:10,:]) # Limit to first 10 samples for efficiency
        
        target_col = self.expected_datashape.target.name

        # Min/max of target for desired_range
        target_min = x_train[target_col].min()
        target_max = x_train[target_col].max()

        for i in range(min(n_samples, len(x_test))):
            query_instance = x_test.drop(columns=target_col).iloc[i:i+1].astype(np.float32)

            # Standardize factual
            factual_scaled = pd.DataFrame(
                scaler.transform(query_instance),
                columns=query_instance.columns,
                index=query_instance.index
            )

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

        counterfactuals = pd.concat(counterfactual_scaled)

        get_logger().info("Computation finished for counterfactuals.")
        return (
            self.expected_datashape,
            factual_scaled,
            counterfactuals
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
