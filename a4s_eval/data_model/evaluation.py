import enum
import uuid
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict


class FeatureType(str, enum.Enum):
    """Enumeration of supported feature data types.

    Attributes:
        INTEGER: For integer numerical features
        FLOAT: For floating-point numerical features
        CATEGORICAL: For categorical/nominal features
        DATE: For date features
    """

    INTEGER = "integer"
    FLOAT = "float"
    CATEGORICAL = "categorical"
    DATE = "date"


class Feature(BaseModel):
    """Represents a single feature (column) in a dataset.

    This class defines the properties of individual features,
    including their data type and valid value ranges.

    Attributes:
        id (int): Primary key for the feature.
        name (str): Name of the feature.
        feature_type (FeatureType): Data type of the feature.
        min_value (Optional[float]): Minimum allowed value.
        max_value (Optional[float]): Maximum allowed value.
    """

    # Feature attributes
    name: str
    feature_type: FeatureType
    min_value: float | Any
    max_value: float | Any


class DataShape(BaseModel):
    features: list[Feature]
    target: Feature
    date: Feature


class Dataset(BaseModel):
    pid: uuid.UUID
    shape: DataShape
    data: pd.DataFrame | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

class Model(BaseModel):
    pid: uuid.UUID
    model: pd.DataFrame | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Project(BaseModel):
    name: str
    frequency: str
    window_size: str

class Evaluation(BaseModel):
    pid: uuid.UUID
    dataset: Dataset
    model: Model
    project: Project


