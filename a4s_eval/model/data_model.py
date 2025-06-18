import enum
import uuid

import pandas as pd
from pydantic import BaseModel


class FeatureType(str, enum.Enum):
    """Enumeration of supported feature data types.

    Attributes:
        INTEGER: For integer numerical features
        FLOAT: For floating-point numerical features
        CATEGORICAL: For categorical/nominal features
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
        datashape_features_id (int): Foreign key for features relationship.
        datashape_target_id (int): Foreign key for target feature relationship.
        datashape_date_id (int): Foreign key for date feature relationship.

    Relationships:
        datashape_features (DataShape): Relationship to the data shape.
        datashape_target (DataShape): Relationship to the target feature.
        datashape_date (DataShape): Relationship to the date feature.
        metrics (list[Metric]): Relationship to the metrics.
    """

    # Feature attributes
    pid: uuid.UUID
    name: str
    feature_type: FeatureType
    min_value: float
    max_value: float


class DataShape(BaseModel):
    features: list[Feature]
    target: Feature
    date: Feature


class Dataset(BaseModel):
    pid: uuid.UUID
    data: pd.DataFrame
    # shape: DataShape
