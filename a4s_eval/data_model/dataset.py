"""Data models for representing datasets and their features in the A4S evaluation framework."""

import enum
from pydantic import BaseModel


class FeatureType(str, enum.Enum):
    """Enumeration of supported feature types in the dataset."""
    INTEGER = "integer"  # Discrete numerical values
    FLOAT = "float"     # Continuous numerical values
    CATEGORICAL = "categorical"  # Categorical values


class Feature(BaseModel):
    """Represents a single feature in a dataset with its metadata."""
    id: int             # Unique identifier for the feature
    name: str          # Name of the feature
    feature_type: FeatureType  # Type of the feature (integer, float, or categorical)
    min_value: float   # Minimum value the feature can take
    max_value: float   # Maximum value the feature can take


class Dataset(BaseModel):
    """Represents a complete dataset with its features and metadata."""
    id: int  # Unique identifier for the dataset

    name: str  # Name of the dataset
    train_file_path: str  # Path to the training data file
    test_file_path: str   # Path to the test data file

    features: list[Feature]  # List of features in the dataset
    target: Feature         # Target feature to predict
    date_feature: Feature   # Feature used for temporal information
