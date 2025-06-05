"""Data model for representing evaluation metrics and their associated metadata."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class Metric(BaseModel):
    """Represents a single evaluation metric with its value and associated metadata.
    
    This class is used to store various types of metrics including model performance metrics,
    data drift metrics, and feature-specific metrics. Each metric is timestamped and can be
    associated with a model, feature, or dataset through their respective IDs.
    """
    name: str          # Name of the metric (e.g., 'accuracy', 'f1_score', 'drift')
    score: float      # Numerical value of the metric
    time: datetime    # Timestamp when the metric was computed

    # Optional associations
    model_id: Optional[int] = None     # ID of the associated model, if applicable
    feature_id: Optional[int] = None   # ID of the associated feature, if applicable
    dataset_id: Optional[int] = None   # ID of the associated dataset, if applicable

    class Config:
        """Pydantic configuration for handling datetime serialization."""
        # Pydantic uses `isoformat()` for datetime by default
        json_encoders = {
            datetime: lambda v: v.isoformat()  # Converts datetime to ISO 8601 string format
        }
