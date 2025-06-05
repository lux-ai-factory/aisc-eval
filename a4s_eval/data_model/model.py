"""Data model for representing machine learning models in the A4S evaluation framework."""

from pydantic import BaseModel, Field


class Model(BaseModel):
    """Represents a machine learning model with its metadata.
    
    This class stores information about a model, including its identifier,
    name, and optional file path where the model is stored.
    """
    id: int                          # Unique identifier for the model
    name: str                       # Name of the model
    file_path: str | None = Field(default=None)  # Optional path to the model file
