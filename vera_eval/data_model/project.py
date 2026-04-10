"""Data model for representing projects in the VERA evaluation framework.

This module defines the Project class which serves as a container for configurations
and associations between datasets, models, and their evaluation settings.
"""
import enum

from pydantic import BaseModel

from a4s_plugin_interface import InputType

class InputFile(BaseModel):
    pid: str
    name: str
    data: str

class InputFileDefinition(BaseModel):
    name: str
    input_type: InputType
    input_file: InputFile

class PluginConfig(BaseModel):
    id: int
    config : dict
    created_at: str

class Plugin(BaseModel):
    name: str
    plugin_config: PluginConfig | None = None
    input_files: list[InputFileDefinition] = []

class Project(BaseModel):
    """Represents a machine learning project with its evaluation configuration.

    This class defines the settings for how a dataset should be evaluated over time,
    including the frequency of evaluations and the size of the time window to analyze.
    It also maintains a reference to the associated dataset.
    """

    name: str  # Name of the project
    plugins: list[Plugin]
