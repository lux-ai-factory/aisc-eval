"""Data model for representing projects in the VERA evaluation framework.

This module defines the Project class which serves as a container for configurations
and associations between datasets, models, and their evaluation settings.
"""

import uuid
from typing import Literal

from pydantic import BaseModel, model_validator

from aisc_plugin_interface import InputType


class InputFile(BaseModel):
    pid: str
    name: str
    data: str


#: the input_type of a helper model of the platform (a judge, an attacker, ...): named by its reference
#: (helper:<platform project>/<name>), with no file; run_plugin resolves it on the platform
HELPER_INPUT = "helper"


class InputFileDefinition(BaseModel):
    """A run's input as the engine sends it: a component of the AI system with its file, or a helper model."""
    name: str
    input_type: InputType | Literal["helper"]
    input_file: InputFile | None = None
    helper_ref: str | None = None

    @model_validator(mode="after")
    def _file_or_helper(self) -> "InputFileDefinition":
        if self.input_type == HELPER_INPUT:
            if not self.helper_ref:
                raise ValueError(f"input {self.name}: a helper input names its helper (helper_ref)")
        elif self.input_file is None:
            raise ValueError(f"input {self.name}: a {self.input_type} input needs its file")
        return self


class PluginConfig(BaseModel):
    id: int
    config: dict
    created_at: str
    project_config_selections: list[dict] = []


class Plugin(BaseModel):
    pid: uuid.UUID
    name: str
    package_name: str
    version: str
    plugin_config: PluginConfig | None = None
    evaluation_inputs: list[InputFileDefinition] = []


class Project(BaseModel):
    """Represents a machine learning project with its evaluation configuration.

    This class defines the settings for how a dataset should be evaluated over time,
    including the frequency of evaluations and the size of the time window to analyze.
    It also maintains a reference to the associated dataset.
    """

    name: str  # Name of the project
    plugins: list[Plugin]
