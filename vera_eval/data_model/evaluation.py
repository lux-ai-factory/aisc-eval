import uuid

from pydantic import BaseModel

from vera_eval.data_model.project import Plugin

class Project(BaseModel):
    pid: uuid.UUID
    name: str

class Evaluation(BaseModel):
    pid: uuid.UUID
    project: Project
    evaluation_plugins: list[Plugin]
