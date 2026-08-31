from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    model_version: str
    class_labels: list[str]
    num_classes: int
    optimization: str
