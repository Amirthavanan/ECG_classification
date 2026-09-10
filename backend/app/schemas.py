from typing import Any, Dict, List
from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    probabilities: Dict[str, float]
    model_input_size: List[int]

class PreprocessResponse(BaseModel):
    status: str = "success"
    message: str = "Image preprocessed successfully without model training."
    metadata: Dict[str, Any]
    engineered_features: Dict[str, Any]
    cnn_features: Dict[str, Any]

