from pydantic import BaseModel, Field
from typing import Dict, List

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float = Field(ge=0, le=1)
    probabilities: Dict[str, float]
    model_input_size: List[int]
