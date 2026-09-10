from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import MAX_FILE_SIZE_BYTES
from .model_service import model_service
from .schemas import PredictionResponse, PreprocessResponse

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(
    title="ECG CNN Classification API",
    version="1.0.0",
    description="FastAPI inference API for a TensorFlow/Keras ECG image classifier.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "service": "ECG CNN Classification API",
        "status": "running",
        "model_loaded": model_service.loaded,
    }

@app.get("/api/health")
def health():
    return {
        "status": "healthy" if model_service.loaded else "degraded",
        "model_loaded": model_service.loaded,
        "classes": model_service.class_names,
        "error": model_service.error,
    }

@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Upload a valid ECG image: JPG, PNG, WEBP, or BMP.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum size is "
                   f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    try:
        return model_service.predict(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.post("/api/preprocess", response_model=PreprocessResponse)
async def preprocess_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded ECG image and performs all the preprocessing,
    feature engineering, and CNN feature extraction steps done for model
    training, without performing any model training.
    """
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Upload a valid ECG image: JPG, PNG, WEBP, or BMP.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum size is "
                   f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
        )

    try:
        result = model_service.preprocess_and_extract(image_bytes)
        return {
            "status": "success",
            "message": "Image preprocessed successfully without model training.",
            "metadata": result["metadata"],
            "engineered_features": result["engineered_features"],
            "cnn_features": result["cnn_features"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
