import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / os.getenv(
    "MODEL_PATH",
    "models/ecg_cnn_model.keras"
)

CLASSES_PATH = BASE_DIR / os.getenv(
    "CLASSES_PATH",
    "models/classes.json"
)

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "10"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

IMG_WIDTH = 224
IMG_HEIGHT = 224
