import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from .config import (
    MODEL_PATH,
    CLASSES_PATH,
    IMG_WIDTH,
    IMG_HEIGHT,
)

class ECGModelService:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.loaded = False
        self.error = None
        self.load()

    def load(self):
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model not found: {MODEL_PATH}. "
                    "Copy ecg_cnn_model.keras into backend/models/."
                )

            if not CLASSES_PATH.exists():
                raise FileNotFoundError(
                    f"Classes file not found: {CLASSES_PATH}."
                )

            with open(CLASSES_PATH, "r", encoding="utf-8") as f:
                self.class_names = json.load(f)

            self.model = tf.keras.models.load_model(MODEL_PATH)

            output_units = self.model.output_shape[-1]
            if output_units != len(self.class_names):
                raise ValueError(
                    f"Model outputs {output_units} classes, but "
                    f"classes.json contains {len(self.class_names)} classes."
                )

            self.loaded = True
            self.error = None

        except Exception as exc:
            self.loaded = False
            self.error = str(exc)

    def preprocess(self, image_bytes: bytes):
        from io import BytesIO

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))

        array = np.asarray(image, dtype=np.float32) / 255.0
        array = np.expand_dims(array, axis=0)

        return array

    def predict(self, image_bytes: bytes):
        if not self.loaded:
            raise RuntimeError(self.error or "Model is not loaded.")

        tensor = self.preprocess(image_bytes)

        probabilities = self.model.predict(tensor, verbose=0)[0]
        probabilities = probabilities.astype(float)

        predicted_index = int(np.argmax(probabilities))
        predicted_class = self.class_names[predicted_index]
        confidence = float(probabilities[predicted_index])

        probability_map = {
            name: round(float(probabilities[i]), 6)
            for i, name in enumerate(self.class_names)
        }

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probability_map,
            "model_input_size": [IMG_WIDTH, IMG_HEIGHT],
        }

model_service = ECGModelService()
