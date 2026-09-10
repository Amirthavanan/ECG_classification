"""
ECG Image Preprocessing, Feature Engineering, and Feature Extraction Pipeline
=============================================================================
This module handles all preprocessing and feature extraction for ECG images
uploaded by the user, matching the exact pipeline used during model training:

1. Image decoding and RGB standardization (3 channels)
2. Resizing to target dimensions (224, 224)
3. Pixel rescaling / normalization using tf.keras.layers.Rescaling(1./255)
   mapping [0, 255] -> [0.0, 1.0]
4. Feature Engineering:
   - Statistical intensity metrics (mean, std, min, max, variance)
   - Color / channel distributions (R, G, B channels)
   - ECG waveform projection profiles (horizontal & vertical projections, trace density)
5. CNN Feature Extraction:
   - Intermediate feature representations across Conv Block 1, 2, 3
   - 128-dimensional dense feature embedding
   - NO model training (no .fit(), no backprop, no optimizer)
"""

import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
import tensorflow as tf

# Standard training parameters
IMG_SIZE = (224, 224)
IMG_WIDTH, IMG_HEIGHT = IMG_SIZE
CHANNELS = 3

# Exact normalization layer used in model training:
# Rescaling(1./255): 0 -> 0.0, 128 -> ~0.502, 255 -> 1.0
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255.0)


class ECGImagePreprocessor:
    """
    Performs preprocessing, feature engineering, and feature extraction for
    uploaded ECG images without performing model training.
    """

    def __init__(self, target_size: Tuple[int, int] = IMG_SIZE):
        self.target_size = target_size
        self._feature_extractor_model = None

    def load_image(
        self, image_input: Union[bytes, io.BytesIO, str, Path, Image.Image, np.ndarray]
    ) -> Image.Image:
        """
        Loads an uploaded image from multiple supported formats (bytes, file path,
        or PIL Image) and standardizes it to 3-channel RGB.
        """
        if isinstance(image_input, Image.Image):
            image = image_input
        elif isinstance(image_input, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image_input))
        elif isinstance(image_input, io.BytesIO):
            image = Image.open(image_input)
        elif isinstance(image_input, (str, Path)):
            image = Image.open(str(image_input))
        elif isinstance(image_input, np.ndarray):
            # Handle numpy array input
            if image_input.dtype != np.uint8:
                clipped = np.clip(image_input * 255.0 if image_input.max() <= 1.0 else image_input, 0, 255)
                image = Image.fromarray(clipped.astype(np.uint8))
            else:
                image = Image.fromarray(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Ensure standard 3-channel RGB
        return image.convert("RGB")

    def resize(self, image: Image.Image) -> Image.Image:
        """
        Resizes the image to target size (224, 224) matching image_dataset_from_directory.
        Uses bilinear interpolation by default.
        """
        return image.resize(self.target_size, Image.Resampling.BILINEAR)

    def normalize(self, image_array: np.ndarray) -> np.ndarray:
        """
        Applies pixel rescaling from [0, 255] to [0.0, 1.0] using the exact
        normalization layer used in training: tf.keras.layers.Rescaling(1./255).
        """
        tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)
        normalized_tensor = normalization_layer(tensor)
        return normalized_tensor.numpy()

    def prepare_tensor(
        self, image_input: Union[bytes, io.BytesIO, str, Path, Image.Image, np.ndarray]
    ) -> np.ndarray:
        """
        Executes the full image preparation sequence:
        Load -> RGB Convert -> Resize (224, 224) -> Rescale (1/255) -> Expand batch dim.
        
        Returns:
            np.ndarray of shape (1, 224, 224, 3) with float32 values in [0.0, 1.0].
        """
        raw_image = self.load_image(image_input)
        resized_image = self.resize(raw_image)
        array = np.asarray(resized_image, dtype=np.float32)
        normalized_array = self.normalize(array)
        batch_tensor = np.expand_dims(normalized_array, axis=0)
        return batch_tensor

    def extract_engineered_features(self, image_array: np.ndarray) -> Dict[str, Any]:
        """
        Extracts engineered visual and signal-profile features from the ECG image array:
        - Global pixel intensity statistics (mean, std, min, max, median, variance)
        - Channel-specific metrics (R, G, B)
        - ECG waveform projection profiles (horizontal & vertical trace energy)
        - Trace density and contrast estimates
        """
        # Ensure array in [0.0, 1.0]
        if image_array.max() > 1.0:
            arr = image_array.astype(np.float32) / 255.0
        else:
            arr = image_array.astype(np.float32)

        # Grayscale approximation (Rec. 601 luma)
        if arr.ndim == 3 and arr.shape[-1] == 3:
            gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        else:
            gray = arr.squeeze()

        # Global statistics
        intensity_stats = {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
            "variance": round(float(np.var(arr)), 4),
            "median": round(float(np.median(arr)), 4),
        }

        # Channel statistics
        channel_stats = {}
        if arr.ndim >= 3 and arr.shape[-1] >= 3:
            for idx, ch_name in enumerate(["red", "green", "blue"]):
                channel_stats[ch_name] = {
                    "mean": round(float(np.mean(arr[:, :, idx])), 4),
                    "std": round(float(np.std(arr[:, :, idx])), 4),
                }

        # Waveform projection profiles
        # In ECG paper records, the trace is usually darker than background (or lighter on dark grid)
        # We invert grayscale so trace waveforms correspond to positive peaks
        trace_inverted = 1.0 - gray
        horizontal_proj = np.mean(trace_inverted, axis=1)  # row-wise profile (voltage distribution across leads)
        vertical_proj = np.mean(trace_inverted, axis=0)    # column-wise profile (temporal waveform energy)

        # Trace density / active area estimation (pixels significantly different from background)
        background_est = np.median(gray)
        active_pixels = np.abs(gray - background_est) > 0.15
        trace_density_pct = round(float(np.mean(active_pixels) * 100.0), 2)

        waveform_profiles = {
            "trace_density_pct": trace_density_pct,
            "horizontal_profile_peak_row": int(np.argmax(horizontal_proj)),
            "horizontal_profile_mean": round(float(np.mean(horizontal_proj)), 4),
            "vertical_profile_peak_col": int(np.argmax(vertical_proj)),
            "vertical_profile_mean": round(float(np.mean(vertical_proj)), 4),
            "contrast_ratio": round(float(np.max(gray) - np.min(gray)), 4),
        }

        return {
            "intensity_statistics": intensity_stats,
            "channel_statistics": channel_stats,
            "waveform_profile": waveform_profiles,
        }

    def build_cnn_feature_extractor(
        self, base_model: Optional[tf.keras.Model] = None
    ) -> tf.keras.Model:
        """
        Standalone architectural feature extractor (matching the user's CNN definition)
        used when no pre-loaded model is provided.
        NOTE: No training or weight updates are performed.
        """
        inputs = tf.keras.layers.Input(shape=(self.target_size[0], self.target_size[1], CHANNELS), name="ecg_input")
        
        # Conv Block 1
        x1 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv2d_block1")(inputs)
        p1 = tf.keras.layers.MaxPooling2D(name="maxpool_block1")(x1)

        # Conv Block 2
        x2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_block2")(p1)
        p2 = tf.keras.layers.MaxPooling2D(name="maxpool_block2")(x2)

        # Conv Block 3
        x3 = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", name="conv2d_block3")(p2)
        p3 = tf.keras.layers.MaxPooling2D(name="maxpool_block3")(x3)

        # Flatten
        f = tf.keras.layers.Flatten(name="flatten_features")(p3)

        # Dense 128
        dense_feat = tf.keras.layers.Dense(128, activation="relu", name="dense_128_embedding")(f)

        extractor = tf.keras.Model(
            inputs=inputs,
            outputs={
                "conv_block_1": p1,
                "conv_block_2": p2,
                "conv_block_3": p3,
                "flatten": f,
                "dense_embedding": dense_feat,
            },
            name="ecg_cnn_feature_extractor",
        )
        extractor.trainable = False
        return extractor

    def extract_cnn_features(
        self,
        batch_tensor: np.ndarray,
        model: Optional[tf.keras.Model] = None,
    ) -> Dict[str, Any]:
        """
        Runs intermediate feature extraction on the preprocessed batch tensor.
        Produces dimensional shapes, activation statistics, and dense feature vectors.
        Strictly inference-mode (training=False), NO backpropagation or model updates.
        """
        feature_summary = {}

        # If a trained model is provided, extract features by passing through its layers
        if model is not None:
            x = tf.convert_to_tensor(batch_tensor, dtype=tf.float32)
            for layer in model.layers:
                x = layer(x, training=False)
                layer_name = layer.name

                # Exclude the final 5-class classification output layer from feature maps
                if getattr(layer, "units", None) == 5 and "dense" in layer_name.lower():
                    continue

                t_np = x.numpy()
                feature_summary[layer_name] = {
                    "shape": list(t_np.shape),
                    "mean_activation": round(float(np.mean(t_np)), 6),
                    "max_activation": round(float(np.max(t_np)), 6),
                    "sparsity_pct": round(float(np.mean(t_np == 0.0) * 100.0), 2),
                }

                # Include a sample 16-dimensional embedding vector for the dense feature layer
                if "dense" in layer_name.lower() and t_np.ndim == 2:
                    feature_summary["dense_embedding_sample_16d"] = [
                        round(float(v), 4) for v in t_np[0][:16]
                    ]

            return feature_summary

        # Fallback to standalone architectural feature extractor
        if self._feature_extractor_model is None:
            self._feature_extractor_model = self.build_cnn_feature_extractor()

        features = self._feature_extractor_model(batch_tensor, training=False)

        if isinstance(features, dict):
            for key, tensor in features.items():
                t_np = tensor.numpy()
                feature_summary[key] = {
                    "shape": list(t_np.shape),
                    "mean_activation": round(float(np.mean(t_np)), 6),
                    "max_activation": round(float(np.max(t_np)), 6),
                    "sparsity_pct": round(float(np.mean(t_np == 0.0) * 100.0), 2),
                }
                if key == "dense_embedding":
                    feature_summary["dense_embedding_sample_16d"] = [
                        round(float(v), 4) for v in t_np[0][:16]
                    ]

        return feature_summary

    def process(
        self,
        image_input: Union[bytes, io.BytesIO, str, Path, Image.Image, np.ndarray],
        model: Optional[tf.keras.Model] = None,
        extract_cnn: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive preprocessing method for an uploaded ECG image:
        - Reads & decodes original image
        - Converts to RGB
        - Resizes to (224, 224)
        - Normalizes using Rescaling(1./255)
        - Extracts engineered features
        - Extracts CNN intermediate features (without any model training)

        Returns:
            A comprehensive dictionary containing:
            - "preprocessed_tensor": np.ndarray of shape (1, 224, 224, 3)
            - "metadata": input details, target shape, normalization info
            - "engineered_features": statistical and waveform features
            - "cnn_features": deep feature representations across layers
        """
        raw_img = self.load_image(image_input)
        orig_width, orig_height = raw_img.size

        # Preprocessing matching model training
        resized_img = self.resize(raw_img)
        array_raw = np.asarray(resized_img, dtype=np.float32)
        normalized_array = self.normalize(array_raw)
        batch_tensor = np.expand_dims(normalized_array, axis=0)

        # Feature engineering
        engineered = self.extract_engineered_features(normalized_array)

        # CNN Feature extraction
        cnn_features = {}
        if extract_cnn:
            cnn_features = self.extract_cnn_features(batch_tensor, model=model)

        return {
            "preprocessed_tensor": batch_tensor,
            "metadata": {
                "original_dimensions": [orig_width, orig_height],
                "preprocessed_shape": list(batch_tensor.shape),
                "channels": CHANNELS,
                "color_mode": "RGB",
                "pixel_range": [
                    round(float(np.min(normalized_array)), 4),
                    round(float(np.max(normalized_array)), 4),
                ],
                "normalization_applied": "tf.keras.layers.Rescaling(1./255)",
                "batch_size": 1,
            },
            "engineered_features": engineered,
            "cnn_features": cnn_features,
        }


# Singleton preprocessor instance for application-wide reuse
preprocessor = ECGImagePreprocessor()


def preprocess_ecg_image(
    image_input: Union[bytes, io.BytesIO, str, Path, Image.Image, np.ndarray],
    model: Optional[tf.keras.Model] = None,
    extract_cnn: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to preprocess an uploaded ECG image and extract features.
    No model training is performed.
    """
    return preprocessor.process(
        image_input=image_input, model=model, extract_cnn=extract_cnn
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess an ECG image and extract features without training."
    )
    parser.add_argument("image_path", type=str, help="Path to the ECG image file.")
    args = parser.parse_args()

    file_path = Path(args.image_path)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        exit(1)

    print(f"\nProcessing ECG image: {file_path}")
    result = preprocess_ecg_image(file_path)

    print("\n[✓] Preprocessing Metadata:")
    for k, v in result["metadata"].items():
        print(f"  - {k}: {v}")

    print("\n[✓] Engineered Intensity Features:")
    for k, v in result["engineered_features"]["intensity_statistics"].items():
        print(f"  - {k}: {v}")

    print("\n[✓] ECG Waveform Profile:")
    for k, v in result["engineered_features"]["waveform_profile"].items():
        print(f"  - {k}: {v}")

    print("\n[✓] CNN Feature Layers:")
    for layer, details in result["cnn_features"].items():
        if isinstance(details, dict) and "shape" in details:
            print(f"  - {layer}: shape {details['shape']}, mean_act={details['mean_activation']}")
        else:
            print(f"  - {layer}: {details}")

    print("\nPreprocessing and feature extraction completed successfully (NO training was performed).")
