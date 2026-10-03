"""Single-image inference — used by the Streamlit app.

Loads the best model (CNN by default, switchable), applies the shared preprocessing
pipeline, and returns a structured prediction dict.

Usage:
    from src.predict import Predictor
    predictor = Predictor("cnn")          # or "svm", "mobilenet"
    result = predictor.identify(img_bgr)
"""

import json

import cv2
import joblib
import numpy as np

from src.config import MODELS
from src.features import hog_features
from src.preprocess import preprocess_signature


class Predictor:
    """Wraps a trained model for single-image inference.

    Parameters
    ----------
    model_type : str
        One of: "svm" (default), "rf", "knn", "lr", "gb", "cnn", "mobilenet"
        All HOG-based models (svm, rf, knn, lr, gb) use the same preprocessing.
    """

    _SUPPORTED = ("svm", "rf", "knn", "lr", "gb", "cnn", "mobilenet")
    # Classical HOG-based model types
    _CLASSICAL = ("svm", "rf", "knn", "lr", "gb")

    def __init__(self, model_type: str = "svm"):
        if model_type not in self._SUPPORTED:
            raise ValueError(f"model_type must be one of {self._SUPPORTED}")
        self.model_type = model_type
        self._load()

    def _load(self):
        # Class map and threshold are shared across model types
        class_map_path = MODELS / "class_map.json"
        threshold_path = MODELS / "threshold.json"

        if not class_map_path.exists():
            raise FileNotFoundError(
                f"class_map.json not found at {class_map_path}. "
                "Run `python -m src.dataset` first."
            )

        raw_map = json.load(open(class_map_path))
        first_k, first_v = next(iter(raw_map.items()))
        if isinstance(first_v, int):
            self.classes = {int(v): str(k) for k, v in raw_map.items()}
        else:
            self.classes = {int(k): str(v) for k, v in raw_map.items()}
        self.n_classes = len(self.classes)

        self.tau = 0.50  # default fallback
        if threshold_path.exists():
            self.tau = json.load(open(threshold_path))["tau"]

        # Load the correct model
        if self.model_type == "svm":
            path = MODELS / "svm.joblib"
            if not path.exists():
                raise FileNotFoundError(f"SVM model not found at {path}.")
            self._model = joblib.load(path)
        elif self.model_type in ("rf", "knn", "lr", "gb"):
            # Best non-SVM classical model (Random Forest by default)
            path = MODELS / "best_classical.joblib"
            if not path.exists():
                raise FileNotFoundError(
                    f"best_classical.joblib not found. Run `python -m src.train_classical` first."
                )
            self._model = joblib.load(path)
        elif self.model_type == "cnn":
            weights_path = MODELS / "cnn.weights.h5"
            keras_path = MODELS / "cnn.keras"
            if weights_path.exists():
                from src.models import build_cnn
                self._model = build_cnn(n_classes=self.n_classes, use_aug=False)
                self._model.load_weights(str(weights_path))
            elif keras_path.exists():
                import keras
                self._model = keras.models.load_model(str(keras_path))
            else:
                raise FileNotFoundError(f"CNN model not found at {weights_path} or {keras_path}.")
        elif self.model_type == "mobilenet":
            weights_path = MODELS / "mobilenet.weights.h5"
            keras_path = MODELS / "mobilenet.keras"
            if weights_path.exists():
                from src.models import build_mobilenet
                self._model = build_mobilenet(n_classes=self.n_classes, use_aug=False)
                self._model.load_weights(str(weights_path))
            elif keras_path.exists():
                import keras
                self._model = keras.models.load_model(str(keras_path))
            else:
                raise FileNotFoundError(f"MobileNet model not found at {weights_path} or {keras_path}.")

    def identify(self, img_bgr: np.ndarray) -> dict:
        """Identify the signer of a signature image.

        Parameters
        ----------
        img_bgr : np.ndarray   BGR uint8 image (from cv2.imread or cv2.imdecode)

        Returns
        -------
        dict with keys:
            prediction   str   — student ID (e.g. "S07") or "Not recognised"
            confidence   float — max softmax probability
            top3         list  — [(student_id, prob), (…), (…)]
            recognised   bool
            preprocessed np.ndarray  — (128, 128) float32 for display
        """
        # Preprocessing (shared with training)
        arr = preprocess_signature(img_bgr)           # (128, 128) float32

        # Feature extraction and prediction
        if self.model_type in self._CLASSICAL:
            x = arr[np.newaxis, ..., np.newaxis]      # (1, 128, 128, 1)
            probs = self._model.predict_proba(hog_features(x))[0]
        else:
            x = arr[np.newaxis, ..., np.newaxis]      # (1, 128, 128, 1)
            probs = self._model.predict(x, verbose=0)[0]

        # Build top-3
        top_idx = np.argsort(-probs)[:3]
        top3 = [(self.classes[i], float(probs[i])) for i in top_idx]

        max_conf = float(probs.max())
        recognised = max_conf >= self.tau

        return {
            "prediction":   top3[0][0] if recognised else "Not recognised",
            "confidence":   max_conf,
            "top3":         top3,
            "recognised":   recognised,
            "preprocessed": arr,
        }


# ── Module-level convenience function ──────────────────────
# Kept for backwards-compatibility with any inline usage.

def identify(img_bgr: np.ndarray, model_type: str = "svm") -> dict:
    """Convenience wrapper — creates a Predictor and calls identify().

    For repeated calls (e.g. in the app) use Predictor directly so the model
    is loaded only once.
    """
    return Predictor(model_type).identify(img_bgr)
