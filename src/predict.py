"""Single-image inference for signature identification.

Supports all 5 trained classical models (Linear SVM, Random Forest, Logistic Regression,
KNN, Gradient Boosting) using HOG feature representation, plus optional CNN backbones.
"""

import json
from pathlib import Path
import cv2
import joblib
import numpy as np

import src.compat  # noqa: F401 — register sklearn version compatibility shims
from src.config import MODELS
from src.features import hog_features
from src.preprocess import preprocess_signature

MODEL_METADATA = {
    "svm": {"name": "Linear SVM", "acc": "80.0%", "desc": "HOG features"},
    "rf":  {"name": "Random Forest", "acc": "80.0%", "desc": "HOG + ensemble"},
    "lr":  {"name": "Logistic Regression", "acc": "73.3%", "desc": "HOG + linear"},
    "knn": {"name": "K-Nearest Neighbors", "acc": "70.0%", "desc": "HOG + distance (k=3)"},
    "gb":  {"name": "Gradient Boosting", "acc": "73.3%", "desc": "HOG + boosted trees"},
    "cnn": {"name": "Custom CNN", "acc": "17.0%", "desc": "3 conv blocks"},
}


class Predictor:
    """Wraps a trained model for single-image inference."""

    _SUPPORTED = ("svm", "rf", "knn", "lr", "gb", "cnn", "mobilenet")
    _CLASSICAL = ("svm", "rf", "knn", "lr", "gb")

    def __init__(self, model_type: str = "svm"):
        if model_type not in self._SUPPORTED:
            raise ValueError(f"model_type must be one of {self._SUPPORTED}")
        self.model_type = model_type
        self._load()

    def _load(self):
        class_map_path = MODELS / "class_map.json"
        threshold_path = MODELS / "threshold.json"

        if not class_map_path.exists():
            raise FileNotFoundError(f"class_map.json not found at {class_map_path}")

        raw_map = json.load(open(class_map_path))
        first_k, first_v = next(iter(raw_map.items()))
        if isinstance(first_v, int):
            self.classes = {int(v): str(k) for k, v in raw_map.items()}
        else:
            self.classes = {int(k): str(v) for k, v in raw_map.items()}
        self.n_classes = len(self.classes)

        self.tau = 0.25
        if threshold_path.exists():
            try:
                self.tau = float(json.load(open(threshold_path)).get("tau", 0.25))
            except Exception:
                self.tau = 0.25

        # Load models
        if self.model_type == "svm":
            path = MODELS / "svm.joblib"
            if not path.exists():
                raise FileNotFoundError(f"SVM model not found at {path}")
            self._model = joblib.load(path)
        elif self.model_type == "rf":
            path = MODELS / "rf.joblib"
            if not path.exists():
                path = MODELS / "best_classical.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Random Forest model not found at {path}")
            self._model = joblib.load(path)
        elif self.model_type == "knn":
            path = MODELS / "knn.joblib"
            if not path.exists():
                raise FileNotFoundError(f"KNN model not found at {path}")
            self._model = joblib.load(path)
        elif self.model_type == "lr":
            path = MODELS / "lr.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Logistic Regression model not found at {path}")
            self._model = joblib.load(path)
        elif self.model_type == "gb":
            path = MODELS / "gb.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Gradient Boosting model not found at {path}")
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
                raise FileNotFoundError(f"CNN model weights not found.")
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
                raise FileNotFoundError(f"MobileNet model weights not found.")

    def _calibrate_probs(self, raw_probs: np.ndarray) -> np.ndarray:
        """Apply temperature calibration for SVM/GB to prevent artificial multiclass flattening."""
        if self.model_type in ("svm", "gb"):
            # Temperature scaling T = 0.35 provides sharp calibrated discrimination
            logits = np.log(np.maximum(raw_probs, 1e-7)) / 0.35
            exp_l = np.exp(logits - logits.max())
            return exp_l / exp_l.sum()
        return raw_probs

    def identify(self, img_bgr: np.ndarray) -> dict:
        """Identify the signer of a signature image."""
        arr = preprocess_signature(img_bgr)
        ink_count = int(np.sum(arr > 0))
        if ink_count < 15:
            return {
                "prediction":   "No signature detected",
                "candidate":    "None",
                "confidence":   0.0,
                "top3":         [(self.classes.get(i, f"S{i+1:02d}"), 0.0) for i in range(min(3, self.n_classes))],
                "recognised":   False,
                "preprocessed": arr,
            }

        if self.model_type in self._CLASSICAL:
            x = arr[np.newaxis, ..., np.newaxis]
            raw_probs = self._model.predict_proba(hog_features(x))[0]
            probs = self._calibrate_probs(raw_probs)
        else:
            x = arr[np.newaxis, ..., np.newaxis]
            probs = self._model.predict(x, verbose=0)[0]

        probs = np.asarray(probs, dtype=np.float64)
        if probs.sum() > 0:
            probs = probs / probs.sum()
        probs = np.clip(probs, 0.0, 1.0)

        top_idx = np.argsort(-probs)[:3]
        top3 = [(self.classes.get(int(i), f"S{int(i)+1:02d}"), float(probs[i])) for i in top_idx]

        max_conf = float(probs.max())
        recognised = max_conf >= self.tau

        return {
            "prediction":   top3[0][0] if recognised else f"Uncertain ({top3[0][0]})",
            "candidate":    top3[0][0],
            "confidence":   round(max_conf, 4),
            "top3":         top3,
            "recognised":   recognised,
            "preprocessed": arr,
        }


def identify_all_models(img_bgr: np.ndarray, tau: float = 0.25) -> list:
    """Run all available classical models on the same image using a single HOG extraction."""
    arr = preprocess_signature(img_bgr)
    ink_count = int(np.sum(arr > 0))
    if ink_count < 15:
        return []

    x = arr[np.newaxis, ..., np.newaxis]
    feats = hog_features(x)

    class_map_path = MODELS / "class_map.json"
    raw_map = json.load(open(class_map_path))
    first_k, first_v = next(iter(raw_map.items()))
    if isinstance(first_v, int):
        classes = {int(v): str(k) for k, v in raw_map.items()}
    else:
        classes = {int(k): str(v) for k, v in raw_map.items()}

    results = []
    models_to_run = [
        ("rf",  "Random Forest",       "80.0%", "rf.joblib",   "best_classical.joblib"),
        ("svm", "Linear SVM",          "80.0%", "svm.joblib",  None),
        ("lr",  "Logistic Regression", "73.3%", "lr.joblib",   None),
        ("knn", "KNN (k=3)",           "70.0%", "knn.joblib",  None),
        ("gb",  "Gradient Boosting",   "73.3%", "gb.joblib",   None),
    ]

    for m_id, name, acc, primary_file, fallback_file in models_to_run:
        p = MODELS / primary_file
        if not p.exists() and fallback_file:
            p = MODELS / fallback_file
        if not p.exists():
            continue

        try:
            m = joblib.load(p)
            raw_p = m.predict_proba(feats)[0]
            if m_id in ("svm", "gb"):
                logits = np.log(np.maximum(raw_p, 1e-7)) / 0.35
                exp_l = np.exp(logits - logits.max())
                probs = exp_l / exp_l.sum()
            else:
                probs = raw_p

            probs = np.asarray(probs, dtype=np.float64)
            if probs.sum() > 0:
                probs = probs / probs.sum()
            probs = np.clip(probs, 0.0, 1.0)

            top1_idx = int(probs.argmax())
            conf = float(probs[top1_idx])
            pred_id = classes.get(top1_idx, f"S{top1_idx+1:02d}")
            rec = conf >= tau

            results.append({
                "id": m_id,
                "name": name,
                "accuracy": acc,
                "prediction": pred_id if rec else f"Uncertain ({pred_id})",
                "candidate": pred_id,
                "raw_pred": pred_id,
                "confidence": round(conf, 4),
                "recognised": rec,
            })
        except Exception as e:
            print(f"Error evaluating {m_id}: {e}")

    return results
