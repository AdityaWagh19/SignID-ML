"""FastAPI backend — serves model predictions for the React frontend.

Endpoints:
    POST /api/predict          — upload image, returns prediction JSON
    GET  /api/samples          — list available test sample filenames
    GET  /api/samples/{name}   — serve a sample image file
    GET  /api/models           — list available trained models

Run (from repo root):
    uvicorn app.api:app --reload --port 8000
"""

import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Allow repo root imports
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATASET, MODELS
from src.predict import Predictor, identify_all_models


app = FastAPI(title="Signature Identification API", version="1.0.0")

# Allow the Vite dev server, local origins, and deployed Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "https://signid.vercel.app",
        # allow all vercel preview URLs for this project
    ],
    allow_origin_regex=r"https://signid.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────
# Model cache  (loaded once per model type)
# ──────────────────────────────────────────────────────────
_predictors: dict[str, Predictor] = {}


def _get_predictor(model_type: str) -> Predictor:
    if model_type not in _predictors:
        _predictors[model_type] = Predictor(model_type)
    return _predictors[model_type]


# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────

def _available_models() -> list[str]:
    """List all available models in priority order."""
    available = []
    # 5 Classical models
    if (MODELS / "rf.joblib").exists() or (MODELS / "best_classical.joblib").exists():
        available.append("rf")
    if (MODELS / "svm.joblib").exists():
        available.append("svm")
    if (MODELS / "lr.joblib").exists():
        available.append("lr")
    if (MODELS / "knn.joblib").exists():
        available.append("knn")
    if (MODELS / "gb.joblib").exists():
        available.append("gb")
    # Optional deep learning models
    if (MODELS / "cnn.weights.h5").exists() or (MODELS / "cnn.keras").exists():
        available.append("cnn")
    if (MODELS / "mobilenet.weights.h5").exists() or (MODELS / "mobilenet.keras").exists():
        available.append("mobilenet")
    return available


def _sample_paths() -> list[Path]:
    # Use bundled manifest (1 curated sample per identity)
    manifest_path = ROOT / "app" / "frontend" / "src" / "data" / "samples.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            bundled_dir = ROOT / "app" / "frontend" / "public"
            paths = [bundled_dir / s["url"].lstrip("/") for s in manifest]
            existing = [p for p in paths if p.exists()]
            if existing:
                return existing
        except Exception:
            pass

    # Fallback: metadata.csv (1 per identity)
    meta_path = DATASET / "metadata.csv"
    if meta_path.exists():
        import pandas as pd
        meta = pd.read_csv(meta_path)
        test = meta[(meta["split"] == "test") & (meta["qc"] == "ok")].drop_duplicates(subset=["identity"])
        paths = [DATASET / f for f in test["file"]]
        existing = [p for p in paths if p.exists()]
        if existing:
            return existing

    return []


# ──────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "models_dir": str(MODELS)}


@app.get("/api/models")
def get_models():
    """List models that are trained and available."""
    available = _available_models()
    default_tau = 0.25
    threshold_path = MODELS / "threshold.json"
    if threshold_path.exists():
        default_tau = json.load(open(threshold_path)).get("tau", 0.25)
    return {"models": available, "default_tau": round(default_tau, 3)}


@app.get("/api/samples")
def get_samples():
    """List available test-split sample filenames."""
    paths = _sample_paths()
    samples = [
        {
            "name": p.name,
            "identity": p.parent.name,
            "url": f"/api/samples/{p.parent.name}/{p.name}",
        }
        for p in paths
    ]
    return {"samples": samples, "count": len(samples)}


@app.get("/api/samples/{identity}/{filename}")
def serve_sample(identity: str, filename: str):
    """Serve a sample signature image."""
    path = DATASET / "crops_sim" / identity / filename
    if not path.exists():
        path = DATASET / "crops" / identity / filename
    if not path.exists():
        # Fallback to bundled public samples
        path = ROOT / "app" / "frontend" / "public" / "samples" / identity / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(str(path), media_type="image/png")


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("svm"),
    tau: float = Form(None),
):
    """Run inference on an uploaded signature image.

    Returns:
        prediction  : str   — student ID or "Uncertain (Sxx)"
        candidate   : str   — closest identity
        confidence  : float — max softmax probability
        recognised  : bool
        top3        : list  — [{id, confidence}]
        preprocessed: str   — base64-encoded preprocessed image (PNG)
        model_used  : str
        ms          : int   — inference time
    """
    _ALLOWED = ("svm", "rf", "knn", "lr", "gb", "cnn", "mobilenet")
    if model not in _ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Choose from {_ALLOWED}")

    # Read image (support PNG transparency / RGBA)
    contents = await file.read()
    raw = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode image file")

    # If image has alpha channel (RGBA), blend onto pure white background
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3].astype(np.float32) / 255.0
        bgr = img[:, :, :3].astype(np.float32)
        white_bg = np.ones_like(bgr) * 255.0
        img = (bgr * alpha[..., np.newaxis] + white_bg * (1.0 - alpha[..., np.newaxis])).astype(np.uint8)
    elif img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if min(img.shape[:2]) < 16:
        raise HTTPException(status_code=422, detail="Image too small (< 16px)")

    try:
        predictor = _get_predictor(model)

        # Override threshold if provided
        original_tau = predictor.tau
        if tau is not None:
            predictor.tau = float(tau)

        t0 = time.perf_counter()
        result = predictor.identify(img)
        ms = round((time.perf_counter() - t0) * 1000)

        # Compute comparative results across all 5 models
        all_models = identify_all_models(img, tau=predictor.tau)

        predictor.tau = original_tau  # restore

        # Encode preprocessed image as base64 PNG
        import base64
        pre = (result["preprocessed"] * 255).clip(0, 255).astype(np.uint8)
        _, buf = cv2.imencode(".png", pre)
        pre_b64 = base64.b64encode(buf.tobytes()).decode()

        return {
            "prediction":   result["prediction"],
            "candidate":    result.get("candidate", result["prediction"]),
            "confidence":   round(result["confidence"], 4),
            "recognised":   result["recognised"],
            "top3": [
                {"id": sid, "confidence": round(conf, 4)}
                for sid, conf in result["top3"]
            ],
            "all_models":   all_models,
            "preprocessed_b64": pre_b64,
            "model_used": model,
            "ms": ms,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# ──────────────────────────────────────────────────────────
# Serve built React frontend if available
# ──────────────────────────────────────────────────────────
DIST_DIR = Path(__file__).resolve().parent / "frontend" / "dist"
if DIST_DIR.exists():
    if (DIST_DIR / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")
        target = DIST_DIR / full_path
        if target.is_file():
            return FileResponse(str(target))
        return FileResponse(str(DIST_DIR / "index.html"))
