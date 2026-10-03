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
from src.predict import Predictor


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
    available = []
    if (MODELS / "svm.joblib").exists():
        available.append("svm")
    if (MODELS / "best_classical.joblib").exists():
        available.append("rf")
    if (MODELS / "cnn.weights.h5").exists() or (MODELS / "cnn.keras").exists():
        available.append("cnn")
    if (MODELS / "mobilenet.weights.h5").exists() or (MODELS / "mobilenet.keras").exists():
        available.append("mobilenet")
    return available


def _sample_paths() -> list[Path]:
    meta_path = DATASET / "metadata.csv"
    if not meta_path.exists():
        return []
    import pandas as pd
    meta = pd.read_csv(meta_path)
    test = meta[(meta["split"] == "test") & (meta["qc"] == "ok")]
    paths = [DATASET / f for f in test["file"]]
    return [p for p in paths if p.exists()]


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
    default_tau = 0.50
    threshold_path = MODELS / "threshold.json"
    if threshold_path.exists():
        default_tau = json.load(open(threshold_path)).get("tau", 0.50)
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
        prediction  : str   — student ID or "Not recognised"
        confidence  : float — max softmax probability
        recognised  : bool
        top3        : list  — [{id, confidence}]
        preprocessed: str   — base64-encoded preprocessed image (PNG)
        model_used  : str
        ms          : int   — inference time
    """
    _ALLOWED = ("svm", "rf", "cnn", "mobilenet")
    if model not in _ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unknown model '{model}'. Choose from {_ALLOWED}")

    # Read image
    contents = await file.read()
    raw = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None or min(img.shape[:2]) < 32:
        raise HTTPException(status_code=422,
                            detail="Could not decode image or image too small (< 32px)")

    try:
        predictor = _get_predictor(model)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Override threshold if provided
    original_tau = predictor.tau
    if tau is not None:
        predictor.tau = float(tau)

    t0 = time.perf_counter()
    result = predictor.identify(img)
    ms = round((time.perf_counter() - t0) * 1000)

    predictor.tau = original_tau  # restore

    # Encode preprocessed image as base64 PNG
    import base64
    pre = (result["preprocessed"] * 255).clip(0, 255).astype(np.uint8)
    _, buf = cv2.imencode(".png", pre)
    pre_b64 = base64.b64encode(buf.tobytes()).decode()

    return {
        "prediction":   result["prediction"],
        "confidence":   round(result["confidence"], 4),
        "recognised":   result["recognised"],
        "top3": [
            {"id": sid, "confidence": round(conf, 4)}
            for sid, conf in result["top3"]
        ],
        "preprocessed_b64": pre_b64,
        "model_used": model,
        "ms": ms,
    }


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
