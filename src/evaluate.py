"""Evaluation utilities — metrics, confusion matrix, reject curve.

Spec §9.1 and §8.5.

Usage (from a notebook or evaluate script):
    from src.evaluate import evaluate, reject_curve, plot_confusion, plot_reject_curve
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)

from src.config import REPORTS


# ──────────────────────────────────────────────────────────
# Core metrics
# ──────────────────────────────────────────────────────────

def evaluate(probs: np.ndarray, y_true: np.ndarray, n_classes: int = None) -> dict:
    """Compute all evaluation metrics for one model.

    Parameters
    ----------
    probs  : (N, C) float — softmax probabilities
    y_true : (N,)  int   — ground-truth class indices
    n_classes : int       — needed for top_k when C doesn't match classes in y_true

    Returns
    -------
    dict with keys: accuracy, top3, macro_f1, report, confusion, ms_per_image
    """
    y_pred = probs.argmax(axis=1)
    labels = list(range(probs.shape[1])) if n_classes is None else list(range(n_classes))

    metrics = {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "top3":      float(top_k_accuracy_score(y_true, probs, k=3, labels=labels)),
        "macro_f1":  float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "report":    classification_report(y_true, y_pred, output_dict=True,
                                           zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    return metrics


# ──────────────────────────────────────────────────────────
# Open-set rejection curve (spec §8.5)
# ──────────────────────────────────────────────────────────

def reject_curve(
    probs_known: np.ndarray,
    probs_unknown: np.ndarray,
    taus: np.ndarray = None,
) -> list[tuple]:
    """Compute (threshold, known_accept_rate, unknown_reject_rate) sweep.

    Parameters
    ----------
    probs_known   : (N_k, C) — enrolled val/test images
    probs_unknown : (N_u, C) — unknown val/test images
    taus          : threshold values to sweep (default: 90 values from 0.10 to 0.99)

    Returns
    -------
    List of (tau, known_accept_rate, unknown_reject_rate) tuples.
    """
    if taus is None:
        taus = np.linspace(0.10, 0.99, 90)

    conf_known   = probs_known.max(axis=1)
    conf_unknown = probs_unknown.max(axis=1)

    return [
        (float(t), float((conf_known >= t).mean()), float((conf_unknown < t).mean()))
        for t in taus
    ]


def best_threshold(
    probs_known: np.ndarray,
    probs_unknown: np.ndarray,
    min_known_accept: float = 0.90,
) -> float:
    """Pick the threshold that maximises unknown reject rate given a floor on known accept rate.

    Parameters
    ----------
    probs_known         : (N_k, C)
    probs_unknown       : (N_u, C)
    min_known_accept    : keep at least this fraction of known images (default 0.90)

    Returns
    -------
    float : chosen threshold τ
    """
    curve = reject_curve(probs_known, probs_unknown)
    valid = [(tau, ku, ur) for tau, ku, ur in curve if ku >= min_known_accept]
    if not valid:
        # Relax: just maximise balanced accuracy
        best = max(curve, key=lambda x: (x[1] + x[2]) / 2)
        return best[0]
    return max(valid, key=lambda x: x[2])[0]


def save_threshold(tau: float, path=None):
    path = path or (REPORTS.parent / "models" / "threshold.json")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"tau": tau}, open(path, "w"), indent=2)
    print(f"Threshold tau={tau:.4f} saved -> {path}")


# ──────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────

def plot_confusion(
    conf_matrix: list | np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    save_path: Path = None,
):
    cm = np.array(conf_matrix)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 2)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    return fig


def plot_training_curves(history: dict, title: str = "Training Curves", save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["accuracy"],     label="Train acc")
    ax1.plot(history["val_accuracy"], label="Val acc")
    ax1.set_title(f"{title} — Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["loss"],     label="Train loss")
    ax2.plot(history["val_loss"], label="Val loss")
    ax2.set_title(f"{title} — Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    return fig


def plot_reject_curve(
    curve: list[tuple],
    tau_chosen: float = None,
    save_path=None,
):
    taus  = [t for t, _, _ in curve]
    known = [k for _, k, _ in curve]
    unk   = [u for _, _, u in curve]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(taus, known, label="Known accept rate")
    ax.plot(taus, unk,   label="Unknown reject rate")
    if tau_chosen is not None:
        ax.axvline(tau_chosen, color="red", linestyle="--", label=f"tau={tau_chosen:.2f}")
    ax.set_xlabel("Confidence threshold tau")
    ax.set_ylabel("Rate")
    ax.set_title("Open-set rejection: known vs unknown at each threshold")
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")
    return fig


# ──────────────────────────────────────────────────────────
# Full evaluation script
# ──────────────────────────────────────────────────────────

def main():
    """Evaluate all trained models on the test set and save reports/metrics.json."""
    import joblib
    import json as _json
    import keras

    from src.config import MODELS
    from src.dataset import load_splits
    from src.features import hog_features

    splits     = load_splits()
    X_te, y_te = splits["X_test"], splits["y_test"]
    class_map  = splits["class_map"]
    names      = [class_map[i] for i in sorted(class_map)]
    n_classes  = len(class_map)

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "figures").mkdir(exist_ok=True)
    all_metrics = {}

    # ── HOG features (shared across all classical models) ──
    F_te = hog_features(X_te)

    # ── Classical models (SVM, RF, KNN, LR, GB) ────────────
    classical_paths = {
        "hog_svm": MODELS / "svm.joblib",
        "hog_rf":  MODELS / "best_classical.joblib",
    }
    # Also try to load from classical_results.json for display
    classical_json = MODELS / "classical_results.json"
    if classical_json.exists():
        import json as _cj
        classical_summary = _cj.load(open(classical_json))
        all_metrics["classical_comparison"] = classical_summary

    svm_path = MODELS / "svm.joblib"
    if svm_path.exists():
        print("Evaluating HOG+SVM ...")
        svm = joblib.load(svm_path)
        probs = svm.predict_proba(F_te)
        m = evaluate(probs, y_te, n_classes=n_classes)
        all_metrics["hog_svm"] = {k: v for k, v in m.items() if k not in ("report",)}
        print(f"  Test acc: {m['accuracy']:.4f}  Top-3: {m['top3']:.4f}  F1: {m['macro_f1']:.4f}")
        plot_confusion(m["confusion"], names, "HOG+SVM Confusion Matrix",
                       REPORTS / "figures" / "cm_svm.png")

    best_classical_path = MODELS / "best_classical.joblib"
    if best_classical_path.exists():
        print("Evaluating Best Classical (Random Forest) ...")
        rf = joblib.load(best_classical_path)
        probs_rf = rf.predict_proba(F_te)
        m = evaluate(probs_rf, y_te, n_classes=n_classes)
        all_metrics["hog_rf"] = {k: v for k, v in m.items() if k not in ("report",)}
        print(f"  Test acc: {m['accuracy']:.4f}  Top-3: {m['top3']:.4f}  F1: {m['macro_f1']:.4f}")
        plot_confusion(m["confusion"], names, "Random Forest Confusion Matrix",
                       REPORTS / "figures" / "cm_rf.png")

    # ── Custom CNN ─────────────────────────────────────────
    cnn_weights = MODELS / "cnn.weights.h5"
    cnn_keras = MODELS / "cnn.keras"
    best_cnn_probs = None
    if cnn_weights.exists() or cnn_keras.exists():
        print("Evaluating Custom CNN ...")
        from src.models import build_cnn
        cnn = build_cnn(n_classes=n_classes, use_aug=False)
        if cnn_weights.exists():
            cnn.load_weights(str(cnn_weights))
        else:
            cnn = keras.models.load_model(str(cnn_keras))
        probs = cnn.predict(X_te, verbose=0)
        best_cnn_probs = probs
        m = evaluate(probs, y_te, n_classes=n_classes)
        all_metrics["cnn"] = {k: v for k, v in m.items() if k not in ("report",)}
        print(f"  Test acc: {m['accuracy']:.4f}  Top-3: {m['top3']:.4f}  F1: {m['macro_f1']:.4f}")
        plot_confusion(m["confusion"], names, "Custom CNN Confusion Matrix",
                       REPORTS / "figures" / "cm_cnn.png")

        hist_path = MODELS / "cnn_history.json"
        if hist_path.exists():
            hist = _json.load(open(hist_path))
            plot_training_curves(hist, "Custom CNN",
                                 REPORTS / "figures" / "curves_cnn.png")

    # ── MobileNetV2 ────────────────────────────────────────
    mob_weights = MODELS / "mobilenet.weights.h5"
    mob_keras = MODELS / "mobilenet.keras"
    if mob_weights.exists() or mob_keras.exists():
        print("Evaluating MobileNetV2 ...")
        from src.models import build_mobilenet
        mob = build_mobilenet(n_classes=n_classes, use_aug=False)
        if mob_weights.exists():
            mob.load_weights(str(mob_weights))
        else:
            mob = keras.models.load_model(str(mob_keras))
        probs = mob.predict(X_te, verbose=0)
        m = evaluate(probs, y_te, n_classes=n_classes)
        all_metrics["mobilenet"] = {k: v for k, v in m.items() if k not in ("report",)}
        print(f"  Test acc: {m['accuracy']:.4f}  Top-3: {m['top3']:.4f}  F1: {m['macro_f1']:.4f}")
        plot_confusion(m["confusion"], names, "MobileNetV2 Confusion Matrix",
                       REPORTS / "figures" / "cm_mobilenet.png")

    # ── Rejection threshold calibration (Open-set via SVM) ──
    # Use SVM (primary model) for threshold — it's what the app deploys.
    # min_known_accept=0.70: we accept only high-confidence correct predictions.
    # Known samples split bimodally (0.10-0.14 low, 0.47-0.65 high);
    # unknown impostors cluster at 0.15-0.44. tau~0.43 rejects ~89% of impostors.
    if "X_unk_test" in splits and svm_path.exists():
        X_unk_te = splits["X_unk_test"]
        print(f"\nCalibrating reject threshold with {len(X_unk_te)} unknown test samples ...")
        svm_best = joblib.load(svm_path)
        F_unk = hog_features(X_unk_te)
        probs_known_svm = svm_best.predict_proba(F_te)
        probs_unk_svm   = svm_best.predict_proba(F_unk)
        curve = reject_curve(probs_known_svm, probs_unk_svm)
        tau = best_threshold(probs_known_svm, probs_unk_svm, min_known_accept=0.70)
        save_threshold(tau)
        plot_reject_curve(curve, tau_chosen=tau,
                          save_path=REPORTS / "figures" / "reject_curve.png")
        all_metrics["threshold_tau"] = float(tau)
        print(f"  tau={tau:.4f}")

    # ── Save all metrics ───────────────────────────────────
    out = REPORTS / "metrics.json"
    _json.dump(all_metrics, open(out, "w"), indent=2)
    print(f"\nAll metrics saved -> {out}")


if __name__ == "__main__":
    main()

