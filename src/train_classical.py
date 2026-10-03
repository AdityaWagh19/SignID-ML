"""Train and compare classical ML models on HOG features.

All models share the same HOG feature pipeline (8100-dim).

Run:
    python -m src.train_classical

Writes:  models/classical_results.json
         models/svm.joblib  (best SVM)
         models/best_classical.joblib  (best non-SVM)
"""

import json
import time

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import MODELS, SEED
from src.dataset import load_splits
from src.features import hog_features


def eval_model(model, F_va, y_va, F_te, y_te, n_classes):
    labels = list(range(n_classes))
    p_va = model.predict_proba(F_va)
    p_te = model.predict_proba(F_te)
    return {
        "val_acc":   round(float(accuracy_score(y_va, p_va.argmax(1))), 4),
        "val_top3":  round(float(top_k_accuracy_score(y_va, p_va, k=3, labels=labels)), 4),
        "val_f1":    round(float(f1_score(y_va, p_va.argmax(1), average="macro", zero_division=0)), 4),
        "test_acc":  round(float(accuracy_score(y_te, p_te.argmax(1))), 4),
        "test_top3": round(float(top_k_accuracy_score(y_te, p_te, k=3, labels=labels)), 4),
        "test_f1":   round(float(f1_score(y_te, p_te.argmax(1), average="macro", zero_division=0)), 4),
    }


def main():
    print("Loading data ...")
    splits = load_splits()
    n_classes = len(splits["class_map"])
    X_tr, y_tr = splits["X_train"], splits["y_train"]
    X_va, y_va = splits["X_val"],   splits["y_val"]
    X_te, y_te = splits["X_test"],  splits["y_test"]

    print("Extracting HOG features ...")
    F_tr = hog_features(X_tr)
    F_va = hog_features(X_va)
    F_te = hog_features(X_te)
    print(f"  Feature dim: {F_tr.shape[1]}")

    MODELS.mkdir(exist_ok=True)
    results = {}
    trained = {}

    # 1. SVM
    print("\n[1/5] SVM (Linear + RBF) via GridSearchCV ...")
    t0 = time.time()
    pipe = Pipeline([("svc", SVC(probability=True, random_state=SEED, class_weight="balanced"))])
    gs = GridSearchCV(pipe, {"svc__kernel": ["linear", "rbf"], "svc__C": [0.1, 1, 10], "svc__gamma": ["scale"]},
                      cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
    gs.fit(F_tr, y_tr)
    best = gs.best_estimator_
    m = eval_model(best, F_va, y_va, F_te, y_te, n_classes)
    m["best_params"] = str(gs.best_params_)
    m["train_time_s"] = round(time.time() - t0, 1)
    results["hog_svm"] = m
    trained["hog_svm"] = best
    joblib.dump(best, MODELS / "svm.joblib")
    print(f"  Val {m['val_acc']:.4f} | Top3 {m['val_top3']:.4f} | F1 {m['val_f1']:.4f} | {gs.best_params_}")

    # 2. KNN
    print("\n[2/5] KNN ...")
    t0 = time.time()
    pipe = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier())])
    gs = GridSearchCV(pipe, {"knn__n_neighbors": [3, 5, 7, 9, 11]}, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(F_tr, y_tr)
    best = gs.best_estimator_
    m = eval_model(best, F_va, y_va, F_te, y_te, n_classes)
    m["best_params"] = str(gs.best_params_)
    m["train_time_s"] = round(time.time() - t0, 1)
    results["hog_knn"] = m
    trained["hog_knn"] = best
    print(f"  Val {m['val_acc']:.4f} | Top3 {m['val_top3']:.4f} | F1 {m['val_f1']:.4f} | {gs.best_params_}")

    # 3. Logistic Regression
    print("\n[3/5] Logistic Regression ...")
    t0 = time.time()
    pipe = Pipeline([("sc", StandardScaler()), ("lr", LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced"))])
    gs = GridSearchCV(pipe, {"lr__C": [0.01, 0.1, 1, 10]}, cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(F_tr, y_tr)
    best = gs.best_estimator_
    m = eval_model(best, F_va, y_va, F_te, y_te, n_classes)
    m["best_params"] = str(gs.best_params_)
    m["train_time_s"] = round(time.time() - t0, 1)
    results["hog_lr"] = m
    trained["hog_lr"] = best
    print(f"  Val {m['val_acc']:.4f} | Top3 {m['val_top3']:.4f} | F1 {m['val_f1']:.4f} | {gs.best_params_}")

    # 4. Random Forest
    print("\n[4/5] Random Forest ...")
    t0 = time.time()
    pipe = Pipeline([("rf", RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced", n_jobs=-1))])
    gs = GridSearchCV(pipe, {"rf__max_depth": [None, 20, 40], "rf__min_samples_leaf": [1, 2]},
                      cv=5, scoring="accuracy", n_jobs=-1)
    gs.fit(F_tr, y_tr)
    best = gs.best_estimator_
    m = eval_model(best, F_va, y_va, F_te, y_te, n_classes)
    m["best_params"] = str(gs.best_params_)
    m["train_time_s"] = round(time.time() - t0, 1)
    results["hog_rf"] = m
    trained["hog_rf"] = best
    print(f"  Val {m['val_acc']:.4f} | Top3 {m['val_top3']:.4f} | F1 {m['val_f1']:.4f} | {gs.best_params_}")

    # 5. Gradient Boosting
    print("\n[5/5] Gradient Boosting ...")
    t0 = time.time()
    pipe = Pipeline([("sc", StandardScaler()),
                     ("gb", GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=SEED))])
    pipe.fit(F_tr, y_tr)
    m = eval_model(pipe, F_va, y_va, F_te, y_te, n_classes)
    m["train_time_s"] = round(time.time() - t0, 1)
    results["hog_gb"] = m
    trained["hog_gb"] = pipe
    print(f"  Val {m['val_acc']:.4f} | Top3 {m['val_top3']:.4f} | F1 {m['val_f1']:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print(f"{'Model':<18} {'ValAcc':>7} {'ValTop3':>8} {'ValF1':>7} {'TestAcc':>8} {'TestF1':>7}")
    print("-" * 70)
    best_name, best_val = None, -1
    for name, m in results.items():
        marker = " *" if m["val_acc"] > best_val else ""
        if m["val_acc"] > best_val:
            best_val = m["val_acc"]
            best_name = name
        print(f"{name:<18} {m['val_acc']:>7.4f} {m['val_top3']:>8.4f} {m['val_f1']:>7.4f} {m['test_acc']:>8.4f} {m['test_f1']:>7.4f}{marker}")
    print("=" * 70)
    print(f"Best: {best_name}  (val_acc={best_val:.4f})")

    out = MODELS / "classical_results.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"Results saved -> {out}")

    non_svm_best = max((k for k in results if k != "hog_svm"), key=lambda k: results[k]["val_acc"])
    joblib.dump(trained[non_svm_best], MODELS / "best_classical.joblib")
    print(f"Best non-SVM ({non_svm_best}) saved -> models/best_classical.joblib")


if __name__ == "__main__":
    main()
