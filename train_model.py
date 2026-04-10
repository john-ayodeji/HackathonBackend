"""
PDI Model Trainer — Neural Network for ICU Deterioration Risk Prediction

Trains two scikit-learn MLP models on synthetically generated clinical vital-signs
data that is grounded in standard adult ICU reference ranges:

  1. MLPRegressor  → predicts a continuous risk_weight (0.0–1.0)
  2. MLPClassifier → predicts a severity class  (0=low, 1=moderate, 2=high)

Feature vector (18 values per sample):
  [hr, rr, spo2, temp, sbp, dbp,                 ← current readings
   hr_slope, rr_slope, spo2_slope, …,             ← per-hour trend
   hr_proj, rr_proj, spo2_proj, …]                ← 4-hour projection

Saved artefacts (in ./models/):
  scaler.joblib            — StandardScaler fit on training data
  risk_regressor.joblib    — MLPRegressor
  severity_classifier.joblib — MLPClassifier

Usage:
    python train_model.py                   # train with default settings
    python train_model.py --samples 20000   # more samples for better accuracy
"""

import argparse
import os

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

# ── Adult ICU normal reference ranges ────────────────────────────────────────
NORMAL_RANGES = {
    "hr":   {"low": 60,   "high": 100,  "min": 20,   "max": 200},
    "rr":   {"low": 12,   "high": 18,   "min": 4,    "max": 60},
    "spo2": {"low": 95,   "high": 100,  "min": 60,   "max": 100},
    "temp": {"low": 36.5, "high": 37.3, "min": 33.0, "max": 42.0},
    "sbp":  {"low": 90,   "high": 120,  "min": 50,   "max": 220},
    "dbp":  {"low": 60,   "high": 80,   "min": 30,   "max": 160},
}

# Clinical sensitivity spans — how many units above/below normal is "maximally dangerous".
# Using a tighter span makes clinically significant deviations score higher.
CLINICAL_SPANS = {
    "hr":   {"high": 50,  "low": 25},   # 100→150 is severe tachycardia; 60→35 is severe bradycardia
    "rr":   {"high": 14,  "low": 8},    # 18→32 is severe tachypnoea; 12→4 is severe bradypnoea
    "spo2": {"high": 5,   "low": 15},   # can't exceed 100; 95→80 is severe hypoxaemia
    "temp": {"high": 2.0, "low": 2.0},  # 37.3→39.3 is fever; 36.5→34.5 is hypothermia
    "sbp":  {"high": 60,  "low": 40},   # 120→180 severe HTN; 90→50 severe hypotension
    "dbp":  {"high": 30,  "low": 25},   # 80→110; 60→35
}

# PDI composite weights (fractional; must sum to 1.0 — mirrors the 100-point scale in engine.py)
PDI_WEIGHTS = {"spo2": 0.25, "hr": 0.20, "sbp": 0.18, "dbp": 0.12, "temp": 0.15, "rr": 0.10}

VITAL_KEYS = ["hr", "rr", "spo2", "temp", "sbp", "dbp"]


# ── Ground-truth label computation ────────────────────────────────────────────

def compute_risk_weight(values: list, slopes: list, projections: list) -> float:
    """
    Compute a continuous risk weight from vital signs using clinically-tuned
    sensitivity spans.  Deviations are measured relative to how far a value
    is from the normal boundary toward a clinically critical level, not the
    full physiological extremes.  This ensures that a SpO2 of 92% or an HR
    of 115 bpm registers as meaningfully risky.
    """
    total = 0.0
    for i, key in enumerate(VITAL_KEYS):
        r    = NORMAL_RANGES[key]
        cs   = CLINICAL_SPANS[key]
        projected = projections[i]
        slope     = slopes[i]
        weight    = PDI_WEIGHTS[key]

        if projected > r["high"]:
            deviation = min(1.0, (projected - r["high"]) / max(cs["high"], 1))
            if slope > 0:
                deviation = min(1.0, deviation * 1.25)
        elif projected < r["low"]:
            deviation = min(1.0, (r["low"] - projected) / max(cs["low"], 1))
            if slope < 0:
                deviation = min(1.0, deviation * 1.25)
        else:
            deviation = 0.0

        total += deviation * weight

    return min(1.0, total)


# ── Synthetic dataset generation ──────────────────────────────────────────────

def generate_training_data(n_samples: int = 15000, random_state: int = 42):
    """
    Generate synthetic vital-signs samples covering three clinical scenarios:

    - normal    (40 %): all vitals within reference range, gentle trends
    - borderline (30 %): one or more vitals at the edge of normal range
    - abnormal  (30 %): one or more vitals clearly outside normal range

    Returns
    -------
    X : ndarray (n_samples, 18)
    y_risk : ndarray (n_samples,)   float 0.0–1.0
    y_severity : ndarray (n_samples,)  int 0/1/2
    """
    rng = np.random.default_rng(random_state)
    X, y_risk, y_severity = [], [], []

    for _ in range(n_samples):
        scenario = rng.choice(
            ["normal", "borderline", "abnormal"],
            p=[0.40, 0.30, 0.30],
        )

        values, slopes, projections = [], [], []

        for key in VITAL_KEYS:
            r = NORMAL_RANGES[key]
            span = r["high"] - r["low"]

            if scenario == "normal":
                # Sample well inside the normal band
                current = rng.uniform(r["low"] + span * 0.05, r["high"] - span * 0.05)
                slope = float(rng.normal(0, span * 0.01))
                projected = float(np.clip(current + slope * 4, r["min"], r["max"]))

            elif scenario == "borderline":
                # Edge of normal range or slightly outside on either side
                if rng.random() < 0.5:
                    current = rng.uniform(r["high"] - span * 0.15, r["high"] + span * 0.35)
                else:
                    current = rng.uniform(r["low"] - span * 0.35, r["low"] + span * 0.15)
                current = float(np.clip(current, r["min"], r["max"]))
                slope = float(rng.normal(0, span * 0.03))
                projected = float(np.clip(current + slope * 4, r["min"], r["max"]))

            else:  # abnormal
                # Significantly outside normal range
                if rng.random() < 0.5:
                    current = float(rng.uniform(r["high"], r["max"]))
                    slope_dir = float(rng.choice([-1, 1], p=[0.3, 0.7]))
                else:
                    current = float(rng.uniform(r["min"], r["low"]))
                    slope_dir = float(rng.choice([-1, 1], p=[0.7, 0.3]))
                current = float(np.clip(current, r["min"], r["max"]))
                slope = float(rng.uniform(0, span * 0.08)) * slope_dir
                projected = float(np.clip(current + slope * 4, r["min"], r["max"]))

            values.append(float(current))
            slopes.append(float(slope))
            projections.append(float(projected))

        risk = compute_risk_weight(values, slopes, projections)
        # Add small Gaussian noise to avoid perfectly deterministic labels
        risk = float(np.clip(risk + rng.normal(0, 0.015), 0.0, 1.0))

        severity = 0 if risk < 0.2 else (1 if risk < 0.5 else 2)

        X.append(values + slopes + projections)
        y_risk.append(risk)
        y_severity.append(severity)

    return np.array(X, dtype=np.float32), np.array(y_risk, dtype=np.float32), np.array(y_severity, dtype=np.int32)


# ── Training ──────────────────────────────────────────────────────────────────

def train_and_save(n_samples: int = 15000, output_dir: str = "models") -> dict:
    """
    Generate data, train both MLP models, evaluate, and persist to *output_dir*.

    Returns
    -------
    dict with keys 'mae' (float) and 'accuracy' (float)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[train] Generating {n_samples:,} synthetic training samples …")
    X, y_risk, y_severity = generate_training_data(n_samples=n_samples)
    print(f"[train] Dataset shape: {X.shape}  severity distribution: "
          f"{dict(zip(*np.unique(y_severity, return_counts=True)))}")

    X_train, X_test, yr_train, yr_test, ys_train, ys_test = train_test_split(
        X, y_risk, y_severity, test_size=0.20, random_state=42
    )

    # Fit a StandardScaler on training data only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Risk-weight regressor ─────────────────────────────────────────────────
    print("[train] Fitting MLPRegressor …")
    regressor = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=600,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=20,
        verbose=False,
    )
    regressor.fit(X_train_s, yr_train)
    mae = mean_absolute_error(yr_test, regressor.predict(X_test_s))
    print(f"[train] Regressor  MAE  = {mae:.4f}")

    # ── Severity classifier ───────────────────────────────────────────────────
    print("[train] Fitting MLPClassifier …")
    classifier = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        max_iter=600,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=20,
        verbose=False,
    )
    classifier.fit(X_train_s, ys_train)
    acc = accuracy_score(ys_test, classifier.predict(X_test_s))
    print(f"[train] Classifier Acc = {acc:.4f}")

    # ── Persist ───────────────────────────────────────────────────────────────
    joblib.dump(scaler,     os.path.join(output_dir, "scaler.joblib"))
    joblib.dump(regressor,  os.path.join(output_dir, "risk_regressor.joblib"))
    joblib.dump(classifier, os.path.join(output_dir, "severity_classifier.joblib"))

    print(f"[train] Models saved to '{output_dir}/'")
    return {"mae": float(mae), "accuracy": float(acc)}


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PDI neural network models")
    parser.add_argument("--samples",    type=int, default=15000,  help="Number of training samples")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save model files")
    args = parser.parse_args()

    metrics = train_and_save(n_samples=args.samples, output_dir=args.output_dir)
    print(f"\n[train] Final metrics — MAE: {metrics['mae']:.4f}  |  Accuracy: {metrics['accuracy']:.4f}")
