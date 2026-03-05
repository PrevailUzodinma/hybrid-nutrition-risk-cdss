"""
Loads and runs the trained Logistic Regression model.
"""
import json
import os

import joblib
import numpy as np

ML_THRESHOLD = 0.45

# Feature names must exactly match feature_names.json produced by train_ml_nhanes.py
FEATURE_NAMES = [
    "age",
    "sex_encoded",
    "bmi",
    "albumin_gdl",
    "haemoglobin_gdl",
    "comorbidity_count",
    "polypharmacy_5plus",
]

FEATURE_LABELS = {
    "age":                "Age",
    "sex_encoded":        "Sex",
    "bmi":                "BMI",
    "albumin_gdl":        "Serum albumin",
    "haemoglobin_gdl":    "Haemoglobin",
    "comorbidity_count":  "Number of long-term conditions",
    "polypharmacy_5plus":  "Polypharmacy (≥5 medications)",
}


def _get_artefact_paths():
    here         = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(here, "..", ".."))
    return {
        "model":  os.path.join(project_root, "model.pkl"),
        "scaler": os.path.join(project_root, "scaler.pkl"),
        "fnames": os.path.join(project_root, "feature_names.json"),
    }


def _load_model():
    """
    Load model.pkl, scaler.pkl, and feature_names.json from the project root.
    Raises RuntimeError if:
    Any file is missing (tells you which ones and where they should be)
    or Feature order in feature_names.json does not match FEATURE_NAMES here
    """
    paths   = _get_artefact_paths()
    missing = [name for name, path in paths.items() if not os.path.exists(path)]

    if missing:
        readable = {"model": "model.pkl", "scaler": "scaler.pkl", "fnames": "feature_names.json"}
        raise RuntimeError(
            f"NHANES-trained model not found. Run train_ml_nhanes.py first.\n"
            f"Missing files: {', '.join(readable[k] for k in missing)}\n"
            f"Expected at: {os.path.normpath(paths['model'])}"
        )

    model  = joblib.load(paths["model"])
    scaler = joblib.load(paths["scaler"])

    with open(paths["fnames"], "r") as f:
        fnames_from_file = json.load(f)

    if fnames_from_file != FEATURE_NAMES:
        raise RuntimeError(
            f"Feature order mismatch — silent wrong predictions prevented.\n"
            f"  FEATURE_NAMES in ml_engine.py:  {FEATURE_NAMES}\n"
            f"  feature_names.json on disk:     {fnames_from_file}\n"
            f"Update FEATURE_NAMES in ml_engine.py to match feature_names.json exactly."
        )

    return model, scaler, fnames_from_file


# Module-level cache — loaded once per Django process, reused on every request
_model, _scaler, _fnames = None, None, None


def _get_model_and_scaler():
    global _model, _scaler, _fnames
    if _model is None:
        _model, _scaler, _fnames = _load_model()
    return _model, _scaler, _fnames


def score_patient(consultation, threshold: float = ML_THRESHOLD) -> dict:
    model, scaler, feature_names = _get_model_and_scaler()

    sex_encoded  = 1 if consultation.patient.sex == "M" else 0
    bmi          = consultation.bmi             if consultation.bmi             is not None else 25.0
    albumin      = consultation.albumin_gdl     if consultation.albumin_gdl     is not None else 4.0
    haemoglobin  = consultation.haemoglobin_gdl if consultation.haemoglobin_gdl is not None else 13.5
    comorbidity  = consultation.comorbidity_count if consultation.comorbidity_count is not None else 0
    polypharmacy = 1 if consultation.polypharmacy_flag else 0

    raw_values = [
        consultation.patient.age,
        sex_encoded,
        bmi,
        albumin,
        haemoglobin,
        comorbidity,
        polypharmacy,
    ]

    feature_vector        = np.array([raw_values], dtype=float)
    feature_vector_scaled = scaler.transform(feature_vector)

    probability = float(model.predict_proba(feature_vector_scaled)[0][1])
    risk_flag   = probability >= threshold

    # Contribution = coefficient × scaled_value (log-odds attribution for XAI)
    coefs         = model.coef_[0]
    scaled_vals   = feature_vector_scaled[0]
    contributions = coefs * scaled_vals

    sorted_indices = np.argsort(np.abs(contributions))[::-1]

    top_factors = []
    for idx in sorted_indices[:4]:
        feature      = feature_names[idx]
        contribution = float(contributions[idx])
        top_factors.append({
            "feature":      feature,
            "label":        FEATURE_LABELS.get(feature, feature),
            "raw_value":    raw_values[idx],
            "contribution": contribution,
            "direction":    "increases" if contribution > 0 else "decreases",
        })

    factor_lines = [
        f"{f['label']} ({f['raw_value']:.1f}) {f['direction']} risk"
        for f in top_factors[:3]
    ]

    explanation = (
        f"ML probability = {probability:.2f} ({'HIGH' if risk_flag else 'LOW'} risk). "
        f"Top contributing factors: {'; '.join(factor_lines)}."
    )

    return {
        "probability": probability,
        "risk_flag":   risk_flag,
        "top_factors": top_factors,
        "explanation": explanation,
    }
