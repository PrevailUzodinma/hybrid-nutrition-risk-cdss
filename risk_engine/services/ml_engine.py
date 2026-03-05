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
    "polypharmacy_flag",
]

FEATURE_LABELS = {
    "age":                "Age",
    "sex_encoded":        "Sex",
    "bmi":                "BMI",
    "albumin_gdl":        "Serum albumin",
    "haemoglobin_gdl":    "Haemoglobin",
    "comorbidity_count":  "Number of long-term conditions",
    "polypharmacy_flag":  "Polypharmacy (≥5 medications)",
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
