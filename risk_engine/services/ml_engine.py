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

# To get Clinician-readable display values for encoded/binary features so that "Sex (0.0)" can be understood as female
FEATURE_DISPLAY_VALUE = {
    "sex_encoded":        {0: "Female",         1: "Male"},
    "polypharmacy_5plus": {0: "No (< 5 meds)",  1: "Yes (≥ 5 meds)"},
}

def _get_artefact_paths():
    here         = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(here, "..", ".."))
    return {
        "model":  os.path.join(project_root, "model.pkl"),
        "scaler": os.path.join(project_root, "scaler.pkl"),
        "fnames": os.path.join(project_root, "feature_names.json"),
    }

#  Load model.pkl, scaler.pkl, and feature_names.json from the project root, if any filr is missing
#  raises RuntimeError and indicate which ones and where they should be or if Feature order in feature_names.json does not match FEATURE_NAMES here
def _load_model():
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

# function to return a clinician-readable display string for a feature value.
def _display_value(feature: str, raw_value) -> str:
    if feature in FEATURE_DISPLAY_VALUE:
        return FEATURE_DISPLAY_VALUE[feature].get(int(raw_value), str(raw_value))
    return f"{raw_value:.1f}"


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
        raw_val      = raw_values[idx]
        contribution = float(contributions[idx])
        top_factors.append({
            "feature":      feature,
            "label":        FEATURE_LABELS.get(feature, feature),
            "raw_value":    raw_val,
            "display_value": _display_value(feature, raw_val),
            "contribution": contribution,
            "direction":    "increases" if contribution > 0 else "decreases",
        })

    medication_count = consultation.medication_count if consultation.medication_count is not None else 0

    explanation  = _build_explanation(
        top_factors, raw_values, feature_names,
        probability, risk_flag, threshold,
        comorbidity, polypharmacy, medication_count,
    )
    input_summary = _build_input_summary(
        raw_values, feature_names, comorbidity, polypharmacy, medication_count
    )

    return {
        "probability":   probability,
        "risk_flag":     risk_flag,
        "top_factors":   top_factors,
        "explanation":   explanation,     # natural language paragraph for the modal
        "input_summary": input_summary,   # one-line list of what the model received
    }

# One-line plain English summary of the values passed to the model.
def _build_input_summary(raw_values, feature_names, comorbidity, polypharmacy, medication_count):
    vals = dict(zip(feature_names, raw_values))

    poly_note = " (polypharmacy)" if polypharmacy else ""
    med_str   = f"{int(medication_count)} medication{'s' if medication_count != 1 else ''}{poly_note}"

    parts = [
        f"BMI {vals['bmi']:.1f}",
        f"Albumin {vals['albumin_gdl']:.1f} g/dL",
        f"Haemoglobin {vals['haemoglobin_gdl']:.1f} g/dL",
        f"{int(comorbidity)} condition{'s' if comorbidity != 1 else ''}",
        med_str,
        f"Age {int(vals['age'])}",
        "Male" if vals["sex_encoded"] == 1 else "Female",
    ]
    return " · ".join(parts)

# to reduce cognitive assembly of the model output, we provide a natural language explanation of the ML's result
def _build_explanation(top_factors, raw_values, feature_names, probability, risk_flag, threshold, comorbidity, polypharmacy, medication_count):

    vals     = dict(zip(feature_names, raw_values))
    prob_pct = round(probability * 100, 1)

    # Input context sentence
    cond_str = f"{int(comorbidity)} condition{'s' if comorbidity != 1 else ''}"
    poly_note = " (polypharmacy)" if polypharmacy else ""
    med_str   = f"{int(medication_count)} medication{'s' if medication_count != 1 else ''}{poly_note}"

    input_clause = (
        f"Based on this patient's BMI of {vals['bmi']:.1f}, "
        f"albumin of {vals['albumin_gdl']:.1f} g/dL, "
        f"{cond_str} and {med_str}"
    )

    # Probability sentence
    prob_sentence = (
        f"the model estimated a {prob_pct}% probability of nutritional risk "
        f"(threshold: {int(threshold * 100)}%)."
    )

    # Signal sentence only for top 2 factors
    SIGNAL_PHRASES = {
        "bmi":               lambda v: f"{'a low BMI' if v < 20 else 'BMI'}",
        "albumin_gdl":       lambda v: f"{'a low serum albumin' if v < 3.8 else 'serum albumin'}",
        "haemoglobin_gdl":   lambda v: f"{'a low haemoglobin' if v < 12 else 'haemoglobin'}",
        "comorbidity_count": lambda v: f"{'multiple long-term conditions' if v >= 3 else 'long-term conditions'}",
        "polypharmacy_5plus":lambda v: "polypharmacy (≥5 medications)" if v >= 1 else "medication count",
        "age":               lambda v: f"age ({int(v)})",
        "sex_encoded":       lambda v: "sex",
    }

    # Take top 2 factors by absolute contribution, skip if contribution near zero
    meaningful = [f for f in top_factors[:2] if abs(f["contribution"]) > 0.05]

    if not meaningful:
        signal_sentence = ""
    elif risk_flag:
        phrases = [SIGNAL_PHRASES.get(f["feature"], lambda v: f["label"])(f["raw_value"])
                   for f in meaningful]
        if len(phrases) == 1:
            signal_sentence = f"The pattern was most strongly associated with {phrases[0]}."
        else:
            signal_sentence = (
                f"The pattern was most strongly associated with "
                f"{phrases[0]} and {phrases[1]}."
            )
    else:
        signal_sentence = (
            "No individual feature pattern was strong enough to meet the risk threshold."
        )

    return f"{input_clause}, {prob_sentence} {signal_sentence}".strip()