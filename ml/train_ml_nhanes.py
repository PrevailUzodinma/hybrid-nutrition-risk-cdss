"""
My script to train a Logistic Regression model for nutritional risk prediction
using NHANES 2017-2018 cross-sectional data (filtered to adults aged 65+).
"""
import argparse
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Feature configuration 
# These are the cross-sectional NHANES variable names (and renamed equivalents).

NHANES_COLUMN_MAP = {
    "age":               ["RIDAGEYR", "age"],
    "sex_encoded":       ["RIAGENDR", "sex_encoded", "sex"],
    "bmi":               ["BMXBMI", "BMI", "bmi"],
    "albumin_gdl":       ["LBXSAL", "albumin_gdl", "albumin"],
    "haemoglobin_gdl":   ["LBXHGB", "haemoglobin_gdl", "haemoglobin"],
    "comorbidity_count": ["comorbidity_count"],
    "polypharmacy_5plus":["polypharmacy_5plus", "polypharmacy_flag"],
}

FEATURE_NAMES = [
    "age",
    "sex_encoded",
    "bmi",
    "albumin_gdl",
    "haemoglobin_gdl",
    "comorbidity_count",
    "polypharmacy_5plus",
]

LABEL_COL = "nutritional_risk_label"

# Step1: Load CSV 

def load_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"\nFile not found: {csv_path}")
        print("    Please provide the path to your merged NHANES master CSV.")
        sys.exit(1)
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"\n{'-'*60}")
    print(f"  NHANES MASTER CSV LOADED")
    print(f"{'-'*60}")
    print(f"  Total rows:    {len(df):,}")
    print(f"  Total columns: {len(df.columns)}")
    return df

# Step 2: Resolve column names to names used in this script, based on NHANES_COLUMN_MAP.

def resolve_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    missing = []

    print(f"\n{'='*60}")
    print("  COLUMN RESOLUTION")
    print(f"{'='*60}")

    for canonical, candidates in NHANES_COLUMN_MAP.items():
        found = None
        for candidate in candidates:
            if candidate in df.columns:
                found = candidate
                break
        if found:
            rename_map[found] = canonical
            status = f"'{found}' - '{canonical}'"
        else:
            missing.append(canonical)
            status = f"'{canonical}' — NOT FOUND (candidates: {candidates})"
        print(f"  {status}")

    df = df.rename(columns=rename_map)

    if missing:
        print(f"\n Missing columns: {missing}")
        print("     These will be excluded from features if unavailable.")
        print("     Check  CSV column names against NHANES_COLUMN_MAP at the top of the script.")

    return df

# Step 4: Encode sex as 0 and 1

def encode_sex(df: pd.DataFrame) -> pd.DataFrame:
    """
    to make sex_encoded is 0=Female, 1=Male, because NHANES uses RIAGENDR: 1=Male, 2=Female, so recode to 0/1.
    """
    if "sex_encoded" not in df.columns:
        return df

    # If NHANES encoding (1=Male, 2=Female), recode
    if df["sex_encoded"].isin([1, 2]).all():
        df["sex_encoded"] = df["sex_encoded"].map({1: 1, 2: 0})
        print("\n sex_encoded recoded: NHANES 1→1 (Male), 2→0 (Female)")
    # If already 0/1, leave as is
    return df


#Step 5: Create GLIM inspired binary label

def create_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create nutritional_risk_label using EXACT GLIM-inspired operational definition:
        Label = 1 if:
            (BMI < 18.5 OR albumin_gdl < 3.5)
            AND
            (comorbidity_count >= 1 OR polypharmacy_5plus == 1)

        Label = 0 otherwise.
    """
    required = {"bmi", "albumin_gdl", "comorbidity_count", "polypharmacy_5plus"}
    available = required.intersection(df.columns)
    missing = required - available

    print(f"\n{'-'*60}")
    print("  LABEL CREATION (GLIM-INSPIRED OPERATIONAL DEFINITION)")
    print(f"{'-'*60}")

    if missing:
        print(f"Missing columns for label creation: {missing}")
        print(" Attempting partial label with available columns.")

    # Nutritional criterion (phenotypic)
    nutritional_criterion = pd.Series(False, index=df.index)
    if "bmi" in df.columns:
        nutritional_criterion |= df["bmi"] < 18.5
    if "albumin_gdl" in df.columns:
        nutritional_criterion |= df["albumin_gdl"] < 3.5

    # Disease burden criterion (aetiology)
    disease_criterion = pd.Series(False, index=df.index)
    if "comorbidity_count" in df.columns:
        disease_criterion |= df["comorbidity_count"] >= 1
    if "polypharmacy_5plus" in df.columns:
        disease_criterion |= df["polypharmacy_5plus"] == 1

    df[LABEL_COL] = (nutritional_criterion & disease_criterion).astype(int)

    # Report class balance
    n_at_risk = df[LABEL_COL].sum()
    n_total = len(df)
    pct_risk = (n_at_risk / n_total) * 100

    print(f"  Label = 1 (at risk):     {n_at_risk:,} ({pct_risk:.1f}%)")
    print(f"  Label = 0 (not at risk): {n_total - n_at_risk:,} ({100 - pct_risk:.1f}%)")
    print(f"  Class ratio (1:0):       1 : {(n_total - n_at_risk) / max(n_at_risk, 1):.1f}")

    if pct_risk < 5:
        print("Class imbalance is severe (<5% positive). class_weight='balanced' will compensate.")
    elif pct_risk > 40:
        print("Relatively balanced dataset.")

    return df

# Step 6: Report missingness and handle via complete-case or median imputation.
def handle_missing(df: pd.DataFrame, features: list, use_impute: bool) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  MISSINGNESS SUMMARY (features + label)")
    print(f"{'='*60}")

    cols_to_check = features + [LABEL_COL]
    for col in cols_to_check:
        if col in df.columns:
            n_missing = df[col].isna().sum()
            pct_missing = (n_missing / len(df)) * 100
            flag = "missing" if pct_missing > 20 else ""
            print(f"  {col:<25} {n_missing:>5} missing  ({pct_missing:5.1f}%){flag}")

    before = len(df)

    if use_impute:
        print(f"\n  Strategy: MEDIAN IMPUTATION")
        for col in features:
            if col in df.columns and df[col].isna().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                print(f"    '{col}' imputed with median = {median_val:.3f}")
        # Drop rows where label is still missing
        df = df.dropna(subset=[LABEL_COL])
    else:
        print(f"\n  Strategy: COMPLETE CASE ANALYSIS")
        available_features = [f for f in features if f in df.columns]
        df = df.dropna(subset=available_features + [LABEL_COL])

    after = len(df)
    print(f"\n  Rows before handling: {before:,}")
    print(f"  Rows after handling:  {after:,}")
    print(f"  Rows dropped:         {before - after:,}")

    return df

# Step 7: Train and test split with a stratified 80/20 split.

def split_data(df: pd.DataFrame, features: list):
    
    available_features = [f for f in features if f in df.columns]
    missing_features = [f for f in features if f not in df.columns]

    if missing_features:
        print(f"\n Features excluded (not in data): {missing_features}")

    X = df[available_features].values
    y = df[LABEL_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print(f"\n{'='*60}")
    print("  TRAIN / TEST SPLIT (80/20 stratified)")
    print(f"{'='*60}")
    print(f"  Training samples:  {len(X_train):,}  (label=1: {y_train.sum():,})")
    print(f"  Test samples:      {len(X_test):,}  (label=1: {y_test.sum():,})")

    return X_train, X_test, y_train, y_test, available_features

# Step 8: Train model, scale features and train Logistic Regression.

def train_model(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=1000,
        solver="lbfgs",
    )
    model.fit(X_train_scaled, y_train)

    
    print("  MODEL TRAINING COMPLETE")
    print(f"  Iterations used:  {model.n_iter_[0]}")

    return model, scaler

# Step 9: Evaluate model and print full evaluation metrics.

def evaluate_model(model, scaler, X_test, y_test, feature_names):
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'-'*60}")
    print("  EVALUATION METRICS (test set)")
    print(f"{'-'*60}")
    print(f"  ROC AUC:    {roc_auc:.4f}")
    print(f"  Accuracy:   {accuracy:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}   ← prioritised (clinical: miss fewer at-risk patients)")
    print(f"  F1-score:   {f1:.4f}")
    print(f"\n  Confusion matrix:")
    print(f"                  Predicted 0    Predicted 1")
    print(f"  Actual 0:       {cm[0][0]:>10,}     {cm[0][1]:>10,}  (true neg / false pos)")
    print(f"  Actual 1:       {cm[1][0]:>10,}     {cm[1][1]:>10,}  (false neg / true pos)")
    print(f"\n  Classification report:")
    print(classification_report(y_test, y_pred, target_names=["Not at risk", "At risk"]))

    # Feature coefficients (for interpretability)
    print(f"{'-'*60}")
    print("  FEATURE COEFFICIENTS (log-odds; ranked by absolute value)")
    print(f"{'-'*60}")
    coefs = model.coef_[0]
    sorted_idx = np.argsort(np.abs(coefs))[::-1]
    for i in sorted_idx:
        direction = "↑ risk" if coefs[i] > 0 else "↓ risk"
        print(f"  {feature_names[i]:<25}  coef = {coefs[i]:+.4f}   ({direction})")


# Step 10: Save artefacts needed by the Django app.
def save_artefacts(model, scaler, feature_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    model_path   = os.path.join(output_dir, "model.pkl")
    scaler_path  = os.path.join(output_dir, "scaler.pkl")
    fnames_path  = os.path.join(output_dir, "feature_names.json")
    coefs_path   = os.path.join(output_dir, "model_coefficients.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    with open(fnames_path, "w") as f:
        json.dump(feature_names, f, indent=2)

    coef_dict = {name: float(coef)
                 for name, coef in zip(feature_names, model.coef_[0])}
    with open(coefs_path, "w") as f:
        json.dump(coef_dict, f, indent=2)

    print("  ARTEFACTS SAVED")



# Main function

def main():
    parser = argparse.ArgumentParser(
        description="Train nutritional risk ML model on real NHANES 2017-2018 data (65+)."
    )
    parser.add_argument(
        "--csv",
        default="nhanes_master_cdss.csv",
        help="Path to merged NHANES master CSV file.",
    )
    parser.add_argument(
        "--impute",
        action="store_true",
        help="Use median imputation instead of complete-case analysis.",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory to save model artefacts (default: current directory).",
    )
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("  NutriCDSS — ML TRAINING SCRIPT")
    print("  Data source: REAL NHANES 2017-2018 (adults 65+)")
    print("  NO SYNTHETIC DATA IS USED IN THIS SCRIPT")
    print(f"{'='*60}")

    # Pipeline
    df = load_csv(args.csv)
    df = resolve_columns(df)
    df = encode_sex(df)
    df = create_label(df)
    df = handle_missing(df, FEATURE_NAMES, use_impute=args.impute)

    X_train, X_test, y_train, y_test, used_features = split_data(df, FEATURE_NAMES)

    model, scaler = train_model(X_train, y_train)
    evaluate_model(model, scaler, X_test, y_test, used_features)
    save_artefacts(model, scaler, used_features, args.output_dir)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Final training sample:  {len(X_train):,}")
    print(f"  Final test sample:      {len(X_test):,}")
    print(f"  Features used:          {used_features}")
    print(f"  Label definition:       (BMI<18.5 OR albumin<3.5) AND (comorbidities>=1 OR polypharmacy)")
    print(f"  random_state:           {RANDOM_STATE}")
    print(f"\n Training complete. Artefacts saved to: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()

