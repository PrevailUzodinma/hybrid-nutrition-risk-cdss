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
    print(f"\n{'='*60}")
    print(f"  NHANES MASTER CSV LOADED")
    print(f"{'='*60}")
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
            status = f"'{found}' → '{canonical}'"
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
    make sex_encoded is 0=Female, 1=Male, because NHANES uses RIAGENDR: 1=Male, 2=Female, so recode to 0/1.
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

    print(f"\n{'='*60}")
    print("  LABEL CREATION (GLIM-INSPIRED OPERATIONAL DEFINITION)")
    print(f"{'='*60}")

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

