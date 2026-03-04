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
        print("     Check your CSV column names against NHANES_COLUMN_MAP at the top of this script.")

    return df
