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