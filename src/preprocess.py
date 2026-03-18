"""
preprocess.py — UCI Student Performance Dataset
Reference: P. Cortez and A. Silva (2008)
https://archive.ics.uci.edu/dataset/320/student+performance
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")
FULL_CSV  = os.path.join(DATA_DIR, "student-mat.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "student-mat-clean.csv")


def load_data(clean=False):
    path = CLEAN_CSV if clean else FULL_CSV
    df   = pd.read_csv(path, sep=";")
    return df


def encode_features(df):
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df


def engineer_features(df, include_prior_grades=True):
    """Add 4 composite features derived from base UCI columns."""
    df = df.copy()
    df["parent_edu"] = df["Medu"] + df["Fedu"]
    df["study_fail"] = df["studytime"] / (df["failures"] + 1)
    if include_prior_grades:
        df["grade_trend"] = df["G2"] - df["G1"]
        df["avg_grade"]   = (df["G1"] + df["G2"]) / 2.0
    return df


def get_features_target(df, include_prior_grades=True):
    df_fe  = engineer_features(df, include_prior_grades)
    df_enc = encode_features(df_fe)
    drop   = ["G3"] + ([] if include_prior_grades else ["G1", "G2"])
    return df_enc.drop(columns=drop), df_enc["G3"]
