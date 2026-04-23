
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(df, target, task="classification"):
    df = df.copy()

    # Drop rows with missing target
    df = df.dropna(subset=[target])

    # Separate features and target
    y_raw = df[target]
    X = df.drop(columns=[target])

    # Encode categorical columns in features
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Encode target if classification
    le = None
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
    else:
        y = y_raw.astype(float).values

    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X_scaled, y, le
