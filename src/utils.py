# src/utils.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

NUMERIC_FEATURES = ["area_sqft", "bhk", "age_years", "bathrooms", "latitude", "longitude"]
CATEGORICAL_FEATURES = ["location", "furnished", "parking", "transaction_type", "society_name"]

def load_data(path):
    """Load CSV to DataFrame and basic cleaning."""
    df = pd.read_csv(path)
    # drop rows missing target if present
    if "price" in df.columns:
        df = df.dropna(subset=["price"])
    # Basic type conversions
    df["area_sqft"] = pd.to_numeric(df["area_sqft"], errors="coerce")
    df["bhk"] = pd.to_numeric(df["bhk"], errors="coerce").fillna(1).astype(int)
    return df

class LogTransformer(BaseEstimator, TransformerMixin):
    """Applies log(1 + x) to target or numeric columns when needed."""
    def __init__(self, add_one=True):
        self.add_one = add_one

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.add_one:
            return np.log1p(X)
        else:
            return np.log(X)
