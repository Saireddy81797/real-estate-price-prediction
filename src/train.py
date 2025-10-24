# src/train.py
import os
import joblib
import argparse
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from utils import load_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES

def build_pipeline():
    # Numeric pipeline
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # Categorical pipeline
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    return pipeline

def main(args):
    df = load_data(args.data_path)
    target = "price"
    if target not in df.columns:
        raise ValueError("Target column 'price' not found in the dataset.")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = build_pipeline()

    if args.tune:
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
        }
        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1, verbose=2)
        grid.fit(X_train, y_train)
        pipeline = grid.best_estimator_
        print("Best params:", grid.best_params_)
    else:
        pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test R2: {r2:.3f}")

    # Save model
    os.makedirs("src/model", exist_ok=True)
    model_path = "src/model/model_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved trained pipeline to: {model_path}")

    # Save simple feature importances if applicable
    try:
        model = pipeline.named_steps["model"]
        pre = pipeline.named_steps["preprocessor"]
        # handle onehot feature names
        num_feats = NUMERIC_FEATURES
        cat_feats = pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(CATEGORICAL_FEATURES)
        all_feats = list(num_feats) + list(cat_feats)
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"feature": all_feats, "importance": importances})
        fi_df = fi_df.sort_values("importance", ascending=False)
        fi_df.to_csv("src/artifacts/feature_importances.csv", index=False)
        print("Saved feature importances to src/artifacts/feature_importances.csv")
    except Exception as e:
        print("Could not compute feature importances:", e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="data/sample_real_estate.csv")
    parser.add_argument("--tune", action="store_true", help="Run a small grid search")
    args = parser.parse_args()
    main(args)
