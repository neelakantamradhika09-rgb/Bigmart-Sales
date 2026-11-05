import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "raw" / "Train.csv"

MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "artifacts"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Item_Outlet_Sales"


def main():

    print("=" * 60)
    print("BIGMART MODEL RETRAINING")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load training data
    # ---------------------------------------------------------
    df = pd.read_csv(TRAIN_PATH)

    print(f"Training dataset shape: {df.shape}")

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found"
        )

    # Remove rows where target itself is missing
    df = df.dropna(subset=[TARGET]).copy()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # ---------------------------------------------------------
    # 2. Identify categorical and numerical columns
    # ---------------------------------------------------------
    categorical = X.select_dtypes(
        include=["str", "object"]
    ).columns.tolist()

    numerical = X.select_dtypes(
        exclude=["str", "object"]
    ).columns.tolist()

    print(f"Numerical features   : {len(numerical)}")
    print(f"Categorical features : {len(categorical)}")

    # ---------------------------------------------------------
    # 3. Build preprocessing pipelines
    # ---------------------------------------------------------
    # Numerical:
    #   - replace missing values with median
    #
    # Categorical:
    #   - replace missing values with most frequent value
    #   - one-hot encode categories
    #
    # This prevents NaN values from reaching RandomForest.
    # ---------------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numerical_pipeline,
                numerical,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical,
            ),
        ]
    )

    # ---------------------------------------------------------
    # 4. Train / validation split
    # ---------------------------------------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print(f"Training rows       : {len(X_train)}")
    print(f"Validation rows     : {len(X_val)}")

    # ---------------------------------------------------------
    # 5. Fit preprocessing
    # ---------------------------------------------------------
    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_val_processed = preprocessor.transform(
        X_val
    )

    print(
        f"Processed training shape   : {X_train_processed.shape}"
    )
    print(
        f"Processed validation shape : {X_val_processed.shape}"
    )

    # ---------------------------------------------------------
    # 6. Create Random Forest model
    # ---------------------------------------------------------
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    # ---------------------------------------------------------
    # 7. Train model
    # ---------------------------------------------------------
    model.fit(
        X_train_processed,
        y_train,
    )

    # ---------------------------------------------------------
    # 8. Validation prediction
    # ---------------------------------------------------------
    predictions = model.predict(
        X_val_processed
    )

    # ---------------------------------------------------------
    # 9. Calculate metrics
    # ---------------------------------------------------------
    rmse = mean_squared_error(
        y_val,
        predictions,
    ) ** 0.5

    mae = mean_absolute_error(
        y_val,
        predictions,
    )

    r2 = r2_score(
        y_val,
        predictions,
    )

    print("\nRetraining metrics:")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R2   : {r2:.4f}")

    # ---------------------------------------------------------
    # 10. Save retrained model
    # ---------------------------------------------------------
    model_path = (
        MODEL_DIR / "retrained_bigmart_model.pkl"
    )

    preprocessor_path = (
        MODEL_DIR / "retrained_bigmart_preprocessor.pkl"
    )

    joblib.dump(
        model,
        model_path,
    )

    joblib.dump(
        preprocessor,
        preprocessor_path,
    )

    # ---------------------------------------------------------
    # 11. Save retraining report
    # ---------------------------------------------------------
    report = {
        "stage": "12_retrain_model",
        "status": "SUCCESS",
        "model": "RandomForestRegressor",
        "training_rows": len(X_train),
        "validation_rows": len(X_val),
        "features": X.shape[1],
        "processed_features": int(
            X_train_processed.shape[1]
        ),
        "missing_values_handled": True,
        "preprocessing": {
            "numerical_imputation": "median",
            "categorical_imputation": "most_frequent",
            "categorical_encoding": "OneHotEncoder(handle_unknown=ignore)",
        },
        "parameters": {
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        },
        "metrics": {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
        },
        "model_path": str(model_path),
        "preprocessor_path": str(preprocessor_path),
    }

    report_path = (
        ARTIFACT_DIR / "retraining_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # 12. Final output
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("RETRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R2   : {r2:.4f}")
    print(f"Model: {model_path}")
    print(f"Preprocessor: {preprocessor_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()