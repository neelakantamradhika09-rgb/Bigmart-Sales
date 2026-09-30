
import json
import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "train_processed.csv"
)

ARTIFACTS_DIR = (
    BASE_DIR
    / "artifacts"
)

MLFLOW_PARAMETERS_FILE = (
    ARTIFACTS_DIR
    / "mlflow_parameters.json"
)

MLFLOW_METRICS_FILE = (
    ARTIFACTS_DIR
    / "mlflow_metrics.json"
)


# ============================================================
# MLflow Training Function
# ============================================================

def train_with_mlflow():

    # --------------------------------------------------------
    # Load processed training data
    # --------------------------------------------------------

    df = pd.read_csv(
        TRAIN_FILE
    )

    print("Processed training data loaded.")
    print("Train shape:", df.shape)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target = "Item_Outlet_Sales"

    X = df.drop(
        columns=[target]
    )

    y = df[target]

    # --------------------------------------------------------
    # Train / Validation Split
    # Same as CLA-1
    # --------------------------------------------------------

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("Train samples:", len(X_train))
    print("Validation samples:", len(X_valid))

    # --------------------------------------------------------
    # Random Forest Configuration
    # Same as CLA-1
    # --------------------------------------------------------

    n_estimators = 200
    max_depth = 10
    random_state = 42

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )

    # --------------------------------------------------------
    # Create artifacts directory
    # --------------------------------------------------------

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # MLflow Experiment
    # --------------------------------------------------------

    mlflow.set_experiment(
        "BigMart-Sales"
    )

    # --------------------------------------------------------
    # Start MLflow Run
    # --------------------------------------------------------

    with mlflow.start_run():

        print("\nMLflow run started.")

        # ----------------------------------------------------
        # Train model
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train
        )

        print("Model training completed.")

        # ----------------------------------------------------
        # Validation prediction
        # ----------------------------------------------------

        predictions = model.predict(
            X_valid
        )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        mae = mean_absolute_error(
            y_valid,
            predictions
        )

        mse = mean_squared_error(
            y_valid,
            predictions
        )

        rmse = mse ** 0.5

        r2 = r2_score(
            y_valid,
            predictions
        )

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        parameters = {
            "model": "RandomForestRegressor",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": random_state,
            "n_jobs": -1,
            "test_size": 0.20
        }

        # ----------------------------------------------------
        # Log parameters to MLflow
        # ----------------------------------------------------

        mlflow.log_params(
            parameters
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metrics = {
            "MAE": float(mae),
            "MSE": float(mse),
            "RMSE": float(rmse),
            "R2_Score": float(r2)
        }

        # ----------------------------------------------------
        # Log metrics to MLflow
        # ----------------------------------------------------

        mlflow.log_metrics(
            metrics
        )

        # ----------------------------------------------------
        # Save parameters locally
        # ----------------------------------------------------

        with open(
            MLFLOW_PARAMETERS_FILE,
            "w"
        ) as file:

            json.dump(
                parameters,
                file,
                indent=4
            )

        # ----------------------------------------------------
        # Save metrics locally
        # ----------------------------------------------------

        with open(
            MLFLOW_METRICS_FILE,
            "w"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4
            )

        # ----------------------------------------------------
        # Log model to MLflow
        #
        # skops requires explicit trust for the internal
        # sklearn Tree type used by RandomForestRegressor.
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            name="bigmart_random_forest",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print("\n" + "=" * 50)
        print("MLFLOW RUN COMPLETED")
        print("=" * 50)

        print("\nParameters")
        print("----------")
        print(
            "Model:",
            parameters["model"]
        )

        print(
            "n_estimators:",
            parameters["n_estimators"]
        )

        print(
            "max_depth:",
            parameters["max_depth"]
        )

        print(
            "random_state:",
            parameters["random_state"]
        )

        print("\nMetrics")
        print("-------")

        print(
            "MAE:",
            round(mae, 3)
        )

        print(
            "MSE:",
            round(mse, 3)
        )

        print(
            "RMSE:",
            round(rmse, 3)
        )

        print(
            "R2 Score:",
            round(r2, 3)
        )

        print("\nArtifacts")
        print("---------")

        print(
            "MLflow parameters:",
            MLFLOW_PARAMETERS_FILE
        )

        print(
            "MLflow metrics:",
            MLFLOW_METRICS_FILE
        )

        print(
            "\nMLflow model artifact:"
            " bigmart_random_forest"
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    train_with_mlflow()

