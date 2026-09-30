import json
import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_FILE = (
    BASE_DIR /
    "data" /
    "processed" /
    "train_processed.csv"
)

TEST_FILE = (
    BASE_DIR /
    "data" /
    "processed" /
    "test_processed.csv"
)

MODEL_FILE = (
    BASE_DIR /
    "models" /
    "bigmart_model.pkl"
)

IDENTIFIER_FILE = (
    BASE_DIR /
    "artifacts" /
    "test_identifiers.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

RESULT_FILE = (
    OUTPUT_DIR /
    "evaluation_results.json"
)

PREDICTION_FILE = (
    OUTPUT_DIR /
    "predictions.csv"
)

ACTUAL_PREDICTED_FILE = (
    ARTIFACTS_DIR /
    "actual_vs_predicted.csv"
)


def evaluate_model():

    train = pd.read_csv(TRAIN_FILE)

    target = "Item_Outlet_Sales"

    X = train.drop(
        columns=[target]
    )

    y = train[target]

    # Same validation split
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = joblib.load(
        MODEL_FILE
    )

    # --------------------------------
    # Validation prediction
    # --------------------------------

    predictions = model.predict(
        X_valid
    )

    # --------------------------------
    # Metrics
    # --------------------------------

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

    results = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2_Score": float(r2)
    }

    # --------------------------------
    # Create directories
    # --------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------
    # Save evaluation results
    # --------------------------------

    with open(
        RESULT_FILE,
        "w"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    # --------------------------------
    # Actual vs Predicted artifact
    # --------------------------------

    actual_vs_predicted = pd.DataFrame({
        "actual": y_valid.values,
        "predicted": predictions
    })

    actual_vs_predicted["error"] = (
        actual_vs_predicted["actual"]
        -
        actual_vs_predicted["predicted"]
    )

    actual_vs_predicted.to_csv(
        ACTUAL_PREDICTED_FILE,
        index=False
    )

    print("\nValidation Results")
    print("------------------")
    print("MAE:", round(mae, 3))
    print("MSE:", round(mse, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))

    print("\nActual vs Predicted saved:")
    print(ACTUAL_PREDICTED_FILE)

    # --------------------------------
    # Final Test prediction
    # --------------------------------

    test = pd.read_csv(
        TEST_FILE
    )

    test_predictions = model.predict(
        test
    )

    identifiers = pd.read_csv(
        IDENTIFIER_FILE
    )

    output = identifiers.copy()

    output["Item_Outlet_Sales"] = (
        test_predictions
    )

    output.to_csv(
        PREDICTION_FILE,
        index=False
    )

    print("\nTest predictions saved:")
    print(PREDICTION_FILE)

    return results


if __name__ == "__main__":
    evaluate_model()