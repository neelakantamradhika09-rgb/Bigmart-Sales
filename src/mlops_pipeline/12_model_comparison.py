import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from api.preprocessing import preprocess_input


ROOT = Path(__file__).resolve().parents[2]

OLD_MODEL = ROOT / "models" / "bigmart_model.pkl"
NEW_MODEL = ROOT / "models" / "retrained_bigmart_model.pkl"

TEST_DATA = ROOT / "data" / "raw" / "Test.csv"

REPORT_PATH = (
    ROOT / "artifacts" / "model_comparison_report.json"
)


def evaluate(model, X, y=None):

    predictions = model.predict(X)

    result = {
        "mean_prediction": float(predictions.mean()),
        "prediction_std": float(predictions.std()),
    }

    if y is not None:
        result["rmse"] = float(
            mean_squared_error(y, predictions) ** 0.5
        )

        result["mae"] = float(
            mean_absolute_error(y, predictions)
        )

        result["r2"] = float(
            r2_score(y, predictions)
        )

    return result


def main():

    print("=" * 60)
    print("BIGMART MODEL COMPARISON")
    print("=" * 60)

    if not OLD_MODEL.exists():
        raise FileNotFoundError(
            f"Production model missing: {OLD_MODEL}"
        )

    if not NEW_MODEL.exists():
        raise FileNotFoundError(
            f"Retrained model missing: {NEW_MODEL}"
        )

    df = pd.read_csv(TEST_DATA)

    X = preprocess_input(
        df.to_dict(orient="records")
    )

    old_model = joblib.load(OLD_MODEL)
    new_model = joblib.load(NEW_MODEL)

    old_predictions = old_model.predict(X)
    new_predictions = new_model.predict(X)

    comparison = {
        "stage": "12_model_comparison",
        "status": "SUCCESS",
        "records": len(df),
        "old_model": {
            "mean_prediction": float(
                old_predictions.mean()
            ),
            "std_prediction": float(
                old_predictions.std()
            ),
        },
        "new_model": {
            "mean_prediction": float(
                new_predictions.mean()
            ),
            "std_prediction": float(
                new_predictions.std()
            ),
        },
        "prediction_difference": {
            "mean_absolute_difference": float(
                abs(
                    old_predictions - new_predictions
                ).mean()
            ),
            "max_absolute_difference": float(
                abs(
                    old_predictions - new_predictions
                ).max()
            ),
        },
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(
        "Mean absolute prediction difference:",
        comparison["prediction_difference"][
            "mean_absolute_difference"
        ],
    )

    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()