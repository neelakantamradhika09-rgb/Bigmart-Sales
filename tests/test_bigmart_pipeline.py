
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parent.parent

# Dataset paths
TRAIN_PATH = ROOT / "data/raw/Train.csv"
TEST_PATH = ROOT / "data/raw/Test.csv"

# Processed data paths
X_TRAIN_PATH = ROOT / "data/processed/X_train.pkl"
X_TEST_PATH = ROOT / "data/processed/X_test.pkl"
Y_TRAIN_PATH = ROOT / "data/processed/y_train.pkl"

# Model paths
MODEL_PATH = ROOT / "models/bigmart_model.pkl"
PREPROCESSOR_PATH = ROOT / "models/preprocessor.pkl"

# Report paths
EVALUATION_PATH = ROOT / "reports/evaluation_report.json"
QUALITY_PATH = ROOT / "reports/quality_gate_report.json"
PREDICTIONS_PATH = ROOT / "reports/predictions.csv"
PIPELINE_PATH = ROOT / "reports/bigmart_sales_pipeline_run_log.json"


# 1. Dataset tests

def test_train_dataset_exists():
    assert TRAIN_PATH.exists()


def test_test_dataset_exists():
    assert TEST_PATH.exists()


def test_training_data_readable():
    df = pd.read_csv(TRAIN_PATH)
    assert not df.empty


def test_training_target_exists():
    df = pd.read_csv(TRAIN_PATH)
    assert "Item_Outlet_Sales" in df.columns


def test_test_data_readable():
    df = pd.read_csv(TEST_PATH)
    assert not df.empty


def test_training_rows_available():
    df = pd.read_csv(TRAIN_PATH)
    assert len(df) > 0


def test_training_columns_available():
    df = pd.read_csv(TRAIN_PATH)
    assert len(df.columns) > 1


def test_target_is_numeric():
    df = pd.read_csv(TRAIN_PATH)
    target = pd.to_numeric(
        df["Item_Outlet_Sales"], errors="coerce"
    )
    assert target.notna().all()


def test_target_has_no_negative_values():
    df = pd.read_csv(TRAIN_PATH)
    assert (df["Item_Outlet_Sales"] >= 0).all()


def test_training_data_has_no_duplicate_columns():
    df = pd.read_csv(TRAIN_PATH)
    assert not df.columns.duplicated().any()


# 2. Preprocessing tests

def test_preprocessor_exists():
    assert PREPROCESSOR_PATH.exists()


def test_preprocessed_training_data_exists():
    assert X_TRAIN_PATH.exists()


def test_processed_test_data_exists():
    assert X_TEST_PATH.exists()


def test_preprocessed_arrays_not_empty():
    X_train = joblib.load(X_TRAIN_PATH)

    assert X_train is not None
    assert X_train.shape[0] > 0


def test_processed_training_values_finite():
    X_train = joblib.load(X_TRAIN_PATH)

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()

    arr = np.asarray(X_train, dtype=float)

    assert arr.size > 0
    assert np.isfinite(arr).all()


# 3. Model tests

def test_model_artifact_exists():
    assert MODEL_PATH.exists()


def test_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_has_predict_method():
    model = joblib.load(MODEL_PATH)
    assert callable(getattr(model, "predict", None))


def test_model_prediction_shape():
    model = joblib.load(MODEL_PATH)
    X_test = joblib.load(X_TEST_PATH)

    predictions = model.predict(X_test)

    assert len(predictions) == X_test.shape[0]
    assert np.isfinite(predictions).all()


# 4. Evaluation tests

def test_evaluation_report_exists():
    assert EVALUATION_PATH.exists()


def test_evaluation_report_valid_json():
    with open(EVALUATION_PATH, encoding="utf-8") as f:
        report = json.load(f)

    assert isinstance(report, dict)


def test_evaluation_metrics_are_finite():
    with open(EVALUATION_PATH, encoding="utf-8") as f:
        report = json.load(f)

    for key in ["MAE", "MSE", "RMSE", "R2"]:
        if key in report:
            assert np.isfinite(float(report[key]))


# 5. Quality gate tests

def test_quality_gate_report_exists():
    assert QUALITY_PATH.exists()


# 6. Prediction tests

def test_predictions_file_exists():
    assert PREDICTIONS_PATH.exists()


def test_predictions_not_empty():
    df = pd.read_csv(PREDICTIONS_PATH)
    assert not df.empty


def test_predictions_are_finite():
    df = pd.read_csv(PREDICTIONS_PATH)

    numeric = df.select_dtypes(include=[np.number])

    assert numeric.shape[1] > 0
    assert np.isfinite(numeric.to_numpy()).all()


# 7. Pipeline validation tests

def test_pipeline_report_exists():
    assert PIPELINE_PATH.exists()


def test_pipeline_status_success():
    with open(PIPELINE_PATH, encoding="utf-8") as f:
        report = json.load(f)

    assert report["pipeline_status"] == "SUCCESS"


def test_all_pipeline_stages_successful():
    with open(PIPELINE_PATH, encoding="utf-8") as f:
        report = json.load(f)

    assert report["completed_stages"] == report["total_stages"]

    for stage in report["stages"]:
        assert stage["status"] == "SUCCESS"
        assert stage["return_code"] == 0
