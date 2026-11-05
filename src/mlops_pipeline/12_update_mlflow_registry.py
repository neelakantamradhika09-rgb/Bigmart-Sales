import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "models" / "retrained_bigmart_model.pkl"
)

PREPROCESSOR_PATH = (
    ROOT / "models" / "retrained_bigmart_preprocessor.pkl"
)

RETRAINING_REPORT_PATH = (
    ROOT / "artifacts" / "retraining_report.json"
)

REPORT_PATH = (
    ROOT / "artifacts" / "mlflow_retraining_report.json"
)

MODEL_NAME = "BigMart_Sales_Production_Model"


def main():

    print("=" * 60)
    print("BIGMART MLFLOW REGISTRY UPDATE")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Validate required files
    # ---------------------------------------------------------
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Retrained model not found: {MODEL_PATH}"
        )

    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(
            f"Retrained preprocessor not found: {PREPROCESSOR_PATH}"
        )

    if not RETRAINING_REPORT_PATH.exists():
        raise FileNotFoundError(
            f"Retraining report not found: {RETRAINING_REPORT_PATH}"
        )

    # ---------------------------------------------------------
    # 2. Load retraining report
    # ---------------------------------------------------------
    with open(
        RETRAINING_REPORT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        retraining_report = json.load(f)

    metrics = retraining_report.get(
        "metrics",
        {}
    )

    parameters = retraining_report.get(
        "parameters",
        {}
    )

    # ---------------------------------------------------------
    # 3. Configure MLflow
    # ---------------------------------------------------------
    mlflow.set_tracking_uri(
        "sqlite:///mlflow.db"
    )

    experiment_name = "BigMart_Sales_Monitoring"

    experiment = mlflow.get_experiment_by_name(
        experiment_name
    )

    if experiment is None:
        mlflow.create_experiment(
            experiment_name
        )

        print(
            f"Created MLflow experiment: {experiment_name}"
        )

    mlflow.set_experiment(
        experiment_name
    )

    print(
        f"MLflow experiment: {experiment_name}"
    )

    # ---------------------------------------------------------
    # 4. Load retrained model
    # ---------------------------------------------------------
    model = joblib.load(
        MODEL_PATH
    )

    # Load preprocessor so it can be stored as an artifact
    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    # ---------------------------------------------------------
    # 5. Start MLflow run
    # ---------------------------------------------------------
    with mlflow.start_run() as run:

        run_id = run.info.run_id

        print(
            f"MLflow Run ID: {run_id}"
        )

        # -----------------------------------------------------
        # 6. Log metrics
        # -----------------------------------------------------
        for metric_name, metric_value in metrics.items():

            mlflow.log_metric(
                metric_name,
                float(metric_value),
            )

        # -----------------------------------------------------
        # 7. Log parameters
        # -----------------------------------------------------
        for parameter_name, parameter_value in parameters.items():

            mlflow.log_param(
                parameter_name,
                parameter_value,
            )

        # -----------------------------------------------------
        # 8. Log tags
        # -----------------------------------------------------
        mlflow.set_tag(
            "workflow",
            "lab12_retraining",
        )

        mlflow.set_tag(
            "model_type",
            "RandomForestRegressor",
        )

        mlflow.set_tag(
            "source_model",
            "retrained_bigmart_model.pkl",
        )

        # -----------------------------------------------------
        # 9. Log preprocessor
        # -----------------------------------------------------
        mlflow.log_artifact(
            str(PREPROCESSOR_PATH),
            artifact_path="preprocessor",
        )

        # -----------------------------------------------------
        # 10. Log sklearn model
        #
        # RandomForestRegressor internally uses
        # sklearn.tree._tree.Tree.
        #
        # This is the specific trusted type reported by skops.
        # -----------------------------------------------------
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ],
        )

        model_uri = model_info.model_uri

        print(
            f"MLflow model logged successfully: {model_uri}"
        )

        # -----------------------------------------------------
        # 11. Register model
        # -----------------------------------------------------
        result = mlflow.register_model(
            model_uri,
            MODEL_NAME,
        )

        # -----------------------------------------------------
        # 12. Create report
        # -----------------------------------------------------
        report = {
            "stage": "12_update_mlflow_registry",
            "status": "SUCCESS",
            "run_id": run_id,
            "registered_model": MODEL_NAME,
            "version": result.version,
            "model_uri": model_uri,
            "tracking_uri": mlflow.get_tracking_uri(),
            "experiment_name": experiment_name,
            "metrics": {
                key: float(value)
                for key, value in metrics.items()
            },
            "parameters": parameters,
            "model_path": str(MODEL_PATH),
            "preprocessor_path": str(PREPROCESSOR_PATH),
        }

    # ---------------------------------------------------------
    # 13. Save registry report
    # ---------------------------------------------------------
    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # 14. Final output
    # ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("MLFLOW REGISTRY UPDATE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        f"Registered model: {MODEL_NAME}"
    )

    print(
        f"Version: {result.version}"
    )

    print(
        f"Run ID: {run_id}"
    )

    print(
        f"Model URI: {model_uri}"
    )

    print(
        f"Report: {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()