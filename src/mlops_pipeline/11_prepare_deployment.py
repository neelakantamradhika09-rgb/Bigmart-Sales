"""
BigMart Sales - Stage 11: Prepare Deployment

Prepares the trained Random Forest model and encoder
for deployment.

Input artifacts:
    artifacts/RandomForest_model.pkl
    artifacts/encoder.pkl
    artifacts/RandomForest_features.json
    artifacts/RandomForest_parameters.json

Output:
    deployment/
        model.pkl
        encoder.pkl
        model_metadata.json

    artifacts/deployment_preparation_report.json
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"

MODEL_SOURCE = ARTIFACTS_DIR / "RandomForest_model.pkl"
ENCODER_SOURCE = ARTIFACTS_DIR / "encoder.pkl"
FEATURES_SOURCE = ARTIFACTS_DIR / "RandomForest_features.json"
PARAMETERS_SOURCE = ARTIFACTS_DIR / "RandomForest_parameters.json"

MODEL_DEST = DEPLOYMENT_DIR / "model.pkl"
ENCODER_DEST = DEPLOYMENT_DIR / "encoder.pkl"
METADATA_DEST = DEPLOYMENT_DIR / "model_metadata.json"

REPORT_PATH = ARTIFACTS_DIR / "deployment_preparation_report.json"


def log(message: str) -> None:
    print(f"[INFO] {message}")


def fail(message: str) -> None:
    print(f"[ERROR] {message}")
    raise SystemExit(1)


def check_required_artifacts() -> None:
    log("Checking BigMart deployment artifacts...")

    required = [
        MODEL_SOURCE,
        ENCODER_SOURCE,
        FEATURES_SOURCE,
        PARAMETERS_SOURCE,
    ]

    missing = [str(path) for path in required if not path.exists()]

    if missing:
        fail(
            "Required BigMart artifacts are missing:\n"
            + "\n".join(f"  - {item}" for item in missing)
        )

    for path in required:
        log(f"Found: {path.relative_to(PROJECT_ROOT)}")


def load_json(path: Path) -> dict | list:
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        fail(f"Invalid JSON in {path.name}: {exc}")


def prepare_deployment_directory() -> None:
    DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

    log("Preparing deployment directory...")

    shutil.copy2(MODEL_SOURCE, MODEL_DEST)
    log(f"Copied model: {MODEL_DEST.relative_to(PROJECT_ROOT)}")

    shutil.copy2(ENCODER_SOURCE, ENCODER_DEST)
    log(f"Copied encoder: {ENCODER_DEST.relative_to(PROJECT_ROOT)}")


def create_metadata() -> dict:
    features = load_json(FEATURES_SOURCE)
    parameters = load_json(PARAMETERS_SOURCE)

    metadata = {
        "project": "bigmart-sales",
        "model_name": "RandomForest",
        "model_type": "RandomForestRegressor",
        "source_model": "artifacts/RandomForest_model.pkl",
        "deployment_model": "deployment/model.pkl",
        "source_encoder": "artifacts/encoder.pkl",
        "deployment_encoder": "deployment/encoder.pkl",
        "features": features,
        "model_parameters": parameters,
        "deployment_prepared": True,
        "deployment_prepared_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    with METADATA_DEST.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    log(
        f"Created metadata: "
        f"{METADATA_DEST.relative_to(PROJECT_ROOT)}"
    )

    return metadata


def create_report(metadata: dict) -> None:
    report = {
        "stage": "11_prepare_deployment",
        "status": "SUCCESS",
        "project": "bigmart-sales",
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "deployment_directory": "deployment",
        "files": {
            "model": {
                "path": "deployment/model.pkl",
                "exists": MODEL_DEST.exists(),
                "size_bytes": MODEL_DEST.stat().st_size,
            },
            "encoder": {
                "path": "deployment/encoder.pkl",
                "exists": ENCODER_DEST.exists(),
                "size_bytes": ENCODER_DEST.stat().st_size,
            },
            "metadata": {
                "path": "deployment/model_metadata.json",
                "exists": METADATA_DEST.exists(),
                "size_bytes": METADATA_DEST.stat().st_size,
            },
        },
        "model_metadata": metadata,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    log(
        f"Created report: "
        f"{REPORT_PATH.relative_to(PROJECT_ROOT)}"
    )


def main() -> None:
    log("=" * 60)
    log("BigMart Sales - Stage 11 Deployment Preparation")
    log("=" * 60)

    check_required_artifacts()

    prepare_deployment_directory()

    metadata = create_metadata()

    create_report(metadata)

    log("")
    log("Deployment preparation completed successfully.")
    log("")
    log("Deployment artifacts:")

    log("  - deployment/model.pkl")
    log("  - deployment/encoder.pkl")
    log("  - deployment/model_metadata.json")
    log("")
    log("Report:")
    log("  - artifacts/deployment_preparation_report.json")
    log("")
    log("Status: SUCCESS")
    log("=" * 60)


if __name__ == "__main__":
    main()