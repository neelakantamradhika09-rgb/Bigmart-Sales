import os
import sys
import json
import subprocess
from datetime import datetime

# Project root directory
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

# Pipeline stages
PIPELINE_STAGES = [
    "01_load_data.py",
    "02_validate_data.py",
    "03_preprocessing.py",
    "04_validate_outputs.py",
    "05_train_registry.py",
    "06_model_evaluation.py",
    "07_quality_gate.py",
    "08_automate_lifecycle.py",
    "09_inference.py",
    "10_failure_injection.py",
    "validation.py"
]

LOG_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(LOG_DIR, exist_ok=True)

RUN_LOG = os.path.join(
    LOG_DIR,
    "bigmart_sales_pipeline_run_log.json"
)


def run_pipeline():

    results = []

    print("\n===================================")
    print(" BIGMART SALES MLOPS PIPELINE")
    print("===================================\n")

    for stage in PIPELINE_STAGES:

        script_path = os.path.join(
            PROJECT_ROOT,
            "src",
            "mlops_pipeline",
            stage
        )

        print(f"\nRunning Stage: {stage}")

        start_time = datetime.now().isoformat()

        if not os.path.exists(script_path):
            print(f"ERROR: File not found: {script_path}")

            results.append({
                "stage": stage,
                "status": "FAILED",
                "message": "Script file not found",
                "start_time": start_time
            })

            break

        try:
            process = subprocess.run(
                [sys.executable, script_path],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True
            )

            status = (
                "SUCCESS"
                if process.returncode == 0
                else "FAILED"
            )

            results.append({
                "stage": stage,
                "status": status,
                "start_time": start_time,
                "end_time": datetime.now().isoformat(),
                "return_code": process.returncode,
                "output": process.stdout,
                "error": process.stderr
            })

            print(process.stdout)

            if process.returncode != 0:
                print(f"Pipeline stopped at: {stage}")
                print(process.stderr)
                break

            print(f"{stage} completed successfully")

        except Exception as error:

            results.append({
                "stage": stage,
                "status": "FAILED",
                "message": str(error)
            })

            print("Pipeline Error:", error)
            break

    overall_status = (
        "SUCCESS"
        if len(results) == len(PIPELINE_STAGES)
        and all(r["status"] == "SUCCESS" for r in results)
        else "FAILED"
    )

    report = {
        "project": "BigMart Sales Prediction",
        "pipeline_status": overall_status,
        "total_stages": len(PIPELINE_STAGES),
        "completed_stages": sum(
            r["status"] == "SUCCESS" for r in results
        ),
        "execution_time": datetime.now().isoformat(),
        "stages": results
    }

    with open(RUN_LOG, "w") as file:
        json.dump(report, file, indent=4)

    print("\n===================================")
    print("PIPELINE STATUS:", overall_status)
    print("Completed Stages:",
          report["completed_stages"])
    print("Report Saved:", RUN_LOG)
    print("===================================")


if __name__ == "__main__":
    run_pipeline()