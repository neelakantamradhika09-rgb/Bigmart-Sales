import os
import sys
import subprocess
from datetime import datetime


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)


# --------------------------------------------------
# PIPELINE STEPS
# --------------------------------------------------

PIPELINE_STEPS = [
    ("Data Validation", "src/validate_data.py"),
    ("Preprocessing", "src/preprocess.py"),
    ("Model Training", "src/train.py"),
    ("Model Registration", "src/train_registry.py"),
    ("Registry Report", "src/generate_registry.py"),
    ("Prediction", "src/predict.py")
]


# --------------------------------------------------
# RUN SCRIPT
# --------------------------------------------------

def run_step(step_name, script_path):

    print("\n" + "=" * 70)
    print(f"STARTING: {step_name}")
    print("=" * 70)

    full_path = os.path.join(PROJECT_ROOT, script_path)

    if not os.path.exists(full_path):
        print(f"ERROR: Script not found: {full_path}")
        return False

    try:

        result = subprocess.run(
            [sys.executable, full_path],
            cwd=PROJECT_ROOT,
            check=True
        )

        print("\n" + "-" * 70)
        print(f"{step_name} COMPLETED SUCCESSFULLY")
        print("-" * 70)

        return True

    except subprocess.CalledProcessError as error:

        print("\n" + "-" * 70)
        print(f"{step_name} FAILED")
        print("-" * 70)

        print(f"Exit code: {error.returncode}")

        return False


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    start_time = datetime.now()

    print("=" * 70)
    print("BIGMART SALES - MLOPS AUTOMATED LIFECYCLE")
    print("=" * 70)

    print("\nProject root:")
    print(PROJECT_ROOT)

    print("\nPipeline started at:")
    print(start_time.strftime("%Y-%m-%d %H:%M:%S"))


    # --------------------------------------------------
    # RUN ALL STEPS
    # --------------------------------------------------

    completed_steps = 0

    for step_name, script_path in PIPELINE_STEPS:

        success = run_step(
            step_name,
            script_path
        )

        if not success:

            print("\n" + "=" * 70)
            print("PIPELINE FAILED")
            print("=" * 70)

            print(
                f"\nPipeline stopped at: {step_name}"
            )

            print(
                "\nFix the error and run the pipeline again."
            )

            sys.exit(1)

        completed_steps += 1


    # --------------------------------------------------
    # PIPELINE COMPLETED
    # --------------------------------------------------

    end_time = datetime.now()

    duration = end_time - start_time


    print("\n" + "=" * 70)
    print("BIGMART MLOPS PIPELINE COMPLETED")
    print("=" * 70)

    print("\nCompleted steps:")

    for index, (step_name, _) in enumerate(
        PIPELINE_STEPS,
        start=1
    ):

        print(f"{index}. {step_name} - SUCCESS")


    print("\nTotal steps:")
    print(completed_steps)

    print("\nStarted:")
    print(start_time.strftime("%Y-%m-%d %H:%M:%S"))

    print("Finished:")
    print(end_time.strftime("%Y-%m-%d %H:%M:%S"))

    print("Duration:")
    print(duration)

    print("\nImportant outputs:")

    print(
        "\nValidation reports:"
        "\n  artifacts/validation/"
    )

    print(
        "\nProcessed data:"
        "\n  artifacts/data/processed/"
    )

    print(
        "\nProduction model:"
        "\n  artifacts/models/production_model.pkl"
    )

    print(
        "\nModel registry:"
        "\n  artifacts/registry/model_registry.json"
    )

    print(
        "\nRegistry report:"
        "\n  artifacts/registry/registry_report.json"
    )

    print(
        "\nPredictions:"
        "\n  artifacts/predictions/test_predictions.csv"
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    main()