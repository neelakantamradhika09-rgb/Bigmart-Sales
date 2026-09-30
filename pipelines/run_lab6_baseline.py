import sys
import os

# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from src.train_mlflow import train_with_mlflow


def main():

    print("=" * 60)
    print("BIGMART SALES - LAB 6")
    print("=" * 60)

    print("\n[1] Model Training with MLflow")
    train_with_mlflow()

    print("\n[2] MLflow Experiment Tracking")
    print("Parameters tracked")
    print("Metrics tracked")
    print("Model artifact stored")

    print("\nLab 6 completed successfully.")


if __name__ == "__main__":
    main()