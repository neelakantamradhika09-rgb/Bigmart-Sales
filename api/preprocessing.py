import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PREPROCESSOR_PATH = ROOT / "models" / "preprocessor.pkl"

if not PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(
        f"Preprocessor not found: {PREPROCESSOR_PATH}"
    )

PREPROCESSOR = joblib.load(PREPROCESSOR_PATH)


def preprocess_input(data):
    df = pd.DataFrame(data)

    # Apply the exact preprocessing fitted during model training
    processed_data = PREPROCESSOR.transform(df)

    return processed_data