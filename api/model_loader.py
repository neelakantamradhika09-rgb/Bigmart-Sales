import joblib
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(
    os.getenv(
        "BIGMART_MODEL_PATH",
        str(ROOT / "models" / "bigmart_model.pkl")
    )
)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"BigMart model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    return model, MODEL_PATH.name