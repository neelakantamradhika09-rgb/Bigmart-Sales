import os
import pickle
import numpy as np
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

TEST_FILE = "data/raw/Test.csv"

PREPROCESSOR_FILE = "artifacts/data/processed/preprocessing.pkl"

MODEL_FILE = "artifacts/models/production_model.pkl"

OUTPUT_DIR = "artifacts/predictions"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Check required files
# ============================================================

if not os.path.exists(TEST_FILE):
    raise FileNotFoundError(
        f"Test dataset not found: {TEST_FILE}"
    )

if not os.path.exists(PREPROCESSOR_FILE):
    raise FileNotFoundError(
        f"Preprocessing file not found: {PREPROCESSOR_FILE}\n"
        "Run preprocess.py first."
    )

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Production model not found: {MODEL_FILE}\n"
        "Run train.py first."
    )


# ============================================================
# 3. Load Test Dataset
# ============================================================

print("=" * 60)
print("LOADING TEST DATA")
print("=" * 60)

test_df = pd.read_csv(TEST_FILE)

print("Test dataset shape:", test_df.shape)


# ============================================================
# 4. Save Item Identifier
# ============================================================

# We keep this column so that the final prediction file
# identifies each BigMart item.

item_identifier = test_df["Item_Identifier"].copy()


# ============================================================
# 5. Prepare Test Features
# ============================================================

X_test = test_df.copy()


# Remove Item_Identifier
if "Item_Identifier" in X_test.columns:
    X_test = X_test.drop(
        columns=["Item_Identifier"]
    )


# Remove Outlet_Identifier
if "Outlet_Identifier" in X_test.columns:
    X_test = X_test.drop(
        columns=["Outlet_Identifier"]
    )


# ============================================================
# 6. Standardize Item_Fat_Content
# ============================================================

if "Item_Fat_Content" in X_test.columns:

    X_test["Item_Fat_Content"] = (
        X_test["Item_Fat_Content"]
        .replace({
            "low fat": "Low Fat",
            "LF": "Low Fat",
            "reg": "Regular"
        })
    )


# ============================================================
# 7. Load Preprocessing Object
# ============================================================

print("\nLoading preprocessing object...")

with open(
    PREPROCESSOR_FILE,
    "rb"
) as file:

    preprocessor = pickle.load(file)

print("Preprocessing object loaded successfully.")


# ============================================================
# 8. Transform Test Data
# ============================================================

print("\nPreprocessing Test.csv...")

X_test_processed = preprocessor.transform(
    X_test
)

X_test_processed = np.asarray(
    X_test_processed,
    dtype=np.float32
)

print(
    "Processed Test shape:",
    X_test_processed.shape
)


# ============================================================
# 9. Load Production Model
# ============================================================

print("\nLoading production model...")

with open(
    MODEL_FILE,
    "rb"
) as file:

    model = pickle.load(file)

print("Production model loaded successfully.")


# ============================================================
# 10. Generate Predictions
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test_processed
)


# ============================================================
# 11. Create Prediction DataFrame
# ============================================================

prediction_df = pd.DataFrame({
    "Item_Identifier": item_identifier,
    "Item_Outlet_Sales": predictions
})


# ============================================================
# 12. Save Predictions
# ============================================================

prediction_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 13. Display Results
# ============================================================

print("\n" + "=" * 60)
print("PREDICTION COMPLETED")
print("=" * 60)

print(
    "Number of predictions:",
    len(predictions)
)

print("\nFirst 10 predictions:")
print(
    prediction_df.head(10)
)

print("\nPrediction file saved at:")
print(OUTPUT_FILE)

print("\nPrediction file shape:")
print(prediction_df.shape)

print("\nDone!")