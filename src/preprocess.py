import os
import json
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ============================================================
# 1. Paths
# ============================================================

INPUT_FILE = "data/raw/Train.csv"

OUTPUT_DIR = "artifacts/data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. Load Dataset
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)
print(df.head())


# ============================================================
# 3. Remove unnecessary columns
# ============================================================

# Item_Identifier is an ID, so it is not directly useful
if "Item_Identifier" in df.columns:
    df = df.drop(columns=["Item_Identifier"])

# Outlet_Identifier is also an ID
if "Outlet_Identifier" in df.columns:
    df = df.drop(columns=["Outlet_Identifier"])


# ============================================================
# 4. Target and Features
# ============================================================

TARGET = "Item_Outlet_Sales"

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# 5. Identify numerical and categorical columns
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print("Numerical columns:", numeric_features)
print("Categorical columns:", categorical_features)


# ============================================================
# 6. Numerical preprocessing
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ============================================================
# 7. Categorical preprocessing
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]
)


# ============================================================
# 8. Combine preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# 9. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ============================================================
# 10. Fit preprocessing only on training data
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# ============================================================
# 11. Convert to NumPy arrays
# ============================================================

X_train_processed = np.asarray(X_train_processed, dtype=np.float32)
X_test_processed = np.asarray(X_test_processed, dtype=np.float32)

y_train = np.asarray(y_train, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.float32)


# ============================================================
# 12. Save processed data
# ============================================================

np.save(
    os.path.join(OUTPUT_DIR, "X_train.npy"),
    X_train_processed
)

np.save(
    os.path.join(OUTPUT_DIR, "X_test.npy"),
    X_test_processed
)

np.save(
    os.path.join(OUTPUT_DIR, "y_train.npy"),
    y_train
)

np.save(
    os.path.join(OUTPUT_DIR, "y_test.npy"),
    y_test
)


# ============================================================
# 13. Save preprocessing object
# ============================================================

with open(
    os.path.join(OUTPUT_DIR, "preprocessing.pkl"),
    "wb"
) as f:
    pickle.dump(preprocessor, f)


# ============================================================
# 14. Save metadata
# ============================================================

metadata = {
    "dataset_name": "BigMart Sales",
    "target_column": TARGET,
    "original_rows": int(df.shape[0]),
    "original_columns": int(df.shape[1]),
    "train_rows": int(X_train.shape[0]),
    "test_rows": int(X_test.shape[0]),
    "processed_train_shape": list(X_train_processed.shape),
    "processed_test_shape": list(X_test_processed.shape),
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "test_size": 0.20,
    "random_state": 42
}

with open(
    os.path.join(OUTPUT_DIR, "dataset_metadata.json"),
    "w"
) as f:
    json.dump(metadata, f, indent=4)


# ============================================================
# 15. Final output
# ============================================================

print("\nPreprocessing completed successfully!")

print("\nSaved files:")

print("X_train.npy")
print("X_test.npy")
print("y_train.npy")
print("y_test.npy")
print("preprocessing.pkl")
print("dataset_metadata.json")

print("\nX_train shape:", X_train_processed.shape)
print("X_test shape :", X_test_processed.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)