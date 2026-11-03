
import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

train = pd.read_csv("data/raw/Train.csv")
test = pd.read_csv("data/raw/Test.csv")

target = "Item_Outlet_Sales"

X = train.drop(columns=[target])
y = train[target]

if not pd.api.types.is_numeric_dtype(y):
    raise ValueError("Target must be numeric")

if y.isnull().any():
    raise ValueError("Target contains missing values")

if not np.isfinite(y.to_numpy()).all():
    raise ValueError("Target contains invalid numeric values")

if target in test.columns:
    test = test.drop(columns=[target])

# Split training data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

numeric_cols = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_cols = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_cols),
    ("categorical", categorical_pipeline, categorical_cols)
])

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform validation and test data
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(test)

# Save preprocessing model
joblib.dump(preprocessor, "models/preprocessor.pkl")

# Save processed datasets
joblib.dump(X_train_processed, "data/processed/X_train.pkl")
joblib.dump(X_val_processed, "data/processed/X_val.pkl")
joblib.dump(X_test_processed, "data/processed/X_test.pkl")

joblib.dump(y_train, "data/processed/y_train.pkl")
joblib.dump(y_val, "data/processed/y_val.pkl")

report = {
    "numeric_features": numeric_cols,
    "categorical_features": categorical_cols,
    "train_shape": list(X_train_processed.shape),
    "validation_shape": list(X_val_processed.shape),
    "test_shape": list(X_test_processed.shape),
    "status": "SUCCESS"
}

with open("reports/preprocessing_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Preprocessing Completed Successfully")
print("Training Shape:", X_train_processed.shape)
print("Validation Shape:", X_val_processed.shape)
print("Test Shape:", X_test_processed.shape)
