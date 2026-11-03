
import joblib
import json
import os
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

X_train = joblib.load("data/processed/X_train.pkl")
y_train = joblib.load("data/processed/y_train.pkl")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

joblib.dump(model, "models/bigmart_model.pkl")

registry = {
    "model_name": "BigMart_RandomForest",
    "model_version": "1.0",
    "model_type": "RandomForestRegressor",
    "training_date": datetime.now().isoformat(),
    "model_path": "models/bigmart_model.pkl",
    "status": "REGISTERED"
}

with open("reports/model_registry_report.json", "w") as f:
    json.dump(registry, f, indent=4)

print("Model Training Completed")
print("Model Registered Successfully")
