import os
import json
import pickle
from datetime import datetime

MODEL_FILE = "artifacts/models/production_model.pkl"
METRICS_FILE = "artifacts/evaluation/RandomForest_metrics.json"

REGISTRY_DIR = "artifacts/registry"
REGISTRY_FILE = os.path.join(REGISTRY_DIR, "model_registry.json")

os.makedirs(REGISTRY_DIR, exist_ok=True)


# --------------------------------------------------
# CHECK REQUIRED FILES
# --------------------------------------------------

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Production model not found: {MODEL_FILE}\n"
        "Run train.py first."
    )

if not os.path.exists(METRICS_FILE):
    raise FileNotFoundError(
        f"Metrics file not found: {METRICS_FILE}\n"
        "Run train.py first."
    )


print("=" * 60)
print("MODEL REGISTRATION")
print("=" * 60)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("\nLoading production model...")

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

print("Production model loaded successfully.")


# --------------------------------------------------
# LOAD METRICS
# --------------------------------------------------

print("\nLoading model metrics...")

with open(METRICS_FILE, "r") as file:
    metrics = json.load(file)

print("Metrics loaded successfully.")


# --------------------------------------------------
# LOAD EXISTING REGISTRY
# --------------------------------------------------

if os.path.exists(REGISTRY_FILE):

    with open(REGISTRY_FILE, "r") as file:
        registry = json.load(file)

else:

    registry = {
        "model_name": "BigMart_Sales_RandomForest",
        "versions": []
    }


# --------------------------------------------------
# CREATE NEW VERSION
# --------------------------------------------------

existing_versions = registry.get("versions", [])

new_version = len(existing_versions) + 1

model_version = {
    "version": new_version,
    "model_name": "BigMart_Sales_RandomForest",
    "model_type": "RandomForestRegressor",
    "model_file": MODEL_FILE,
    "metrics_file": METRICS_FILE,
    "training_mae": metrics.get("training_mae"),
    "training_rmse": metrics.get("training_rmse"),
    "training_r2": metrics.get("training_r2"),
    "registered_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "status": "registered"
}


# --------------------------------------------------
# ADD VERSION TO REGISTRY
# --------------------------------------------------

registry["versions"].append(model_version)

registry["latest_version"] = new_version


# --------------------------------------------------
# SAVE REGISTRY
# --------------------------------------------------

with open(REGISTRY_FILE, "w") as file:
    json.dump(registry, file, indent=4)


# --------------------------------------------------
# DISPLAY RESULT
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL REGISTRATION COMPLETED")
print("=" * 60)

print("\nModel Name:")
print("BigMart_Sales_RandomForest")

print("\nModel Version:")
print(f"v{new_version}")

print("\nModel Type:")
print("RandomForestRegressor")

print("\nTraining Metrics:")
print("MAE :", metrics.get("training_mae"))
print("RMSE:", metrics.get("training_rmse"))
print("R2  :", metrics.get("training_r2"))

print("\nModel File:")
print(MODEL_FILE)

print("\nRegistry File:")
print(REGISTRY_FILE)

print("\nStatus:")
print("registered")

print("\nDone!")