import os
import json
from datetime import datetime

REGISTRY_FILE = "artifacts/registry/model_registry.json"
OUTPUT_DIR = "artifacts/registry"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "registry_report.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# CHECK REGISTRY FILE
# --------------------------------------------------

if not os.path.exists(REGISTRY_FILE):
    raise FileNotFoundError(
        f"Model registry not found: {REGISTRY_FILE}\n"
        "Run train_registry.py first."
    )


print("=" * 60)
print("GENERATING MODEL REGISTRY REPORT")
print("=" * 60)


# --------------------------------------------------
# LOAD REGISTRY
# --------------------------------------------------

print("\nLoading model registry...")

with open(REGISTRY_FILE, "r") as file:
    registry = json.load(file)

print("Model registry loaded successfully.")


# --------------------------------------------------
# READ REGISTRY INFORMATION
# --------------------------------------------------

model_name = registry.get(
    "model_name",
    "Unknown"
)

versions = registry.get(
    "versions",
    []
)

latest_version = registry.get(
    "latest_version",
    None
)


# --------------------------------------------------
# CREATE REGISTRY REPORT
# --------------------------------------------------

report = {
    "report_generated_at": datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    ),

    "model_name": model_name,

    "total_versions": len(versions),

    "latest_version": latest_version,

    "registered_models": versions
}


# --------------------------------------------------
# SAVE REPORT
# --------------------------------------------------

with open(OUTPUT_FILE, "w") as file:
    json.dump(report, file, indent=4)


# --------------------------------------------------
# DISPLAY REPORT
# --------------------------------------------------

print("\n" + "=" * 60)
print("REGISTRY REPORT GENERATED")
print("=" * 60)

print("\nModel Name:")
print(model_name)

print("\nTotal Versions:")
print(len(versions))

print("\nLatest Version:")
print(f"v{latest_version}")


if versions:

    print("\nRegistered Versions:")

    for version in versions:

        print(
            f"  v{version.get('version')} "
            f"- {version.get('model_type')} "
            f"- {version.get('status')}"
        )


print("\nReport saved at:")
print(OUTPUT_FILE)

print("\nDone!")