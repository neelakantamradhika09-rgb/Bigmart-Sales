
import pandas as pd
import joblib
import os
import json

os.makedirs("artifacts", exist_ok=True)
os.makedirs("reports", exist_ok=True)

model = joblib.load("models/bigmart_model.pkl")
X_test = joblib.load("data/processed/X_test.pkl")

predictions = model.predict(X_test)

output = pd.DataFrame({
    "Predicted_Item_Outlet_Sales": predictions
})

test = pd.read_csv("data/raw/Test.csv")

for col in ["Item_Identifier", "Outlet_Identifier"]:
    if col in test.columns:
        output.insert(
            0, col, test[col].values
        )

output.to_csv(
    "artifacts/predictions.csv",
    index=False
)

report = {
    "total_predictions": len(predictions),
    "output_file": "artifacts/predictions.csv",
    "status": "SUCCESS"
}

with open("reports/inference_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Inference Completed")
print("Predictions Saved Successfully")
