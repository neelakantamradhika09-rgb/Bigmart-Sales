
import joblib
import json
import os
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

model = joblib.load("models/bigmart_model.pkl")

X_val = joblib.load("data/processed/X_val.pkl")
y_val = joblib.load("data/processed/y_val.pkl")

predictions = model.predict(X_val)

mae = mean_absolute_error(y_val, predictions)
mse = mean_squared_error(y_val, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, predictions)

report = {
    "MAE": float(mae),
    "MSE": float(mse),
    "RMSE": float(rmse),
    "R2_Score": float(r2),
    "evaluation_status": "COMPLETED"
}

os.makedirs("reports", exist_ok=True)

with open("reports/evaluation_report.json", "w") as f:
    json.dump(report, f, indent=4)

print("Model Evaluation Completed")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)
