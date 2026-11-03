
# LAB 7 – ML Pipeline Engineering

## Project Name
BigMart Sales Prediction

## 1. Objective
To develop an automated Machine Learning pipeline for BigMart Sales Prediction using Python and Scikit-learn.

## 2. Dataset
- Train.csv
- Test.csv
- Target Column: Item_Outlet_Sales
- Problem Type: Regression

## 3. Pipeline Architecture

1. Data Loading
2. Data Validation
3. Data Preprocessing
4. Processed Output Validation
5. Model Training
6. Model Evaluation
7. Model Quality Gate
8. Model Lifecycle Automation
9. Model Inference
10. Failure Injection Testing
11. Final Pipeline Validation

## 4. Pipeline Files

- 01_load_data.py
- 02_validate_data.py
- 03_preprocessing.py
- 04_validate_outputs.py
- 05_train_registry.py
- 06_model_evaluation.py
- 07_quality_gate.py
- 08_automate_lifecycle.py
- 09_inference.py
- 10_failure_injection.py
- validation.py
- bigmart_sales_pipeline.py

## 5. Model
RandomForestRegressor

## 6. Evaluation Metrics
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R2 Score

## 7. Output Artifacts
- bigmart_model.pkl
- preprocessor.pkl
- predictions.csv
- evaluation_report.json
- quality_gate_report.json
- lifecycle_report.json
- pipeline_run_log.json

## 8. Execution Command

```bash
python src/mlops_pipeline/bigmart_sales_pipeline.py
```

## 9. Expected Result

The pipeline should execute all stages sequentially, validate the processed data, train and evaluate the regression model, generate predictions, and save execution reports.

## 10. Conclusion

The BigMart Sales Prediction ML pipeline automates data loading, validation, preprocessing, model training, evaluation, inference, and failure handling.
