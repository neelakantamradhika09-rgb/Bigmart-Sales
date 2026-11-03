# LAB 8 – CI Workflow Report

## Project Name
BigMart Sales Prediction

## 1. Objective
To automate the BigMart Sales Machine Learning pipeline
using GitHub Actions.

## 2. Tools Used
- Python
- Scikit-learn
- Git and GitHub
- GitHub Actions
- MLflow

## 3. Workflow Configuration

Workflow File:
.github/workflows/lab8_ci.yml

## 4. Workflow Triggers

1. Push to main branch
2. Pull requests
3. Manual workflow dispatch
4. Changes in source code
5. Changes in dataset and configuration files

## 5. CI Pipeline Stages

1. Repository Checkout
2. Python Environment Setup
3. Dependency Installation
4. Data Loading
5. Data Validation
6. Data Preprocessing
7. Model Training
8. Model Evaluation
9. Quality Gate Validation
10. Pipeline Report Verification
11. Artifact Upload

## 6. Model Details

Algorithm: RandomForestRegressor

Target Column: Item_Outlet_Sales

Problem Type: Regression

## 7. Evaluation Metrics

- MAE
- MSE
- RMSE
- R2 Score

## 8. Expected Results

- Workflow executes automatically.
- Pipeline completes successfully.
- Quality gate passes.
- Pipeline reports are generated.
- Artifacts are uploaded successfully.

## 9. Output Files

- reports/bigmart_sales_pipeline_run_log.json
- reports/lab8_ci_report.json
- artifacts/

## 10. Conclusion

The BigMart Sales ML pipeline is integrated with
GitHub Actions for automated execution, validation,
training, evaluation, and artifact collection.