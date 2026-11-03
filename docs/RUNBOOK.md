# BIGMART SALES – RUNBOOK

## 1. Project Overview

This project predicts Item Outlet Sales using
Machine Learning.

## 2. Environment Setup

Create virtual environment:

python -m venv .venv

Activate environment:

Windows PowerShell:

.venv\Scripts\Activate.ps1

## 3. Install Dependencies

pip install -r requirements.txt

## 4. Run Individual Pipeline Stages

python src/mlops_pipeline/01_load_data.py

python src/mlops_pipeline/02_validate_data.py

python src/mlops_pipeline/03_preprocessing.py

python src/mlops_pipeline/04_validate_outputs.py

python src/mlops_pipeline/05_train_registry.py

python src/mlops_pipeline/06_model_evaluation.py

python src/mlops_pipeline/07_quality_gate.py

python src/mlops_pipeline/08_automate_lifecycle.py

python src/mlops_pipeline/09_inference.py

python src/mlops_pipeline/10_failure_injection.py

python src/mlops_pipeline/validation.py

## 5. Run Complete Pipeline

python src/mlops_pipeline/bigmart_sales_pipeline.py

## 6. Check Pipeline Report

Get-Content reports/bigmart_sales_pipeline_run_log.json

## 7. GitHub Actions

1. Open the GitHub repository.
2. Select Actions.
3. Open Lab 8 - BigMart Sales CI.
4. Check workflow execution.
5. Verify all steps have green check marks.

## 8. Troubleshooting

### Missing Dependencies

pip install -r requirements.txt

### Pipeline File Not Found

Check the src/mlops_pipeline directory.

### Workflow Failure

Open GitHub Actions and inspect the failed step logs.

### Pipeline Report Missing

Run the complete pipeline again and verify the reports directory.

## 9. Conclusion

This runbook provides the commands and procedures
required to execute, monitor, and troubleshoot
the BigMart Sales ML pipeline.