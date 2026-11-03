# MLFLOW CI SETUP

## Project Name
BigMart Sales Prediction

## 1. Objective

To integrate MLflow experiment tracking and model
management with the BigMart Sales CI workflow.

## 2. Tools Used

- MLflow
- Python
- Scikit-learn
- GitHub Actions

## 3. MLflow Configuration

MLflow is used to track machine learning experiments,
parameters, metrics, and model versions.

## 4. Experiment Details

Experiment Name: BigMart-Sales-Prediction

Model: RandomForestRegressor

Problem Type: Regression

Target Column: Item_Outlet_Sales

## 5. Parameters Tracked

- n_estimators
- max_depth
- random_state

## 6. Metrics Tracked

- Mean Absolute Error
- Mean Squared Error
- Root Mean Squared Error
- R2 Score

## 7. Model Registry

The trained BigMart Sales model can be registered
and managed using MLflow Model Registry.

## 8. CI Integration

GitHub Actions executes the ML pipeline whenever
configured repository changes occur.

MLflow tracking configuration must be available
inside the GitHub Actions environment.

## 9. Conclusion

MLflow provides experiment tracking and model
management for the BigMart Sales prediction system.