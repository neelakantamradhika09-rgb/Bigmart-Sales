\# Lab 13 - End-to-End MLOps System



\## Project



BigMart Sales Prediction



\## Objective



Develop and validate an end-to-end MLOps workflow for the BigMart Sales prediction system.



The workflow integrates:



\- Data preprocessing

\- Data validation

\- Model training

\- Experiment tracking

\- Model registry

\- Model evaluation

\- Model deployment

\- Kubernetes deployment

\- Performance monitoring

\- Data drift monitoring

\- Prediction drift monitoring

\- Concept drift monitoring

\- Retraining

\- Model registry update

\- CI/CD validation



\## Monitoring



The following monitoring stages are executed:



1\. Data drift

2\. Prediction drift

3\. Concept drift

4\. Drift analysis

5\. API performance monitoring

6\. Prediction monitoring

7\. Performance monitoring



\## Drift Detection



The BigMart monitoring workflow evaluates:



\- PSI-based data drift

\- Prediction distribution drift

\- Concept drift using model performance



If concept drift exceeds the configured threshold, retraining is recommended.



\## Retraining



The existing BigMart retraining pipeline is executed using:



`12\_retrain\_model.py`



The retrained model and preprocessing artifacts are generated under the models directory.



\## Model Registry



The retrained BigMart model is registered using:



`12\_update\_mlflow\_registry.py`



Registered model:



`BigMart\_Sales\_Production\_Model`



\## CI/CD



GitHub Actions workflows:



\- `lab13\_monitoring.yml`

\- `lab13\_runner\_test.yml`



These workflows validate the monitoring and MLOps pipeline automatically.



\## End-to-End Validation



The final validation confirms:



\- Required scripts exist

\- Required artifacts exist

\- Monitoring executes successfully

\- Retraining executes successfully

\- MLflow registry update succeeds

\- Reports are generated

\- GitHub Actions workflow passes



\## Result



The BigMart Sales MLOps system is validated through monitoring, drift detection, retraining, model registry management, deployment and CI/CD automation.

