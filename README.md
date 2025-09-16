📌 Employee Churn Prediction Pipeline

📖 Overview

An end-to-end ML pipeline for Employee Churn Prediction, orchestrated with Apache Airflow and powered by BigQuery, XGBoost, and Looker.

The pipeline automates weekly retraining, generates churn predictions, and delivers insights to business users via interactive Looker dashboards.

🔹 Features

Automated Retraining
Airflow DAG retrains churn models weekly on updated HR datasets in BigQuery.

ML Modeling & Explainability
XGBoost models trained on ~5M records, with SHAP-based feature attribution to explain churn drivers.

Business Impact via Looker
Predictions stored in BigQuery are automatically surfaced in Looker dashboards, enabling ROI simulations and targeted retention campaigns.

⚙️ Tech Stack

Apache Airflow → Orchestration

Google BigQuery → Data storage & querying

XGBoost, Scikit-learn, SHAP → ML training & interpretability

Looker → BI dashboards and ROI simulations
