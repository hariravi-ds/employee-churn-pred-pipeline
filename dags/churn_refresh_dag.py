from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

# Import ML functions
from models.churn_model import train_churn_model, predict_churn

# Default args
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# Define DAG
with DAG(
    dag_id="churn_model_weekly_refresh",
    default_args=default_args,
    description="Weekly churn retrain + prediction pipeline",
    start_date=datetime(2025, 1, 1),
    schedule="0 6 * * 1",   # every Monday 6 AM
    catchup=False,
    tags=["churn", "bigquery", "ml"],
) as dag:

    # Train model task
    train_task = PythonOperator(
        task_id="train_churn_model",
        python_callable=train_churn_model,
    )

    # Prediction task
    predict_task = PythonOperator(
        task_id="predict_churn",
        python_callable=predict_churn,
    )

    # Set dependencies
    train_task >> predict_task

