import os
import joblib
import pandas as pd
from datetime import datetime
from google.cloud import bigquery, storage
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import shap
import logging

# GCP settings
PROJECT_ID = "robotic-facet-471416-s5"
BUCKET_NAME = "my-ml-model-bucket"
MODEL_PATH = "models/churn/"

# BigQuery tables
TRAIN_TABLE = "my_employee_data.tbl_hr_data"
TEST_TABLE = "my_employee_data.tl_new_employee"

# Initialize clients
bq_client = bigquery.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)


def preprocess(df: pd.DataFrame):
    """Prepare features and target."""
    df = df.drop(columns=["employee_id"], errors="ignore")
    y = df["Quit_the_Company"]
    X = df.drop(columns=["Quit_the_Company"])

    for col in ["salary", "Departments"]:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
    return X, y


def load_data():
    """Load train and test data from BigQuery."""
    train_df = bq_client.query(f"SELECT * FROM `{PROJECT_ID}.{TRAIN_TABLE}`").to_dataframe()
    test_df = bq_client.query(f"SELECT * FROM `{PROJECT_ID}.{TEST_TABLE}`").to_dataframe()
    return train_df, test_df


def save_model(model, accuracy: float):
    """Save model to GCS with timestamp."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"churn_model_{ts}_{round(accuracy*100,2)}.pkl"

    # Save locally
    joblib.dump(model, filename)

    # Upload to GCS
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"{MODEL_PATH}{filename}")
    blob.upload_from_filename(filename)

    logging.info(f"Model saved to gs://{BUCKET_NAME}/{MODEL_PATH}{filename}")
    return filename


def load_latest_model():
    """Load the latest model from GCS for comparison."""
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(bucket.list_blobs(prefix=MODEL_PATH))

    if not blobs:
        logging.warning("No previous models found in GCS.")
        return None

    # Pick latest by name
    latest_blob = sorted(blobs, key=lambda b: b.name, reverse=True)[0]
    latest_blob.download_to_filename("latest_model.pkl")
    return joblib.load("latest_model.pkl")


def train_churn_model():
    """Train new model and compare with previous one."""
    train_df, test_df = load_data()

    X_train, y_train = preprocess(train_df)
    X_test, y_test = preprocess(test_df)

    model = XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logging.info(f"New model accuracy: {acc:.4f}")

    # Drift check: compare with latest model
    prev_model = load_latest_model()
    if prev_model:
        prev_pred = prev_model.predict(X_test)
        prev_acc = accuracy_score(y_test, prev_pred)
        logging.info(f"Previous model accuracy: {prev_acc:.4f}")

        if acc < prev_acc:
            logging.warning("New model underperforms previous model. Keeping old model.")
            return

    # Save only if better (or first model)
    save_model(model, acc)


def explain_model():
    """Generate SHAP feature importance for business users."""
    train_df, _ = load_data()
    X_train, y_train = preprocess(train_df)

    model = load_latest_model()
    if not model:
        logging.warning("No model available for SHAP explanations.")
        return

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)

    shap.summary_plot(shap_values, X_train, show=False)  # disable auto-display
    logging.info("SHAP feature importance generated.")
