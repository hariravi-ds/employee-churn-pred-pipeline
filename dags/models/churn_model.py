from google.cloud import bigquery, storage
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import shap, joblib, datetime, os

# Project config
PROJECT_ID = "robotic-facet-471416-s5"
DATASET = "my_employee_data"
MODEL_BUCKET = "my-ml-model-bucket"  # 👈 replace with your GCS bucket name
MODEL_FILENAME = "churn_model_latest.pkl"


# ----------------------------
# 1. TRAIN FUNCTION
# ----------------------------
def train_churn_model():
    client = bigquery.Client(project=PROJECT_ID)

    # --- Load training data ---
    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET}.tbl_hr_data`
    """
    df = client.query(query).to_dataframe()

    # --- Drop ignored column ---
    if "employee_id" in df.columns:
        df = df.drop(columns=["employee_id"])

    # --- Separate target ---
    if "Quit_the_Company" not in df.columns:
        raise KeyError("❌ Column 'Quit_the_Company' not found in training table")
    y = df["Quit_the_Company"]
    X = df.drop(columns=["Quit_the_Company"])

    # --- Encode categorical columns ---
    for col in ["salary", "Departments"]:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # --- Train model ---
    model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)
    model.fit(X, y)

    # --- Explainability (SHAP) ---
    explainer = shap.TreeExplainer(model)
    _ = explainer.shap_values(X.sample(min(200, len(X))))

    # --- Save locally ---
    model_path = f"/tmp/{MODEL_FILENAME}"
    joblib.dump(model, model_path)

    # --- Upload to GCS ---
    storage_client = storage.Client()
    bucket = storage_client.bucket(MODEL_BUCKET)
    blob = bucket.blob(MODEL_FILENAME)
    blob.upload_from_filename(model_path)

    print(f"✅ Model trained and saved to gs://{MODEL_BUCKET}/{MODEL_FILENAME}")


# ----------------------------
# 2. PREDICT FUNCTION
# ----------------------------
def predict_churn():
    client = bigquery.Client(project=PROJECT_ID)
    storage_client = storage.Client()

    # --- Download trained model from GCS ---
    model_path = f"/tmp/{MODEL_FILENAME}"
    blob = storage_client.bucket(MODEL_BUCKET).blob(MODEL_FILENAME)
    blob.download_to_filename(model_path)
    model = joblib.load(model_path)

    # --- Load scoring data ---
    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET}.tl_new_employee`
    """
    df = client.query(query).to_dataframe()

    # --- Drop ignored + label columns ---
    if "employee_id" in df.columns:
        df = df.drop(columns=["employee_id"])
    if "Quit_the_Company" in df.columns:
        df = df.drop(columns=["Quit_the_Company"])

    # --- Encode categorical ---
    X = df.copy()
    for col in ["salary", "Departments"]:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # --- Predict churn probabilities ---
    churn_probs = model.predict_proba(X)[:, 1]

    # --- Create results DataFrame ---
    df_predictions = pd.DataFrame({
        "scoring_date": datetime.date.today(),
        "churn_probability": churn_probs
    })

    if "user_id" in df.columns:  # preserve user_id if available
        df_predictions["user_id"] = df["user_id"]

    # --- Write back to BigQuery ---
    table_id = f"{PROJECT_ID}.{DATASET}.churn_predictions"
    client.load_table_from_dataframe(
        df_predictions, table_id, if_exists="replace"
    ).result()

    print(f"✅ Predictions written to {table_id}")

