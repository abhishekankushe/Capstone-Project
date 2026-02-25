from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI(title="ConnectTel Churn Prediction API")

# --- 1. SET UP YOUR CUSTOM API KEY ---
API_KEY = "prabhudev-connecttel-secret-key-2026"  
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(status_code=403, detail="Could not validate credentials. Invalid API Key.")

# --- 2. CORS MIDDLEWARE ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. LOAD THE MODEL ---
model = joblib.load("models/connecttel_churn_pipeline.pkl")

# --- 4. DEFINE THE DATA SCHEMA ---
class CustomerData(BaseModel):
    tenure_months: int
    monthly_charges: float
    total_charges: float
    arpu: float
    num_complaints_3m: int = 0
    num_complaints_12m: int = 0
    app_logins_30d: int = 5
    selfcare_transactions_30d: int = 2
    network_issues_3m: int = 0
    last_complaint_resolution_days: int = 0
    late_payment_flag_3m: int = 0
    avg_payment_delay_days: int = 0
    service_rating_last_6m: float = 4.0
    nps_score: int = 8
    received_competitor_offer_flag: int = 0
    gender: str = "Male"
    region_circle: str = "Metro"
    connection_type: str = "5G"
    plan_type: str = "Postpaid"
    contract_type: str = "Month-to-Month"
    base_plan_category: str = "Medium"
    segment_value: str = "Medium"

# --- 5. SECURED PREDICTION ENDPOINT ---
# The endpoint is now locked and requires the API key to run
@app.post("/predict")
def predict_churn(customer: CustomerData, api_key: str = Depends(get_api_key)):
    input_dict = customer.dict()
    df = pd.DataFrame([input_dict])
    
    # Feature Engineering
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"], 
        bins=[0, 12, 24, 48, 72, 120], 
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr", "6yr+"]
    ).fillna("0-1yr")
    
    df["complaint_intensity"] = df["num_complaints_3m"] + df["num_complaints_12m"]
    df["engagement_score"] = df["app_logins_30d"] + df["selfcare_transactions_30d"]
    df["bill_shock"] = np.where(df["monthly_charges"] > 80.0, 1, 0) 
    df["high_value_flag"] = np.where(df["arpu"] > 900.0, 1, 0)
    
    # Model Prediction
    probability = model.predict_proba(df)[0][1]
    
    prediction = 1 if probability >= 0.45 else 0
    risk_level = "High Risk" if prediction == 1 else "Low Risk"
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability),
        "risk_level": risk_level
    }