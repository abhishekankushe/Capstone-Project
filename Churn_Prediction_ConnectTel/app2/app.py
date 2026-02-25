from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

# 1. Initialize FastAPI
app = FastAPI(title="ConnectTel Churn Prediction API")

# Allow the frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your website URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the trained model
# Ensure the path matches where your .pkl is saved
model = joblib.load("models/connecttel_churn_pipeline.pkl")

# 3. Define the Input Data Schema using Pydantic
# We set default values for fields so the user doesn't have to fill out 20 form fields
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

# 4. Create the Prediction Endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):
    # Convert input data to a Pandas DataFrame
    input_dict = customer.dict()
    df = pd.DataFrame([input_dict])
    
    # --- Apply the exact Feature Engineering from training ---
    
    # 1. Tenure Bucket
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"], 
        bins=[0, 12, 24, 48, 72, 120], 
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr", "6yr+"]
    )
    # Fill NaN just in case tenure is 0
    df["tenure_bucket"] = df["tenure_bucket"].fillna("0-1yr")
    
    # 2. Engineered Features
    df["complaint_intensity"] = df["num_complaints_3m"] + df["num_complaints_12m"]
    df["engagement_score"] = df["app_logins_30d"] + df["selfcare_transactions_30d"]
    
    # Note: Using static thresholds for a single API request instead of quantiles
    df["bill_shock"] = np.where(df["monthly_charges"] > 80.0, 1, 0) 
    df["high_value_flag"] = np.where(df["arpu"] > 900.0, 1, 0)
    
    # Make Prediction
    probability = model.predict_proba(df)[0][1]
    
    # Using our custom optimized threshold of 0.35
    prediction = 1 if probability >= 0.35 else 0
    risk_level = "High Risk" if prediction == 1 else "Low Risk"
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": float(probability),
        "risk_level": risk_level
    }