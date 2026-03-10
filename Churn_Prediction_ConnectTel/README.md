
# ConnectTel Customer Churn Prediction Engine

![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)
[![Live Demo](https://img.shields.io/badge/Live_Demo-Check_it_Out-brightgreen?style=for-the-badge&logo=rocket)](https://churnprediction-analysis.vercel.app/)

![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-tree?logo=xgboost&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Preprocessing-orange?logo=scikit-learn&logoColor=white)
![HTML/CSS](https://img.shields.io/badge/Frontend-HTML%2FCSS%2FJS-38B2AC?logo=html5&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end Machine Learning solution to predict customer churn for ConnectTel. It features an optimized **XGBoost** pipeline and a production-ready **FastAPI** service for real-time risk assessment and batch scoring.

---

## 🚀 What This Project Demonstrates

- **End-to-end ML pipeline:** From raw data ingestion to feature engineering, model training, and API deployment.
- **Class imbalance handling:** Utilizing XGBoost's `scale_pos_weight` to prioritize minority class detection.
- **Threshold-based decisioning:** Custom 0.42 decision threshold optimized for maximum revenue preservation (76% Recall).
- **Clean modular architecture:** Separation of ML scripts, API backend, and frontend assets.
- **Production API:** Secured single and batch prediction endpoints using FastAPI.

---

## 🧱 Project Structure

```text
Churn_Prediction_ConnectTel/
├── app/
│   ├── main.py                        # FastAPI Backend (Endpoints: /predict, /predict_batch)
│   └── frontend/
│       ├── index.html                 # Frontend Dashboard
│       └── scripts.js                 # Frontend logic + Chart.js
├── data/
│   ├── raw/
│   │   └── telecom_churn.csv          # Input dataset
│   ├── interim/
│   │   └── telecom_clean.csv          # Cleaned data after EDA
│   └── processed/
│       └── telecom_features.csv       # Final engineered feature set
├── models/
│   └── connecttel_churn_final.pkl     # Optimized XGBoost Pipeline (joblib)
├── notebooks/
│   ├── 01_EDA_Telecom_Churn.ipynb     # Exploratory Data Analysis
│   ├── 02_Feature_Engineering.ipynb   # Feature development logic
│   └── 03_Model_Training.ipynb        # Final XGBoost training & tuning
├── output/
│   └── plots/                         # EDA and Feature Importance charts
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation

```

---

## 📦 Run It Locally

1️⃣ **Clone & Install**

```bash
git clone [https://github.com/](https://github.com/)<your-username>/Churn_Prediction_ConnectTel.git
cd Churn_Prediction_ConnectTel

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

```

2️⃣ **Launch the API & Dashboard**

```bash
uvicorn app.main:app --reload

```

3️⃣ **Access the Application**

* **Live Dashboard:** Open `http://127.0.0.1:8000/static/index.html` in your browser.
* **Interactive API Docs:** `http://127.0.0.1:8000/docs`

---

## 🖥 API & Dashboard Features

* **Live Scoring**
* Enter customer attributes via the UI or API.
* Get real-time fraud probability and risk categorization (High/Low).


* **Batch Processing**
* Upload entire CSV datasets to the `/predict_batch` endpoint.
* Receive aggregate Churn Metrics (Total High-Risk vs. Low-Risk customers).


* **Secured Endpoints**
* All API routes are protected by an `X-API-Key` header authentication.



---

## 🧠 Feature Engineering Highlights

* **Technical Friction Index:** `dropped_call_rate * (complaints_3m + 1)` — Targets customers frustrated by service quality.
* **Tenure Velocity:** `tenure_months / (monthly_charges + 1)` — Balances loyalty against billing amounts.
* **Bill Shock Signal:** Flags customers experiencing sudden price hikes (above the 75th percentile).
* **Engagement Score:** Combined digital footprint (`app_logins` + `selfcare_transactions`).

---

## 📊 Model Performance

* **Algorithm:** XGBoost Classifier
* **Imbalance Handling:** `scale_pos_weight` = 1.41
* **Recall (Churners):** **76%** (Successfully identifies 3 out of 4 departing customers)
* **AUC-ROC:** 0.6564
* **Decision Policy:** 0.42 threshold optimized to prioritize retention efforts over precision, maximizing saved revenue.

---

## 📌 Future Improvements

* Model explainability (SHAP values integration)
* Data drift monitoring
* Advanced hyperparameter tuning tracking with MLflow
* Containerization with Docker

---

## 📄 License

MIT

```

```