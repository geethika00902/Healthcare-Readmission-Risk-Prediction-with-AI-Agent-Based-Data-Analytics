from fastapi import FastAPI
import joblib
import numpy as np
import psycopg2

app = FastAPI(title="Healthcare Readmission Risk API")

# Load ML model
model = joblib.load("../ml/models/xgboost_model.pkl")


@app.get("/")
def home():
    return {"message": "Healthcare Readmission Risk API Running"}


@app.post("/predict")
def predict(
    age: float,
    gender: str,
    length_of_stay: float,
    icu_los: float,
    num_diagnoses: int,
    num_medications: int
):

    # Encode gender
    gender_encoded = 1 if gender.upper() == "M" else 0

    features = np.array([[
        age,
        gender_encoded,
        length_of_stay,
        icu_los,
        num_diagnoses,
        num_medications
    ]])

    probability = model.predict_proba(features)[0][1]

    if probability > 0.7:
        risk = "High"
    elif probability > 0.4:
        risk = "Medium"
    else:
        risk = "Low"
    print("Saving prediction to database...")
    # Connect to PostgreSQL
    conn = psycopg2.connect(
        dbname="healthcare_db",
        user="postgres",
        password="newpass123",
        host="localhost",
        port="8080"
    )

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO patient_predictions
        (age, gender, length_of_stay, icu_los,
         num_diagnoses, num_medications,
         probability, risk_level)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        age,
        gender,
        length_of_stay,
        icu_los,
        num_diagnoses,
        num_medications,
        float(probability),
        risk
    ))

    conn.commit()
    cursor.close()
    conn.close()

    return {
        "readmission_probability": round(float(probability), 4),
        "risk_level": risk
    }