import pandas as pd
from sqlalchemy import create_engine
from langchain.tools import tool
import requests

# PostgreSQL connection
engine = create_engine(
    "postgresql://postgres:newpass123@localhost:5432/healthcare_db"
)


@tool
def query_database(sql_query: str):
    """
    Execute SQL query on healthcare database and return results.
    """

    try:
        df = pd.read_sql(sql_query, engine)

        if df.empty:
            return "No records found."

        return df.to_string()

    except Exception as e:
        return f"Database error: {str(e)}"


@tool
def predict_patient_risk(data: str):
    """
    Predict readmission risk.

    Example input:
    age=70 icu_los=3 num_diagnoses=5 num_medications=10
    """

    try:

        values = {}

        parts = data.split()

        for part in parts:
            key, value = part.split("=")

            value = value.replace(",", "").strip()

            values[key] = float(value)

        payload = {
            "age": values.get("age", 0),
            "icu_los": values.get("icu_los", 0),
            "num_diagnoses": int(values.get("num_diagnoses", 0)),
            "num_medications": int(values.get("num_medications", 0))
        }

        url = "http://127.0.0.1:8000/predict"

        response = requests.post(url, json=payload)

        return response.json()

    except Exception as e:
        return f"Prediction error: {str(e)}"