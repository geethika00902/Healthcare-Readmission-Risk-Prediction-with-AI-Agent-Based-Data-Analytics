import pandas as pd
import joblib
import psycopg2

# Load trained model
model = joblib.load("ml/models/xgboost_model.pkl")

# Load dataset
df = pd.read_csv("data/curated/final_dataset_augmented.csv")

# Encode gender
df["gender"] = df["gender"].map({"M": 1, "F": 0})

# Features for prediction
X = df[
    [
        "age",
        "gender",
        "length_of_stay",
        "icu_los",
        "num_diagnoses",
        "num_medications",
    ]
]

# Predict probabilities
probs = model.predict_proba(X)[:, 1]

df["readmission_probability"] = probs

# Risk classification
df["risk_level"] = df["readmission_probability"].apply(
    lambda x: "High" if x > 0.7 else "Medium" if x > 0.4 else "Low"
)

# Connect to PostgreSQL
conn = psycopg2.connect(
    dbname="healthcare_db",
    user="postgres",
    password="newpass123",
    host="localhost",
    port="8080",
)

cur = conn.cursor()
cur.execute("TRUNCATE TABLE patient_predictions;")
# Insert predictions
for _, row in df.iterrows():

    cur.execute(
        """
        INSERT INTO patient_predictions
        (age, gender, length_of_stay, icu_los, num_diagnoses, num_medications, probability, risk_level)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
        """,
        (
            float(row["age"]),
            int(row["gender"]),
            float(row["length_of_stay"]),
            float(row["icu_los"]),
            int(row["num_diagnoses"]),
            int(row["num_medications"]),
            float(row["readmission_probability"]),
            row["risk_level"],
        ),
    )

conn.commit()
cur.close()
conn.close()

print("All predictions inserted successfully")