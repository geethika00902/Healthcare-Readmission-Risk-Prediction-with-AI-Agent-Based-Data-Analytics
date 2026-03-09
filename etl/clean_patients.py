import polars as pl
import os

RAW_PATH = "data/raw/PATIENTS.csv"
CURATED_PATH = "data/curated/patients_clean.csv"

def clean_patients():
    os.makedirs("data/curated", exist_ok=True)

    df = pl.read_csv(RAW_PATH)

    df = df.rename({
        "subject_id": "patient_id",
        "dob": "date_of_birth",
        "expire_flag": "expired"
    })

    df = df.select([
        "patient_id",
        "gender",
        "date_of_birth",
        "expired"
    ])

    df = df.with_columns(
        pl.col("date_of_birth")
        .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
    )

    df = df.unique()

    df.write_csv(CURATED_PATH)
    print("Patients cleaned successfully")

if __name__ == "__main__":
    clean_patients()