import polars as pl
import os

RAW_PATH = "data/raw/ADMISSIONS.csv"
CURATED_PATH = "data/curated/admissions_clean.csv"

def clean_admissions():
    os.makedirs("data/curated", exist_ok=True)

    df = pl.read_csv(RAW_PATH)

    print("Initial rows:", df.shape)

    # Rename columns (based on your actual schema)
    df = df.rename({
        "subject_id": "patient_id",
        "hadm_id": "admission_id",
        "admittime": "admit_time",
        "dischtime": "discharge_time",
        "diagnosis": "primary_diagnosis",
        "hospital_expire_flag": "hospital_expired"
    })

    # Select required columns
    df = df.select([
        "patient_id",
        "admission_id",
        "admit_time",
        "discharge_time",
        "admission_type",
        "primary_diagnosis",
        "hospital_expired"
    ])

    # Convert datetime (CORRECT FORMAT FOR YOUR FILE)
    df = df.with_columns([
        pl.col("admit_time")
        .str.strptime(pl.Datetime, "%d-%m-%Y %H:%M", strict=False),

        pl.col("discharge_time")
        .str.strptime(pl.Datetime, "%d-%m-%Y %H:%M", strict=False)
    ])

    print("Null admit_time count:",
          df.select(pl.col("admit_time").null_count()))
    print("Null discharge_time count:",
          df.select(pl.col("discharge_time").null_count()))

    # Now create length of stay
    df = df.with_columns(
        (
            (pl.col("discharge_time") - pl.col("admit_time"))
            .dt.total_seconds() / 86400
        ).alias("length_of_stay")
    )

    df.write_csv(CURATED_PATH)

    print("Final rows after cleaning:", df.shape)
    print("Admissions cleaned successfully")

if __name__ == "__main__":
    clean_admissions()