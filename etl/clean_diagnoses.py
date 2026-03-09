import polars as pl
import os

RAW_PATH = "data/raw/DIAGNOSES_ICD.csv"
CURATED_PATH = "data/curated/diagnoses_agg.csv"

def clean_diagnoses():
    os.makedirs("data/curated", exist_ok=True)

    df = pl.read_csv(RAW_PATH)

    df = df.rename({
        "hadm_id": "admission_id",
        "icd9_code": "icd_code"
    })

    df = (
        df.group_by("admission_id")
        .count()
        .select([
            "admission_id",
            pl.col("count").alias("num_diagnoses")
        ])
    )

    df.write_csv(CURATED_PATH)
    print("Diagnoses cleaned successfully")

if __name__ == "__main__":
    clean_diagnoses()