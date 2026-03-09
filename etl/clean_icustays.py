import polars as pl
import os

RAW_PATH = "data/raw/ICUSTAYS.csv"
CURATED_PATH = "data/curated/icustays_clean.csv"

def clean_icustays():
    os.makedirs("data/curated", exist_ok=True)

    df = pl.read_csv(RAW_PATH)

    df = df.rename({
        "hadm_id": "admission_id",
        "icustay_id": "icu_stay_id",
        "los": "icu_los"
    })

    df = df.select([
        "admission_id",
        "icu_stay_id",
        "icu_los"
    ])

    df.write_csv(CURATED_PATH)
    print("ICU stays cleaned successfully")

if __name__ == "__main__":
    clean_icustays()