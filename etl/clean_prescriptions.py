import polars as pl
import os

RAW_PATH = "data/raw/PRESCRIPTIONS.csv"
CURATED_PATH = "data/curated/prescriptions_agg.csv"

def clean_prescriptions():
    os.makedirs("data/curated", exist_ok=True)

    df = pl.read_csv(
        RAW_PATH,
        infer_schema_length=None,
        ignore_errors=True
    )

    print("Columns detected:", df.columns)

    df = df.rename({
        "hadm_id": "admission_id",
        "drug": "drug_name"
    })

    df = (
        df.group_by("admission_id")
        .count()
        .select([
            "admission_id",
            pl.col("count").alias("num_medications")
        ])
    )

    df.write_csv(CURATED_PATH)
    print("Prescriptions cleaned successfully")

if __name__ == "__main__":
    clean_prescriptions()