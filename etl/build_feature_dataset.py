import polars as pl
import os

CURATED_PATH = "data/curated/"
OUTPUT_PATH = "data/curated/final_dataset.csv"

def build_final_dataset():
    os.makedirs("data/curated", exist_ok=True)

    # Load cleaned tables
    patients = pl.read_csv(CURATED_PATH + "patients_clean.csv")
    admissions = pl.read_csv(CURATED_PATH + "admissions_clean.csv")
    icu = pl.read_csv(CURATED_PATH + "icustays_clean.csv")
    diagnoses = pl.read_csv(CURATED_PATH + "diagnoses_agg.csv")
    prescriptions = pl.read_csv(CURATED_PATH + "prescriptions_agg.csv")

    # Convert datetime columns again (important after CSV load)
    admissions = admissions.with_columns([
        pl.col("admit_time").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M", strict=False),
        pl.col("discharge_time").str.strptime(pl.Datetime, "%d-%m-%Y %H:%M", strict=False)
    ])

    patients = patients.with_columns(
        pl.col("date_of_birth").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
    )

    # Merge all tables
    df = admissions.join(patients, on="patient_id", how="left")
    df = df.join(icu, on="admission_id", how="left")
    df = df.join(diagnoses, on="admission_id", how="left")
    df = df.join(prescriptions, on="admission_id", how="left")

    # Fill missing numeric values
    df = df.with_columns([
        pl.col("icu_los").fill_null(0),
        pl.col("num_diagnoses").fill_null(0),
        pl.col("num_medications").fill_null(0)
    ])

    # Calculate age at admission
    df = df.with_columns(
        (
            (pl.col("admit_time") - pl.col("date_of_birth"))
            .dt.total_seconds() / (60 * 60 * 24 * 365)
        ).alias("age")
    )

    # Sort by patient and admission time
    df = df.sort(["patient_id", "admit_time"])

    # Next admission time per patient
    df = df.with_columns(
        pl.col("admit_time")
        .shift(-1)
        .over("patient_id")
        .alias("next_admit_time")
    )

    # Days to next admission
    df = df.with_columns(
        (
            (pl.col("next_admit_time") - pl.col("discharge_time"))
            .dt.total_seconds() / 86400
        ).alias("days_to_next_admit")
    )

    # Readmission flag (within 30 days)
    df = df.with_columns(
        pl.when(
            (pl.col("days_to_next_admit") <= 30) &
            (pl.col("days_to_next_admit") > 0)
        )
        .then(1)
        .otherwise(0)
        .alias("readmitted")
    )

    # Select final ML columns
    df = df.select([
        "patient_id",
        "admission_id",
        "age",
        "gender",
        "length_of_stay",
        "icu_los",
        "num_diagnoses",
        "num_medications",
        "readmitted"
    ])

    df.write_csv(OUTPUT_PATH)

    print("Final ML dataset created successfully")
    print("Final dataset shape:", df.shape)

if __name__ == "__main__":
    build_final_dataset()