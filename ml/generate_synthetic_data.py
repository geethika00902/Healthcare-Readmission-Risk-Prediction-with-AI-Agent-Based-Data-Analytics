import pandas as pd
import numpy as np
import os

REAL_DATA_PATH = "data/curated/final_dataset.csv"
OUTPUT_PATH = "data/curated/final_dataset_augmented.csv"

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_synthetic_data(target_rows=3000, readmit_rate=0.25):

    real_df = pd.read_csv(REAL_DATA_PATH)

    real_rows = len(real_df)
    synthetic_needed = target_rows - real_rows

    print("Real rows:", real_rows)
    print("Generating synthetic rows:", synthetic_needed)

    np.random.seed(42)

    synthetic_data = pd.DataFrame({
        "patient_id": np.random.randint(50000, 90000, synthetic_needed),
        "admission_id": np.random.randint(500000, 900000, synthetic_needed),
        "age": np.random.normal(60, 15, synthetic_needed).clip(18, 95),
        "gender": np.random.choice(["M", "F"], synthetic_needed),
        "length_of_stay": np.random.gamma(2, 3, synthetic_needed),
        "icu_los": np.random.gamma(1.5, 2, synthetic_needed),
        "num_diagnoses": np.random.poisson(10, synthetic_needed),
        "num_medications": np.random.poisson(80, synthetic_needed)
    })

    # Create realistic risk-based readmission probability
    risk_score = (
        0.03 * synthetic_data["age"] +
        0.6 * (synthetic_data["length_of_stay"] > 7).astype(int) +
        0.5 * (synthetic_data["num_diagnoses"] > 12).astype(int) +
        0.5 * (synthetic_data["icu_los"] > 3).astype(int)
    )

    probabilities = sigmoid(risk_score - risk_score.mean())

    synthetic_data["readmitted"] = (
        probabilities > np.quantile(probabilities, 1 - readmit_rate)
    ).astype(int)

    # Combine with real data
    final_df = pd.concat([real_df, synthetic_data], ignore_index=True)

    final_df.to_csv(OUTPUT_PATH, index=False)

    print("Final dataset shape:", final_df.shape)
    print("Readmission distribution:")
    print(final_df["readmitted"].value_counts())

if __name__ == "__main__":
    generate_synthetic_data()