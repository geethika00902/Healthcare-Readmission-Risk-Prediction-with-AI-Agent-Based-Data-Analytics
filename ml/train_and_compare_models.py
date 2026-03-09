import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


DATA_PATH = "data/curated/final_dataset_augmented.csv"
MODEL_DIR = "ml/models"


def evaluate_model(name, model, X_test, y_test):

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc  = round(accuracy_score(y_test, y_pred), 4)
    prec = round(precision_score(y_test, y_pred), 4)
    rec  = round(recall_score(y_test, y_pred), 4)
    f1   = round(f1_score(y_test, y_pred), 4)
    auc  = round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else "N/A"

    print(f"\n===== {name} =====")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print("AUC-ROC  :", auc)

    return {"model": name, "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "auc": auc}


def train_models():

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("\nLoading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)
    print("\nClass distribution:")
    print(df["readmitted"].value_counts())

    # Encode gender — map M/F to 1/0, anything else becomes NaN then filled with 0
    df["gender"] = df["gender"].map({"M": 1, "F": 0})

    # ── FIX: drop rows with NaN values so all models train cleanly ──
    before = len(df)
    df = df.dropna()
    after  = len(df)
    if before != after:
        print(f"\nDropped {before - after} rows with NaN values ({after} rows remaining).")

    # Features and target
    X = df.drop(columns=["patient_id", "admission_id", "readmitted"])
    y = df["readmitted"]

    print("\nFeatures used for training:", list(X.columns))

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Feature Scaling ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    print("\nScaler saved.")

    results = []

    # ── 1. Logistic Regression ──
    print("\nTraining Logistic Regression ...")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    results.append(evaluate_model("Logistic Regression", lr, X_test_scaled, y_test))

    # ── 2. Random Forest ──
    print("\nTraining Random Forest ...")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    results.append(evaluate_model("Random Forest", rf, X_test, y_test))

    # ── 3. XGBoost ──
    print("\nTraining XGBoost ...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    xgb.fit(X_train, y_train)
    results.append(evaluate_model("XGBoost", xgb, X_test, y_test))

    # ── Summary table ──
    print("\n" + "═" * 55)
    print("  MODEL COMPARISON SUMMARY")
    print("═" * 55)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    # Save results so the Streamlit app can display real numbers
    results_df.to_csv(f"{MODEL_DIR}/model_results.csv", index=False)
    print(f"\nResults saved → {MODEL_DIR}/model_results.csv")

    # ── Save best model (XGBoost) ──
    model_path = f"{MODEL_DIR}/xgboost_model.pkl"
    joblib.dump(xgb, model_path)
    print(f"XGBoost model saved → {model_path}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    train_models()