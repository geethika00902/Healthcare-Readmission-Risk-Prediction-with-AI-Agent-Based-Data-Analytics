"""
run_pipeline.py
================
Healthcare Readmission Risk — Master Pipeline
Place this file in:  Desktop/health/

Run with:
    python run_pipeline.py
"""

import subprocess
import sys
import os
import time

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))   # health/
ETL    = os.path.join(BASE, "etl")
ML     = os.path.join(BASE, "ml")
API    = os.path.join(BASE, "api")
AGENT  = os.path.join(BASE, "ai_agent")
PYTHON = sys.executable

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def banner(step, title):
    print("\n" + "═" * 55)
    print(f"  STEP {step}  →  {title}")
    print("═" * 55)

def run(script_path, label):
    """
    Run a python script with BASE (health/) as the working directory.
    This ensures relative paths like 'data/raw/PATIENTS.csv' always resolve
    correctly regardless of where the user calls python from.
    """
    print(f"\n▶  Running {label} ...")
    result = subprocess.run(
        [PYTHON, script_path],
        cwd=BASE,   # ← always health/ so all relative paths work
    )
    if result.returncode != 0:
        print(f"\n✗  {label} failed (exit code {result.returncode})")
        print("   Fix the error above before continuing.")
        sys.exit(1)
    print(f"✓  {label} completed successfully.")

def start_background(cmd, cwd, label):
    """Start a service in the background (non-blocking)."""
    print(f"\n▶  Starting {label} in background ...")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)
    print(f"✓  {label} running  (PID {proc.pid})")
    return proc

# ─────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────
def main():
    print("\n" + "█" * 55)
    print("  Healthcare Readmission Risk — Full Pipeline")
    print("█" * 55)

    # Step 1: ETL
    banner(1, "ETL — Clean raw MIMIC CSVs")
    run(os.path.join(ETL, "run_etl.py"), "run_etl")

    # Step 2: Synthetic data
    banner(2, "Augment — Generate synthetic data to 3000 rows")
    run(os.path.join(ML, "generate_synthetic_data.py"), "generate_synthetic_data")

    # Step 3: Train models
    banner(3, "ML — Train & compare LR / RF / XGBoost")
    run(os.path.join(ML, "train_and_compare_models.py"), "train_and_compare_models")

    # Step 4: Batch predict → DB
    banner(4, "Predict — Batch predict & insert into PostgreSQL")
    run(os.path.join(ML, "batch_predict_to_db.py"), "batch_predict_to_db")

    # Step 5: Start FastAPI
    banner(5, "API — Start FastAPI prediction service")
    api_proc = start_background(
        ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=API,
        label="FastAPI  →  http://127.0.0.1:8000",
    )

    # Step 6: Start Streamlit
    banner(6, "UI — Start Streamlit AI Assistant")
    print("\n▶  Launching Streamlit (browser will open automatically) ...")
    print("   Press  Ctrl + C  to stop everything.\n")

    try:
        subprocess.run(
            ["streamlit", "run", "app.py"],
            cwd=AGENT,
        )
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nShutting down FastAPI ...")
        api_proc.terminate()
        print("✓  All services stopped. Goodbye.")

if __name__ == "__main__":
    main()