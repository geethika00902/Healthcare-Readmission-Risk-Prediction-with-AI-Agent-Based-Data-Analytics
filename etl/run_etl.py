from clean_patients import clean_patients
from clean_admissions import clean_admissions
from clean_icustays import clean_icustays
from clean_diagnoses import clean_diagnoses
from clean_prescriptions import clean_prescriptions

def run_all():
    clean_patients()
    clean_admissions()
    clean_icustays()
    clean_diagnoses()
    clean_prescriptions()
    print("All tables cleaned successfully")

if __name__ == "__main__":
    run_all()