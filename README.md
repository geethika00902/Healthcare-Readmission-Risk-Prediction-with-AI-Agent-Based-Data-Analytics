# AI-Powered Healthcare Readmission Risk Prediction & Analytics System

This project is an end-to-end healthcare analytics platform that predicts the probability of hospital patient readmission within 30 days using machine learning and provides an interactive dashboard with an AI-powered data assistant.

The system combines machine learning, APIs, databases, and large language models (LLMs) to help healthcare professionals analyze patient risk and make better decisions.

Hospital readmissions are a major challenge in healthcare systems. Patients who are discharged may return within a short period due to complications or incomplete treatment. This project predicts the risk of patient readmission using clinical and hospital stay information. The predictions are stored in a database and can be analyzed through an interactive dashboard and an AI assistant that answers questions in natural language.

Example question the system can answer:

How many patients are classified as high risk?

The AI agent converts the question into SQL and retrieves the answer from the database automatically.

Key Features

- Machine learning model for predicting readmission probability  
- Automated data processing and feature engineering pipeline  
- Synthetic data generation to increase training data size  
- FastAPI prediction API for real-time patient risk prediction  
- PostgreSQL database to store predictions  
- Streamlit interactive dashboard for analytics and visualization  
- AI agent powered by LangChain and Ollama  
- Natural language database querying using a large language model  

System Architecture

Raw Healthcare Dataset  
↓  
Data Cleaning & Feature Engineering  
↓  
Synthetic Data Generation  
↓  
Machine Learning Model Training (XGBoost)  
↓  
Prediction API (FastAPI)  
↓  
Predictions Stored in PostgreSQL  
↓  
Streamlit Dashboard  
↓  
AI Agent (LangChain + Ollama LLM)

Tech Stack

Programming Language  
Python

Machine Learning  
XGBoost  
Scikit-learn  
Pandas  
NumPy

Backend API  
FastAPI

Database  
PostgreSQL  
SQLAlchemy

AI Agent  
LangChain  
Ollama  
Phi-3 LLM

Dashboard & Visualization  
Streamlit  
Matplotlib

Machine Learning Model

The project uses XGBoost, a gradient boosting decision tree algorithm, to predict the probability of patient readmission.

Input features used for prediction include:

Age  
Gender  
Length of hospital stay  
ICU length of stay  
Number of diagnoses  
Number of medications  

Model output includes:

Readmission probability  
Risk classification (Low / Medium / High)

AI Agent

The system includes an AI-powered assistant that allows users to query patient prediction data using natural language.

Example queries:

How many high risk patients are there?

What is the average ICU stay for high risk patients?

The AI agent performs the following steps:

1. Interprets the user’s natural language question  
2. Generates the appropriate SQL query  
3. Executes the query on the PostgreSQL database  
4. Returns the result in a readable format  

Project Structure

project-root

data  
- raw  
- curated  

ml  
- models  

scripts  
- clean_patients.py  
- build_final_dataset.py  
- generate_synthetic_data.py  
- train_models.py  

api  
- main.py  

app.py  
requirements.txt  
README.md

Installation

Clone the repository

git clone https://github.com/yourusername/your-repo-name.git

Move into the project directory

cd your-repo-name

Install dependencies

pip install -r requirements.txt

Running the System

Start the Ollama server

ollama serve

Start the prediction API

uvicorn main:app --reload

Run the Streamlit dashboard

streamlit run app.py

Open the dashboard in your browser to interact with the system.

Example Dashboard Features

Patient risk distribution  
Age distribution visualization  
ICU stay analysis  
AI assistant for querying prediction data  
Real-time patient readmission risk prediction

Future Improvements

Integration with hospital Electronic Health Record (EHR) systems  
Real-time patient monitoring and alerts  
Explainable AI for model predictions  
Multi-hospital analytics platform

Author

Developed as a healthcare analytics and machine learning project demonstrating an end-to-end AI system integrating machine learning models, APIs, databases, dashboards, and AI agents.
