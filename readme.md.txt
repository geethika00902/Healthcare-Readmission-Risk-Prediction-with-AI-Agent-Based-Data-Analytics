# MediRisk AI — Patient Readmission Risk Prediction

An end-to-end healthcare data pipeline that predicts 
patient readmission risk using machine learning and 
provides a GenAI-powered natural language query interface.

## Tech Stack
- **ETL** — Polars
- **Database** — PostgreSQL
- **ML** — XGBoost (94% accuracy)
- **API** — FastAPI
- **BI** — Power BI
- **AI Agent** — Groq Llama3
- **UI** — Streamlit

## Project Structure
```
health/
├── data/          # Raw and curated datasets
├── etl/           # Polars cleaning scripts
├── ml/            # Model training and prediction
├── api/           # FastAPI prediction service
├── ai_agent/      # Streamlit AI assistant
└── run_pipeline.py
```

## How to Run
1. Start PostgreSQL
2. Run `python run_pipeline.py`
3. Open `http://localhost:8501`