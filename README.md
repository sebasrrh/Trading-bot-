# AI Trading Application

This is a production-ready AI-powered stock market analysis and prediction application.
It features a FastAPI backend that fetches data and trains an XGBoost model, and a Streamlit frontend for interactive visualization.

## Prerequisites
- Python 3.9+
- A free API key from [NewsAPI](https://newsapi.org) (optional for sentiment, but recommended).

## Setup Instructions

### 1. Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

Create a `.env` file in the `backend/` directory:
```
NEWS_API_KEY=your_newsapi_key_here
```

Run the backend:
```bash
uvicorn main:app --reload --port 8000
```

### 2. Frontend Setup
Open a new terminal window.
```bash
cd frontend
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

Run the frontend dashboard:
```bash
streamlit run app.py
```

The frontend will be available at `http://localhost:8501`.
