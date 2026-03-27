from fastapi import FastAPI, BackgroundTasks, HTTPException
from contextlib import asynccontextmanager
from apscheduler.schedulers.background import BackgroundScheduler
import uvicorn
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from core.config import settings
from data.ingestion import get_all_data
from features.engineering import prepare_features, prepare_features_htf
from models.predictor import train_model, predict_next_signal, get_raw_prediction
from models.signal_engine import run_signal_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state to serve the scheduled updates
app_state = {}


def perform_analysis(ticker: str):
    """Fetches data, runs ML pipeline and signal engine, returns state dict for a ticker."""
    logger.info(f"Running full analysis for {ticker} at {datetime.now()}")

    df_5m, df_1d, df_htf, sentiment, fundamentals = get_all_data(ticker)

    if df_5m.empty:
        raise ValueError(f"No market data found for {ticker}")

    df_5m_features = prepare_features(df_5m, sentiment)

    if df_5m_features.empty:
        raise ValueError(f"Not enough data to build features for {ticker}")

    df_1d_features = prepare_features_htf(df_1d)
    df_htf_features = prepare_features_htf(df_htf)

    # Train ML model on primary 5m data
    train_model(df_5m_features)

    # Get raw unthresholded probability from the trained model
    raw_confidence, walk_forward_accuracy = get_raw_prediction(df_5m_features)

    # Run the full institutional signal engine
    engine_result = run_signal_engine(
        df_5m_features,
        df_1d_features,
        df_htf_features,
        raw_confidence,
        walk_forward_accuracy,
        sentiment,
        fundamentals,
    )

    # Clean DataFrame for JSON serialization
    records = df_5m_features.reset_index()
    if 'Datetime' in records.columns:
        records['Datetime'] = records['Datetime'].astype(str)
    elif 'Date' in records.columns:
        records['Date'] = records['Date'].astype(str)

    records = records.replace({np.nan: None})

    state = {
        "ticker": ticker.upper(),
        "latest_data": records.tail(150).to_dict(orient="records"),
        "last_updated": datetime.now().isoformat(),
        "sentiment": sentiment,
        "fundamentals": fundamentals,
        **engine_result,
        "explanation": engine_result["reasoning"],  # legacy alias
    }
    return state


def scheduled_job():
    """Runs the scheduled update for the globally tracked ticker."""
    try:
        global app_state
        state = perform_analysis(settings.TICKER)
        app_state.update(state)
        logger.info(f"Scheduled update for {settings.TICKER} complete.")
    except Exception as e:
        logger.error(f"Error in scheduled task: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing application and running first baseline pull...")
    scheduled_job()

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        scheduled_job,
        'interval',
        minutes=settings.PREDICTION_INTERVAL_MINUTES,
        id='update_job'
    )
    scheduler.start()

    yield
    scheduler.shutdown()


app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION, lifespan=lifespan)


@app.get("/")
def read_root():
    return {"status": "ok", "app": settings.PROJECT_NAME}


@app.get("/api/state")
def get_state():
    return app_state


@app.get("/api/analyze")
def analyze_endpoint(ticker: str):
    """On-demand full analysis for any ticker."""
    try:
        state = perform_analysis(ticker)
        settings.TICKER = ticker.upper()
        global app_state
        app_state.update(state)
        return state
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error during analysis")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
