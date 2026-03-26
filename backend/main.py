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
from features.engineering import prepare_features
from models.predictor import train_model, predict_next_signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state to serve the scheduled updates
app_state = {}

def generate_explanation(df: pd.DataFrame, signal: str, confidence: float, sentiment: float) -> str:
    """Generates simple rules-based text explanations for the AI's signal."""
    if df.empty: return "No data available."
    last = df.iloc[-1]
    reasons = []
    
    if sentiment > 0.1: reasons.append("Bullish news sentiment")
    elif sentiment < -0.1: reasons.append("Bearish news sentiment")
    
    if 'RSI_14' in last and pd.notnull(last['RSI_14']):
        if last['RSI_14'] < 35: reasons.append(f"RSI Oversold ({last['RSI_14']:.1f})")
        elif last['RSI_14'] > 65: reasons.append(f"RSI Overbought ({last['RSI_14']:.1f})")
        
    if 'SMA_20' in last and 'SMA_50' in last and pd.notnull(last['SMA_20']) and pd.notnull(last['SMA_50']):
        if last['SMA_20'] > last['SMA_50']: reasons.append("Price momentum is accelerating (SMA20 > SMA50)")
        else: reasons.append("Price momentum is decelerating (SMA20 < SMA50)")
        
    if 'MACDh_12_26_9' in last and pd.notnull(last['MACDh_12_26_9']):
        if last['MACDh_12_26_9'] > 0: reasons.append("Positive MACD momentum")
        else: reasons.append("Negative MACD momentum")
        
    if not reasons:
        reasons.append("Model detected hidden complex multi-factor patterns")
        
    reason_str = ", ".join(reasons)
    return f"Based on: {reason_str}. The XGBoost-style HistGradient model analyzed this technical setup and converged on a {signal} outcome for the next interval."

def perform_analysis(ticker: str):
    """Fetches data, runs ML pipeline, and generates state dict for a ticker"""
    logger.info(f"Running full analysis for {ticker} at {datetime.now()}")
    
    df, sentiment, fundamentals = get_all_data(ticker)
    
    if df.empty:
        raise ValueError(f"No market data found for {ticker}")
        
    df_features = prepare_features(df, sentiment)
    
    if df_features.empty:
         raise ValueError(f"Not enough data to build features for {ticker}")
         
    # Train
    train_model(df_features)
    
    # Predict
    signal, confidence, accuracy = predict_next_signal(df_features)
    
    explanation = generate_explanation(df_features, signal, confidence, sentiment)
    
    # Clean DataFrame for JSON
    records = df_features.reset_index()
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
        "signal": signal,
        "confidence": confidence,
        "model_accuracy": accuracy,
        "explanation": explanation
    }
    return state

def scheduled_job():
    """Runs the scheduled update for the globally tracked ticker"""
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
    """On-demand full analysis for any ticker"""
    try:
        state = perform_analysis(ticker)
        # Update setting so scheduler tracks the new one
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
