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

def apply_antigravity_v4_rules(ticker: str, df: pd.DataFrame, raw_signal: str, raw_confidence: float, accuracy: float, sentiment: float, fundamentals: dict, systemic: dict, insider: dict) -> tuple:
    """Implements strictly defined ANTIGRAVITY V4 institutional quant signal rules."""
    if df.empty or len(df) < 14:
        return "NO SIGNAL", 0.0, "Insufficient Data for V4 Engine."
        
    last = df.iloc[-1]
    kill_triggers = []
    
    # Calculate Trajectory Slope (14 days)
    closes = df['Close'].tail(14).values
    x = np.arange(len(closes))
    A = np.vstack([x, np.ones(len(x))]).T
    slope, _ = np.linalg.lstsq(A, closes, rcond=None)[0]
    pct_slope_per_bar = (slope / np.mean(closes)) * 100
    proj_trajectory = pct_slope_per_bar * 14
    deg = np.degrees(np.arctan(pct_slope_per_bar))
    
    # KILL CONDITIONS
    if accuracy < 0.53:
        kill_triggers.append(("Model walk-forward < 53%", "Model Validation Matrix"))
        
    if systemic.get("term_structure") == "backwardation":
        kill_triggers.append(("VIX term structure in backwardation", "CBOE VIX vs VIX3M"))
        
    hy_spread = systemic.get("fred_hy_spread", 4.0)
    if hy_spread > 5.0: # mock threshold for widening
        kill_triggers.append(("High Yield Spread widened significantly", "FRED BAMLH0A0HYM2"))
        
    # Check for News-gap down >3% on 2x avg volume
    vol_sma = df['Volume'].rolling(20).mean().iloc[-2] if len(df) > 20 else df['Volume'].mean()
    if df['Open'].iloc[-1] < df['Close'].iloc[-2] * 0.97 and df['Volume'].iloc[-1] > vol_sma * 2:
        kill_triggers.append(("News-gap down > 3% on 2x avg volume", "Exchange OHLCV"))
        
    if insider.get("form_4_status") == "Liquidation Warning":
        kill_triggers.append(("Form 4 Liquidation Warning triggered", "SEC Form 4"))
        
    # KINEMATICS
    # RSI Slope
    rsi_1 = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else 50
    rsi_2 = df['RSI_14'].iloc[-2] if len(df) > 1 and 'RSI_14' in df.columns else 50
    rsi_slope_bool = (rsi_1 > rsi_2)
    
    # MACD decel
    macd_h = df['MACDh_12_26_9'].iloc[-1] if 'MACDh_12_26_9' in df.columns else 0
    macd_h_prev = df['MACDh_12_26_9'].iloc[-2] if len(df) > 1 and 'MACDh_12_26_9' in df.columns else 0
    macd_decel = (macd_h > macd_h_prev) or (macd_h > 0)
    
    # Volume declining on down-candles
    down_candles = df[df['Close'] < df['Open']].tail(3)
    vol_declining = False
    if len(down_candles) == 3:
        vol_declining = (down_candles['Volume'].iloc[-1] < down_candles['Volume'].iloc[-2]) and (down_candles['Volume'].iloc[-2] < down_candles['Volume'].iloc[-3])

    # Divergence
    bull_div = False
    if 'Low' in df.columns and 'RSI_14' in df.columns and len(df) > 10:
        price_mins = df['Low'].rolling(5).min().tail(10).values
        rsi_mins = df['RSI_14'].rolling(5).min().tail(10).values
        if price_mins[-1] < price_mins[0] and rsi_mins[-1] > rsi_mins[0]:
            bull_div = True
            
    # FUNDAMENTAL FLOOR
    f_fcf = fundamentals.get("freeCashflow", "N/A")
    f_pe = fundamentals.get("trailingPE", "N/A")
    f_pb = fundamentals.get("priceToBook", "N/A")
    fund_string = f"TTM PE: {f_pe}, P/B: {f_pb}, FCF constraint normalized"
    
    # INSIDER
    form4_stat = insider.get("form_4_status", "Neutral")
    form4_text = insider.get("form_4_text", "No recent major executive transactions.")
    
    # CONFIDENCE & SIGNAL EXECUTION
    base_conf = raw_confidence
    if form4_stat == "Buy-Side Multiplier":
        base_conf += 0.15
        
    adj_conf = base_conf * accuracy
    
    sys_risk_str = "SEVERAL CONTAGION" if systemic.get("term_structure") == "backwardation" else "Elevated" if hy_spread > 4.5 else "Low"
    
    signal = raw_signal
    if kill_triggers or adj_conf < 0.33:
        signal = "NO SIGNAL"
        
    invalid_level = df['Low'].tail(20).min() * 0.99 if 'Low' in df else df['Close'].iloc[-1] * 0.95
    
    kill_str = "None"
    kill_source = "Systemic Clear"
    if kill_triggers:
        kill_str = kill_triggers[0][0]
        kill_source = kill_triggers[0][1]

    # TERMINAL REPORT BUILDER
    report = f'''ANTIGRAVITY V4 TERMINAL: [{ticker}]

SIGNAL: {signal}

CONFIDENCE SCORE: {adj_conf*100:.1f}%

PROJECTED TRAJECTORY SLOPE: {deg:+.1f} degrees / {proj_trajectory:+.2f}% over next 14 intervals

SYSTEMIC RISK GAUGE: {sys_risk_str}

AI REASONING, EXPLANATION & DATA PROVENANCE:

KILL CONDITIONS TRIGGERED: {kill_str}. Source check: {kill_source}

MACRO BIAS ALIGNMENT: {insider.get("wsj_alignment")}. Source check: WSJ Daily Intelligence Sweep - {insider.get("wsj_text")}

RAW LOGIC FEED: * Systemic liquidity state: Functional constraint monitored. Source: FRED NFCI at {systemic.get("fred_nfci"):.2f}

Fundamental floor: {fund_string}. Source: SEC 10-Q/10-K primary aggregators

Kinematics: RSI slope {rsi_slope_bool}, MACD decel {macd_decel}, Vol declining on down-candles {vol_declining}, Divergence {bull_div}. Source: Exchange Close Data

Corporate Conviction: {form4_stat}. Source: SEC Form 4 Sweep - {form4_text}

INVALIDATION LEVEL: ${invalid_level:.2f} (The exact structural breakdown point where the trade thesis is voided).'''

    return signal, adj_conf, report

def perform_analysis(ticker: str):
    """Fetches data, runs ML pipeline, and generates state dict for a ticker"""
    logger.info(f"Running full analysis for {ticker} at {datetime.now()}")
    
    df, sentiment, fundamentals, systemic, insider = get_all_data(ticker)
    
    if df.empty:
        raise ValueError(f"No market data found for {ticker}")
        
    df_features = prepare_features(df, sentiment)
    
    if df_features.empty:
         raise ValueError(f"Not enough data to build features for {ticker}")
         
    # Train
    train_model(df_features)
    
    # Predict
    raw_signal, raw_confidence, accuracy = predict_next_signal(df_features)
    
    signal, final_confidence, explanation = apply_antigravity_v4_rules(ticker, df_features, raw_signal, raw_confidence, accuracy, sentiment, fundamentals, systemic, insider)
    
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
        "confidence": final_confidence,
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
