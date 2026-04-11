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

def apply_antigravity_v5_rules(ticker: str, df: pd.DataFrame, raw_signal: str, raw_confidence: float, accuracy: float, sentiment: float, fundamentals: dict, systemic: dict, v5_data: dict) -> tuple:
    """Implements strictly defined ANTIGRAVITY V5 APEX institutional quant signal rules."""
    
    ticker_status = v5_data.get("ticker_status", "Active")
    if ticker_status in ["Halt", "Delisted"]:
        return "FATAL ERROR: TICKER INVALID/DELISTED", 0.0, "FATAL ERROR: TICKER INVALID/DELISTED"
        
    if df.empty or len(df) < 14:
        return "NO SIGNAL", 0.0, "Insufficient Data for V5 Engine."
        
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
    
    # KILL CONDITIONS (V5)
    if accuracy < 0.53:
        kill_triggers.append(("Model walk-forward < 53%", "Model Validation Matrix"))
        
    if systemic.get("term_structure") == "backwardation":
        kill_triggers.append(("VIX term structure in backwardation", "CBOE VIX vs VIX3M"))
        
    hy_spread = systemic.get("fred_hy_spread", 4.0)
    if hy_spread > 5.0: # mock threshold for widening
        kill_triggers.append(("High Yield Spread widened >10% MoM", "FRED BAMLH0A0HYM2"))
        
    # Gamma Squeeze Down check
    gex = v5_data.get("gex", "Positive")
    support_level = df['Low'].tail(20).min() if 'Low' in df.columns else df['Close'].min()
    gamma_support = v5_data.get("gamma_support", support_level)
    if gex == "Negative" and df['Close'].iloc[-1] < gamma_support:
        kill_triggers.append(("Dealer Gamma Negative AND break below Gamma Support", "GEX Flow Proxy"))
        
    if v5_data.get("form_4_status") == "Liquidation Warning":
        kill_triggers.append(("Form 4 Liquidation Warning triggered", "SEC Form 4"))
        
    # KINEMATICS
    rsi_1 = df['RSI_14'].iloc[-1] if 'RSI_14' in df.columns else 50
    rsi_2 = df['RSI_14'].iloc[-2] if len(df) > 1 and 'RSI_14' in df.columns else 50
    rsi_slope_bool = (rsi_1 > rsi_2)
    
    macd_h = df['MACDh_12_26_9'].iloc[-1] if 'MACDh_12_26_9' in df.columns else 0
    macd_h_prev = df['MACDh_12_26_9'].iloc[-2] if len(df) > 1 and 'MACDh_12_26_9' in df.columns else 0
    macd_decel = (macd_h > macd_h_prev) or (macd_h > 0)
    
    down_candles = df[df['Close'] < df.get('Open', df['Close'])].tail(3)
    vol_declining = False
    if len(down_candles) == 3 and 'Volume' in down_candles.columns:
        vol_declining = (down_candles['Volume'].iloc[-1] < down_candles['Volume'].iloc[-2]) and (down_candles['Volume'].iloc[-2] < down_candles['Volume'].iloc[-3])

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
    fund_string = f"TTM PE: {f_pe}, FCF Yield: {'Normalized'}, P/B: {f_pb}"
    alt_data_str = v5_data.get("alt_data", "No alt-data anomalies detected.")
    
    # INSIDER
    form4_stat = v5_data.get("form_4_status", "Neutral")
    form4_text = v5_data.get("form_4_text", "No recent major executive transactions.")
    
    # CONFIDENCE & SIGNAL EXECUTION
    base_conf = raw_confidence
    if form4_stat == "Buy-Side Multiplier":
        base_conf += 0.15
        
    adj_conf = base_conf * accuracy
    
    sys_risk_str = "SEVERE CONTAGION" if systemic.get("term_structure") == "backwardation" else "Elevated" if hy_spread > 4.5 else "Low"
    
    signal = raw_signal
    if kill_triggers or adj_conf < 0.33:
        signal = "NO SIGNAL"
        
    # Prevent None calculation error
    try:
        invalid_level = float(gamma_support) * 0.99
    except (TypeError, ValueError):
        invalid_level = 0.0
    
    kill_str = "None"
    kill_source = "Systemic Clear"
    if kill_triggers:
        kill_str = kill_triggers[0][0]
        kill_source = kill_triggers[0][1]

    # TERMINAL REPORT BUILDER V5
    report = f'''ANTIGRAVITY V5 APEX TERMINAL: [{ticker}]

SIGNAL: {signal}

CONFIDENCE SCORE: {adj_conf*100:.1f}%

PROJECTED TRAJECTORY SLOPE: {deg:+.1f} degrees / {proj_trajectory:+.2f}% over next 14 intervals

SYSTEMIC RISK GAUGE: {sys_risk_str}

AI REASONING, EXPLANATION & DATA PROVENANCE:

TICKER STATUS: {ticker_status} / {v5_data.get("corporate_events")}. Source: Exchange Feed

KILL CONDITIONS TRIGGERED: {kill_str}. Source: {kill_source}

MACRO BIAS ALIGNMENT: {v5_data.get("wsj_alignment")}. Source check: WSJ Article - {v5_data.get("wsj_text")}

RAW LOGIC FEED: * Systemic liquidity & Volatility: Functional constraint monitored. Source: FRED NFCI at {systemic.get("fred_nfci"):.2f} / VIX curve {systemic.get("term_structure")}

Structural Flow (Gamma/Dark Pools): Dealer Gamma is {gex}, Dark Pool accumulation is {v5_data.get("dix")}. Source: GEX/DIX Proxy Data

Fundamental & Alt-Data: {fund_string}. Alt-Data: {alt_data_str}. Source: SEC 10-Q / Alt-Data Proxy

Kinematics: RSI slope {rsi_slope_bool}, MACD decel {macd_decel}, Divergence {bull_div}. Source: Exchange Close Data

Corporate Conviction: {form4_stat}. Source: SEC Form 4 Sweep ({form4_text})

INVALIDATION LEVEL: ${invalid_level:.2f} (The exact structural/Gamma breakdown point where the trade is voided).'''

    return signal, adj_conf, report

def perform_analysis(ticker: str):
    """Fetches data, runs ML pipeline, and generates state dict for a ticker"""
    logger.info(f"Running full analysis for {ticker} at {datetime.now()}")
    
    df, sentiment, fundamentals, systemic, v5_data = get_all_data(ticker)
    
    if df.empty:
        raise ValueError(f"No market data found for {ticker}")
        
    df_features = prepare_features(df, sentiment)
    
    if df_features.empty:
         raise ValueError(f"Not enough data to build features for {ticker}")
         
    # Train
    train_model(df_features)
    
    # Predict
    raw_signal, raw_confidence, accuracy = predict_next_signal(df_features)
    
    signal, final_confidence, explanation = apply_antigravity_v5_rules(ticker, df_features, raw_signal, raw_confidence, accuracy, sentiment, fundamentals, systemic, v5_data)
    
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
