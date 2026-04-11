import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple
from core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_market_data(ticker: str, period: str = "30d", interval: str = "5m") -> pd.DataFrame:
    """
    Fetches historical OHLCV data from Yahoo Finance.
    """
    try:
        logger.info(f"Fetching market data for {ticker} (Period: {period}, Interval: {interval})")
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return pd.DataFrame()
            
        # Ensure index is datetime and localized/tz-naive as needed
        df.index = pd.to_datetime(df.index)
        
        # Drop columns like 'Dividends' and 'Stock Splits' if present
        cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[[c for c in cols_to_keep if c in df.columns]]
        
        return df
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return pd.DataFrame()

def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetches fundamental data from Yahoo Finance.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow")
        }
    except Exception as e:
        logger.error(f"Error fetching fundamentals: {e}")
        return {}

def fetch_sentiment_score(ticker: str) -> float:
    """
    Fetches recent news articles and computes a basic sentiment score.
    Using NewsAPI. If API key is missing, returns neutral (0.0).
    Score range: -1.0 to 1.0 (Mock NLP implementation for simplicity)
    """
    if not settings.NEWS_API_KEY or settings.NEWS_API_KEY == "your_newsapi_key_here":
        logger.warning("No NewsAPI key found, returning neutral sentiment.")
        return 0.0
        
    try:
        # Fetch news from yesterday to today
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        url = f"https://newsapi.org/v2/everything?q={ticker}&from={yesterday}&to={today}&sortBy=popularity&apiKey={settings.NEWS_API_KEY}"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if data.get("status") != "ok":
            logger.error(f"NewsAPI error: {data.get('message')}")
            return 0.0
            
        articles = data.get("articles", [])
        if not articles:
            return 0.0
            
        # Simple keyword-based sentiment
        positive_words = ['surge', 'jump', 'gain', 'profit', 'rise', 'up', 'beat', 'bull', 'upgrade']
        negative_words = ['plunge', 'drop', 'loss', 'fall', 'down', 'miss', 'bear', 'downgrade']
        
        score = 0.0
        for article in articles[:10]: # Process top 10 recent
            text = (str(article.get('title', '')) + " " + str(article.get('description', ''))).lower()
            
            p_count = sum(1 for w in positive_words if w in text)
            n_count = sum(1 for w in negative_words if w in text)
            
            if p_count > n_count:
                score += 0.1
            elif n_count > p_count:
                score -= 0.1
                
        # Clamp between -1.0 and 1.0
        return max(min(score, 1.0), -1.0)
        
    except Exception as e:
        logger.error(f"Error fetching sentiment: {e}")
        return 0.0
        
def fetch_systemic_risk() -> dict:
    """Fetch VIX term structure and mock FRED systemic indicators for V5."""
    try:
        # Fetch VIX and VIX3M for term structure backwardation check
        vix = yf.Ticker("^VIX").history(period="5d").Close.iloc[-1]
        vix3m = yf.Ticker("^VIX3M").history(period="5d").Close.iloc[-1]
        term_structure = "backwardation" if vix > vix3m else "contango"
    except Exception as e:
        logger.error(f"Error fetching VIX: {e}")
        vix, vix3m, term_structure = 20.0, 22.0, "contango"
        
    return {
        "vix": vix,
        "vix3m": vix3m,
        "term_structure": term_structure,
        "fred_hy_spread": 4.15, 
        "fred_nfci": -0.45       
    }

def fetch_v5_apex_layers(ticker: str) -> dict:
    """Mock advanced Alt-Data, GEX, DIX and Corporate Actions."""
    return {
        "form_4_status": "Neutral",
        "form_4_text": "No significant insider liquidation detected.",
        "wsj_alignment": "Aligned",
        "wsj_text": "Cost of capital stable. Institutional consensus neutral to positive. Oil margins steady.",
        "ticker_status": "Active", # Active, Halt, Delisted
        "corporate_events": "No imminent M&A or earnings calls within 48h.",
        "gex": "Positive", # Positive/Negative
        "dix": "High", # High/Low
        "alt_data": "Credit card transaction volume proxy holding steady.",
        "gamma_support": 100.0 # Placeholder for Gamma breakdown level
    }

def get_all_data(ticker: str) -> Tuple[pd.DataFrame, float, dict, dict, dict]:
    """Convenience func to fetch all current data state for V5."""
    df = fetch_market_data(ticker, period=f"{settings.HISTORY_DAYS}d", interval=settings.TIMEFRAME)
    sentiment = fetch_sentiment_score(ticker)
    fundamentals = fetch_fundamentals(ticker)
    systemic = fetch_systemic_risk()
    v5_data = fetch_v5_apex_layers(ticker)
    
    return df, sentiment, fundamentals, systemic, v5_data
