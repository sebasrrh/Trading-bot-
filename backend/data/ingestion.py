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
        
def fetch_multitimeframe_data(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetches OHLCV data at three timeframes: 5m (primary), 1d, and 1h (4H proxy).
    yfinance does not offer a native 4H interval; 1h over 60 days is used instead.
    Returns (df_5m, df_1d, df_htf) — any frame may be an empty DataFrame on error.
    """
    df_5m = fetch_market_data(ticker, period=f"{settings.HISTORY_DAYS}d", interval=settings.TIMEFRAME)
    df_1d = fetch_market_data(ticker, period=settings.MTF_1D_PERIOD, interval="1d")
    df_htf = fetch_market_data(ticker, period=settings.MTF_HTF_PERIOD, interval="1h")
    return df_5m, df_1d, df_htf


def get_all_data(ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float, dict]:
    """Convenience func to fetch all current data state across timeframes"""
    df_5m, df_1d, df_htf = fetch_multitimeframe_data(ticker)
    sentiment = fetch_sentiment_score(ticker)
    fundamentals = fetch_fundamentals(ticker)
    return df_5m, df_1d, df_htf, sentiment, fundamentals
