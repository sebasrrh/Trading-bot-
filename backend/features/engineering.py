import pandas as pd
import numpy as np
import logging
from core.config import settings

logger = logging.getLogger(__name__)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to the DataFrame using standard pandas.
    """
    if df.empty:
        return df
        
    df = df.copy()
    
    try:
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # MACD (12, 26, 9)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_12_26_9'] = ema_12 - ema_26
        df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
        df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
        
        # RSI (14) - simple Wilder's smoothed RSI approximation using SMA for simplicity
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Handle division by zero
        rs = np.where(loss == 0, 100, gain / loss)
        df['RSI_14'] = np.where(loss == 0, 100, 100 - (100 / (1 + rs)))
        
        # Bollinger Bands (20, 2)
        std_20 = df['Close'].rolling(window=20).std()
        df['BBL_20_2.0'] = df['SMA_20'] - (std_20 * 2)
        df['BBU_20_2.0'] = df['SMA_20'] + (std_20 * 2)
        
        # VWAP
        q = df['Volume']
        p = (df['High'] + df['Low'] + df['Close']) / 3
        # Simple VWAP proxy for each interval cumulative (in production, reset daily)
        df['VWAP_D'] = (p * q).cumsum() / q.cumsum()
        
        # Volume features
        df['VOL_SMA_20'] = df['Volume'].rolling(window=20).mean()
        df['VOL_SPIKE'] = (df['Volume'] > 2 * df['VOL_SMA_20']).astype(int)
        
    except Exception as e:
        logger.error(f"Error computing technical indicators: {e}")
        
    # Drop NaNs that are introduced by indicator lookback periods
    df.dropna(inplace=True)
    return df

def prepare_features(df: pd.DataFrame, sentiment_score: float) -> pd.DataFrame:
    """
    Creates the final feature set for training and prediction.
    """
    if df.empty:
        return df
        
    df = add_technical_indicators(df)
    
    # Target creation: 1 if next period close > current close, else 0
    df['Future_Close'] = df['Close'].shift(-1)
    df[settings.TARGET_COLUMN] = (df['Future_Close'] > df['Close']).astype(int)
    
    df['Sentiment'] = sentiment_score
    
    return df
