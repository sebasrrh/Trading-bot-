import os
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseModel):
    PROJECT_NAME: str = "AI Trading App"
    VERSION: str = "1.0.0"
    
    # API Keys
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    
    # Trading/Model Settings
    TICKER: str = "AAPL" # Default ticker to track
    TIMEFRAME: str = "5m" # Data timeframe
    HISTORY_DAYS: int = 30 # Days of history to fetch for training
    PREDICTION_INTERVAL_MINUTES: int = 2
    
    # Model configuration
    TARGET_COLUMN: str = "Target"

    # Signal Engine Thresholds
    ACCURACY_GATE_THRESHOLD: float = 0.53
    ADJUSTED_CONFIDENCE_THRESHOLD: float = 0.40
    BUY_CONFIRMATION_REQUIRED: int = 5
    HOLD_CONFIRMATION_REQUIRED: int = 4
    SWING_LOW_ORDER: int = 5
    DIVERGENCE_LOOKBACK: int = 30
    MTF_1D_PERIOD: str = "1y"
    MTF_HTF_PERIOD: str = "60d"

settings = Settings()
