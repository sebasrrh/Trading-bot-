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
    
settings = Settings()
