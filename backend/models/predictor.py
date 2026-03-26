import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
import logging
from core.config import settings

logger = logging.getLogger(__name__)

# Cache model in memory
_model = None
_last_accuracy = 0.0

def train_model(df: pd.DataFrame) -> HistGradientBoostingClassifier:
    """
    Trains an XGBoost classifier on the historical data.
    """
    global _model, _last_accuracy
    
    if df.empty or len(df) < 50:
        logger.warning("Not enough data to train model.")
        return None
        
    # The last row has NaN for Future_Close or Target, so we drop it for training
    train_df = df.dropna(subset=[settings.TARGET_COLUMN])
    
    if len(train_df) < 50:
         logger.warning("Not enough data after dropping NaNs to train model.")
         return None
         
    # Define features to drop
    features_to_drop = [settings.TARGET_COLUMN, 'Future_Close', 'Open', 'High', 'Low', 'Close']
    X = train_df.drop(columns=[col for col in features_to_drop if col in train_df.columns])
    y = train_df[settings.TARGET_COLUMN]
    
    # Chronological Train/Test split to avoid lookahead bias
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    logger.info(f"Model trained. Test Accuracy: {accuracy:.4f}")
    
    _model = model
    _last_accuracy = accuracy
    return model

def predict_next_signal(df: pd.DataFrame):
    """
    Predicts the signal for the very last row (current active interval).
    Returns signal ('BUY', 'SELL', 'HOLD'), confidence, and latest accuracy.
    """
    global _model, _last_accuracy
    
    if _model is None:
        return "HOLD", 0.0, 0.0
        
    if df.empty:
        return "HOLD", 0.0, 0.0
        
    # Get the latest row (the one we want to predict the future for)
    latest_row = df.iloc[-1:]
    
    features_to_drop = [settings.TARGET_COLUMN, 'Future_Close', 'Open', 'High', 'Low', 'Close']
    X_latest = latest_row.drop(columns=[col for col in features_to_drop if col in latest_row.columns])
    
    # Ensure proper ordering of columns as expected by the trained model
    expected_cols = _model.feature_names_in_
    X_latest = X_latest[expected_cols]
    
    prob = _model.predict_proba(X_latest)[0]  # [prob_down, prob_up]
    prob_up = prob[1]
    
    signal = "HOLD"
    confidence = prob_up
    
    # Simple thresholds for signals
    if prob_up > 0.55:
        signal = "BUY"
    elif prob_up < 0.45:
        signal = "SELL"
        confidence = prob[0] # Confidence is the prob of Down
        
    return signal, float(confidence), float(_last_accuracy)
