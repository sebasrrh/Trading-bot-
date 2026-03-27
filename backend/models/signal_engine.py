import pandas as pd
import numpy as np
import logging
from core.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def compute_rsi_slope(df: pd.DataFrame) -> pd.Series:
    """First difference of RSI_14 — positive means RSI is rising."""
    return df['RSI_14'].diff()


def detect_swing_lows(df: pd.DataFrame, order: int = 5) -> pd.Series:
    """
    Returns a boolean Series that is True at confirmed swing-low bars.
    A bar is a swing low if its Low is strictly less than the `order` bars
    immediately before it AND the `order` bars immediately after it.
    The right-side look-ahead is acceptable here because swing lows are
    used as structural reference levels on historical data, not real-time signals.
    """
    if len(df) < 2 * order + 1:
        return pd.Series(False, index=df.index)

    low = df['Low']
    left_min = low.shift(1).rolling(order).min()
    # Right-side minimum: reverse, rolling, reverse back
    right_min = low[::-1].shift(1).rolling(order).min()[::-1]

    return (low < left_min) & (low < right_min)


def find_nearest_support(df: pd.DataFrame, current_price: float, order: int = 5) -> float | None:
    """
    Returns the price of the nearest confirmed swing-low below current_price,
    or None if no valid level exists in the data.
    """
    if df.empty or len(df) < 2 * order + 1:
        return None

    swing_mask = detect_swing_lows(df, order)
    support_levels = df.loc[swing_mask, 'Low']
    candidates = support_levels[support_levels < current_price]

    if candidates.empty:
        return None

    return float(candidates.max())  # nearest support below price


# ---------------------------------------------------------------------------
# Kill condition checks
# ---------------------------------------------------------------------------

def check_kill_conditions(
    df: pd.DataFrame,
    walk_forward_accuracy: float,
    sentiment: float
) -> tuple[list[str], float]:
    """
    Evaluates the four hard kill conditions.
    Returns (kill_list, confidence_penalty).
    kill_list contains human-readable strings for each triggered condition.
    confidence_penalty is 0.20 when sentiment is absent, else 0.0.
    """
    kill_list: list[str] = []
    confidence_penalty = 0.0

    # Kill 1 — Model accuracy gate
    if walk_forward_accuracy < settings.ACCURACY_GATE_THRESHOLD:
        kill_list.append("MODEL ACCURACY INSUFFICIENT — NO SIGNAL")

    # Kill 2 — RSI oversold with no confirmed upturn
    if not df.empty and len(df) >= 3 and 'RSI_14' in df.columns:
        last = df.iloc[-1]
        if pd.notnull(last['RSI_14']) and last['RSI_14'] < 30:
            rsi_slope = compute_rsi_slope(df)
            slope_ok = (
                len(rsi_slope) >= 2
                and pd.notnull(rsi_slope.iloc[-1])
                and pd.notnull(rsi_slope.iloc[-2])
                and rsi_slope.iloc[-1] > 0
                and rsi_slope.iloc[-2] > 0
            )
            if not slope_ok:
                kill_list.append("RSI OVERSOLD — NO CONFIRMED UPTURN (REQUIRE ≥2 BARS POSITIVE SLOPE)")

    # Kill 3 — Triple bearish stack
    if not df.empty and all(c in df.columns for c in ['SMA_20', 'SMA_50', 'MACDh_12_26_9', 'MACD_12_26_9', 'MACDs_12_26_9']):
        last = df.iloc[-1]
        triple_bear = (
            pd.notnull(last['SMA_20']) and pd.notnull(last['SMA_50'])
            and last['SMA_20'] < last['SMA_50']
            and pd.notnull(last['MACDh_12_26_9']) and last['MACDh_12_26_9'] < 0
            and pd.notnull(last['MACD_12_26_9']) and pd.notnull(last['MACDs_12_26_9'])
            and last['MACD_12_26_9'] < last['MACDs_12_26_9']
        )
        if triple_bear:
            kill_list.append("TRIPLE BEARISH STACK — SMA20<SMA50 + MACD HISTOGRAM NEG + MACD<SIGNAL: BUY BLOCKED")

    # Kill 4 — Sentiment absent (data gap, not neutral)
    if sentiment == 0.0:
        kill_list.append("SENTIMENT DATA ABSENT — CONFIDENCE REDUCED 20%")
        confidence_penalty = 0.20

    return kill_list, confidence_penalty


# ---------------------------------------------------------------------------
# Confirmation stack requirements
# ---------------------------------------------------------------------------

def check_macd_deceleration(df: pd.DataFrame, bars: int = 3) -> bool:
    """REQ2: MACD histogram absolute values monotonically decreasing → momentum decelerating."""
    if df.empty or 'MACDh_12_26_9' not in df.columns or len(df) < bars:
        return False
    abs_hist = df['MACDh_12_26_9'].tail(bars).abs()
    return bool(abs_hist.is_monotonic_decreasing)


def check_volume_on_down_candles(df: pd.DataFrame, bars: int = 3) -> bool:
    """
    REQ3: Volume on down-candles declining for the most recent `bars` bars.
    A down-candle is one where Close < Open.
    If fewer than 2 down candles are found in the window, returns True (neutral — insufficient data).
    """
    if df.empty or not all(c in df.columns for c in ['Open', 'Close', 'Volume']):
        return False
    if len(df) < bars + 2:
        return True  # insufficient history → neutral

    window = df.tail(bars + 2)
    is_down = window['Close'] < window['Open']
    down_volumes = window.loc[is_down, 'Volume']

    if len(down_volumes) < 2:
        return True  # not enough down candles to confirm — neutral

    return bool(down_volumes.is_monotonic_decreasing)


def check_price_at_support(current_price: float, support: float | None, tol: float = 0.005) -> bool:
    """
    REQ4: Price is testing (not breaking) the nearest swing-low support.
    Uses asymmetric tolerance: 0.5% below is a structural break; up to 1.5% above is still a test.
    """
    if support is None:
        return False
    lower = support * (1 - tol)
    upper = support * (1 + tol * 3)
    return lower <= current_price <= upper


def detect_bullish_divergence(df: pd.DataFrame, lookback: int = 30) -> bool:
    """
    REQ6: Classic bullish divergence — price makes a lower low while RSI makes a higher low.
    Requires at least 2 confirmed swing lows within the lookback window.
    """
    if df.empty or len(df) < lookback or 'RSI_14' not in df.columns:
        return False

    window = df.tail(lookback).copy()
    swing_mask = detect_swing_lows(window, order=min(3, len(window) // 4))
    swing_indices = window.index[swing_mask].tolist()

    if len(swing_indices) < 2:
        return False

    # Most recent swing low vs the one before it
    last_idx = swing_indices[-1]
    prior_idx = swing_indices[-2]

    price_lower_low = window.loc[last_idx, 'Close'] < window.loc[prior_idx, 'Close']
    rsi_higher_low = window.loc[last_idx, 'RSI_14'] > window.loc[prior_idx, 'RSI_14']

    return bool(price_lower_low and rsi_higher_low)


def check_multitimeframe_alignment(df_1d: pd.DataFrame, df_htf: pd.DataFrame) -> str:
    """
    REQ5: Checks if 1D and higher-TF (1H proxy for 4H) both show an uptrend.
    Uptrend definition: Close > SMA_20 > SMA_50 AND MACDh >= 0.
    Returns 'ALIGNED' only if both frames confirm uptrend. Defaults to 'COUNTER-TREND'
    on any empty or incomplete data (conservative).
    """
    def is_uptrend(df: pd.DataFrame) -> bool:
        if df.empty or not all(c in df.columns for c in ['SMA_20', 'SMA_50', 'MACDh_12_26_9', 'Close']):
            return False
        last = df.iloc[-1]
        return bool(
            pd.notnull(last['Close']) and pd.notnull(last['SMA_20']) and pd.notnull(last['SMA_50'])
            and last['Close'] > last['SMA_20'] > last['SMA_50']
            and pd.notnull(last['MACDh_12_26_9']) and last['MACDh_12_26_9'] >= 0
        )

    if is_uptrend(df_1d) and is_uptrend(df_htf):
        return "ALIGNED"
    return "COUNTER-TREND"


def score_confirmation_stack(
    df_5m: pd.DataFrame,
    df_1d: pd.DataFrame,
    df_htf: pd.DataFrame
) -> tuple[int, dict[str, bool]]:
    """
    Evaluates all 6 BUY confirmation requirements.
    Returns (score, details_dict) where score is 0–6 and details maps REQ1..REQ6 to bool.
    """
    details: dict[str, bool] = {
        "REQ1_rsi_slope": False,
        "REQ2_macd_decel": False,
        "REQ3_volume_down_candles": False,
        "REQ4_price_at_support": False,
        "REQ5_mtf_aligned": False,
        "REQ6_bullish_divergence": False,
    }

    if df_5m.empty or len(df_5m) < 5:
        return 0, details

    last = df_5m.iloc[-1]

    # REQ1 — RSI rising ≥2 consecutive bars after touching < 30
    if 'RSI_14' in df_5m.columns and pd.notnull(last['RSI_14']) and last['RSI_14'] < 30:
        rsi_slope = compute_rsi_slope(df_5m)
        if (len(rsi_slope) >= 2
                and pd.notnull(rsi_slope.iloc[-1]) and pd.notnull(rsi_slope.iloc[-2])
                and rsi_slope.iloc[-1] > 0 and rsi_slope.iloc[-2] > 0):
            details["REQ1_rsi_slope"] = True

    # REQ2 — MACD histogram deceleration
    details["REQ2_macd_decel"] = check_macd_deceleration(df_5m)

    # REQ3 — Volume on down-candles declining
    details["REQ3_volume_down_candles"] = check_volume_on_down_candles(df_5m)

    # REQ4 — Price testing nearest swing-low support
    if 'Close' in df_5m.columns and pd.notnull(last['Close']):
        current_price = float(last['Close'])
        nearest_support = find_nearest_support(df_5m, current_price, order=settings.SWING_LOW_ORDER)
        details["REQ4_price_at_support"] = check_price_at_support(current_price, nearest_support)

    # REQ5 — Multi-timeframe alignment
    details["REQ5_mtf_aligned"] = check_multitimeframe_alignment(df_1d, df_htf) == "ALIGNED"

    # REQ6 — Bullish divergence
    details["REQ6_bullish_divergence"] = detect_bullish_divergence(df_5m, lookback=settings.DIVERGENCE_LOOKBACK)

    score = sum(details.values())
    return score, details


# ---------------------------------------------------------------------------
# Fundamentals risk evaluation
# ---------------------------------------------------------------------------

def evaluate_fundamentals_risk(fundamentals: dict, current_price: float | None = None) -> str:
    """
    Produces a primary risk string from fundamental data.
    Priority: elevated valuation → no dividend cushion → near 52-week high → standard.
    """
    risks: list[str] = []

    pe = fundamentals.get("pe_ratio")
    if pe is not None and isinstance(pe, (int, float)) and pe > 100:
        risks.append(f"ELEVATED VALUATION RISK: P/E {pe:.1f} > 100")

    div = fundamentals.get("dividend_yield")
    if div is None or div == 0:
        risks.append("No dividend cushion — downside unmitigated by income")

    if current_price is not None:
        high_52w = fundamentals.get("52_week_high")
        low_52w = fundamentals.get("52_week_low")
        if (high_52w and low_52w and isinstance(high_52w, (int, float))
                and isinstance(low_52w, (int, float)) and high_52w != low_52w):
            range_pos = (current_price - low_52w) / (high_52w - low_52w)
            if range_pos > 0.85:
                risks.append(f"Near 52-week high ({range_pos*100:.0f}% of range) — elevated drawdown risk")

    return " | ".join(risks) if risks else "Standard risk profile"


# ---------------------------------------------------------------------------
# Reasoning builder
# ---------------------------------------------------------------------------

def build_reasoning(
    kill_list: list[str],
    req_details: dict[str, bool],
    alignment: str,
    accuracy: float,
    raw_conf: float,
    adj_conf: float,
    score: int,
    signal: str,
    fundamentals_risk: str,
) -> str:
    lines = []

    lines.append(f"[1] MODEL ACCURACY GATE: {accuracy*100:.1f}% — {'PASS' if accuracy >= settings.ACCURACY_GATE_THRESHOLD else 'FAIL (threshold: 53%)'}")

    if kill_list:
        lines.append(f"[2] KILL CONDITIONS TRIGGERED ({len(kill_list)}):")
        for k in kill_list:
            lines.append(f"    • {k}")
    else:
        lines.append("[2] KILL CONDITIONS: NONE")

    lines.append(f"[3] CONFIRMATION STACK ({score}/6 requirements met):")
    req_labels = {
        "REQ1_rsi_slope":           "REQ1 — RSI rising ≥2 bars after touching <30",
        "REQ2_macd_decel":          "REQ2 — MACD histogram decelerating toward zero",
        "REQ3_volume_down_candles": "REQ3 — Volume on down-candles declining (3 bars)",
        "REQ4_price_at_support":    "REQ4 — Price testing nearest structural support",
        "REQ5_mtf_aligned":         "REQ5 — Multi-timeframe alignment (1D + 1H)",
        "REQ6_bullish_divergence":  "REQ6 — Bullish divergence confirmed",
    }
    for key, label in req_labels.items():
        status = "PASS" if req_details.get(key) else "FAIL"
        lines.append(f"    {label}: {status}")

    lines.append(f"[4] TIMEFRAME ALIGNMENT: {alignment}")
    if alignment == "COUNTER-TREND":
        lines.append("    → Counter-trend trade: reduce position sizing by 50%")

    lines.append(f"[5] CONFIDENCE CALCULATION: raw={raw_conf*100:.1f}% × accuracy={accuracy*100:.1f}% = adjusted={adj_conf*100:.1f}%")

    lines.append(f"[6] FUNDAMENTALS: {fundamentals_risk}")

    lines.append(f"[7] SIGNAL DETERMINATION: {signal}")
    if signal == "BUY":
        lines.append(f"    → {score}/6 requirements met (≥{settings.BUY_CONFIRMATION_REQUIRED} required for BUY)")
    elif signal == "HOLD":
        lines.append(f"    → {score}/6 requirements met (={settings.HOLD_CONFIRMATION_REQUIRED} for HOLD)")
    else:
        lines.append(f"    → {score}/6 requirements met (<{settings.HOLD_CONFIRMATION_REQUIRED} = NO SIGNAL) or kill condition active")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_signal_engine(
    df_5m: pd.DataFrame,
    df_1d: pd.DataFrame,
    df_htf: pd.DataFrame,
    raw_confidence: float,
    walk_forward_accuracy: float,
    sentiment: float,
    fundamentals: dict,
) -> dict:
    """
    Runs the full institutional signal engine pipeline.
    Returns a dict matching the required output schema.
    """
    # Step 1 — Kill conditions
    kill_list, confidence_penalty = check_kill_conditions(df_5m, walk_forward_accuracy, sentiment)

    # Step 2 — Accuracy gate string
    accuracy_gate = (
        f"{walk_forward_accuracy*100:.1f}% "
        f"{'PASS' if walk_forward_accuracy >= settings.ACCURACY_GATE_THRESHOLD else 'FAIL'}"
    )

    # Step 3 — Adjusted confidence (cardinal rule: raw × accuracy, minus sentiment penalty)
    adj_conf = max(0.0, min(1.0, (raw_confidence - confidence_penalty) * walk_forward_accuracy))

    # Step 4 — Default values for early-abort path
    score = 0
    req_details: dict[str, bool] = {
        "REQ1_rsi_slope": False, "REQ2_macd_decel": False,
        "REQ3_volume_down_candles": False, "REQ4_price_at_support": False,
        "REQ5_mtf_aligned": False, "REQ6_bullish_divergence": False,
    }
    signal = "NO SIGNAL"

    # Check for hard abort conditions
    accuracy_kill = any("MODEL ACCURACY INSUFFICIENT" in k for k in kill_list)
    confidence_too_low = adj_conf < settings.ADJUSTED_CONFIDENCE_THRESHOLD

    if not accuracy_kill and not confidence_too_low:
        # Step 5 — Score the confirmation stack
        score, req_details = score_confirmation_stack(df_5m, df_1d, df_htf)

        # Step 6 — Determine signal
        buy_blocking_kill = any(
            "TRIPLE BEARISH STACK" in k or "RSI OVERSOLD" in k
            for k in kill_list
        )

        if score >= settings.BUY_CONFIRMATION_REQUIRED and not buy_blocking_kill:
            signal = "BUY"
        elif score >= settings.HOLD_CONFIRMATION_REQUIRED:
            signal = "HOLD"
        else:
            signal = "NO SIGNAL"
    elif confidence_too_low and not accuracy_kill:
        kill_list.append(
            f"ADJUSTED CONFIDENCE {adj_conf*100:.1f}% BELOW THRESHOLD "
            f"({settings.ADJUSTED_CONFIDENCE_THRESHOLD*100:.0f}%) — NO SIGNAL"
        )

    # Step 7 — Multi-timeframe alignment (computed independently for reporting)
    alignment = check_multitimeframe_alignment(df_1d, df_htf)

    # Step 8 — Invalidation level (nearest swing-low support)
    current_price: float | None = None
    invalidation_level: float | None = None
    if not df_5m.empty and 'Close' in df_5m.columns:
        current_price = float(df_5m.iloc[-1]['Close'])
        invalidation_level = find_nearest_support(df_5m, current_price, order=settings.SWING_LOW_ORDER)

    # Step 9 — Fundamentals risk
    primary_risk = evaluate_fundamentals_risk(fundamentals, current_price)

    # Step 10 — Reasoning text
    reasoning = build_reasoning(
        kill_list=kill_list,
        req_details=req_details,
        alignment=alignment,
        accuracy=walk_forward_accuracy,
        raw_conf=raw_confidence,
        adj_conf=adj_conf,
        score=score,
        signal=signal,
        fundamentals_risk=primary_risk,
    )

    return {
        "signal": signal,
        "confirmation_score": f"{score}/6",
        "kill_conditions": kill_list if kill_list else ["NONE"],
        "model_accuracy_gate": accuracy_gate,
        "confidence_adjusted": round(adj_conf, 4),
        "timeframe_alignment": alignment,
        "primary_risk": primary_risk,
        "invalidation_level": invalidation_level,
        "reasoning": reasoning,
        # Legacy fields preserved for backward compatibility
        "confidence": round(raw_confidence, 4),
        "model_accuracy": round(walk_forward_accuracy, 4),
    }
