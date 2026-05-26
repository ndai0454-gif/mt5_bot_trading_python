import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NEUTRAL = "NEUTRAL"
LONG = "LONG"
SHORT = "SHORT"
BULLISH = "BULLISH"
BEARISH = "BEARISH"

def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
    return prices.ewm(span=period, adjust=False).mean()

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(com=period - 1, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(com=period - 1, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(com=period - 1, adjust=False).mean() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(com=period - 1, adjust=False).mean()

def get_ema_alignment(ef, em, es) -> str:
    if ef > em > es:
        return BULLISH
    if ef < em < es:
        return BEARISH
    return NEUTRAL

def check_pullback(df, em_ser, es_ser, alignment, lookback=3) -> bool:
    if len(df) < lookback + 1:
        return False
    lows = df["low"].iloc[-lookback - 1:-1]
    highs = df["high"].iloc[-lookback - 1:-1]
    em = em_ser.iloc[-lookback - 1:-1]
    es = es_ser.iloc[-lookback - 1:-1]
    if alignment == BULLISH:
        return (lows <= em).any() or (lows <= es).any()
    if alignment == BEARISH:
        return (highs >= em).any() or (highs >= es).any()
    return False

def check_engulfing(df, direction) -> bool:
    if len(df) < 2:
        return False
    c1, o1 = df["close"].iloc[-2], df["open"].iloc[-2]
    c2, o2 = df["close"].iloc[-1], df["open"].iloc[-1]
    if direction == LONG:
        return (c2 > o2) and (o2 <= c1) and (c2 > o1)
    if direction == SHORT:
        return (c2 < o2) and (o2 >= c1) and (c2 < o1)
    return False

def check_candle_confirmation(df, direction, min_body_ratio=0.5) -> bool:
    if len(df) < 2:
        return False
    candle = df.iloc[-2]
    candle_range = candle["high"] - candle["low"]
    if candle_range == 0:
        return False
    body = abs(candle["close"] - candle["open"])
    if body / candle_range < min_body_ratio:
        return False
    return (candle["close"] > candle["open"]) if direction == LONG else (candle["close"] < candle["open"])

class SignalEngine:
    def __init__(self, config: dict):
        self.cfg = config

    def check_trend_alignment(self, df_h1: Optional[pd.DataFrame]) -> str:
        if df_h1 is None or len(df_h1) < 200:
            return NEUTRAL
        ema_200 = calculate_ema(df_h1["close"], self.cfg.get("ema_trend", 200))
        close = df_h1["close"].iloc[-1]
        return BULLISH if close > ema_200.iloc[-1] else BEARISH if close < ema_200.iloc[-1] else NEUTRAL

    def _attach_price_levels(self, result: dict, price: float, atr: float, direction: str, min_sl: float = 0.0):
        sl_dist = max(atr * self.cfg["atr_sl_multiplier"], min_sl)
        tp1_m = self.cfg["tp1_multiplier"]
        tp2_m = self.cfg["tp2_multiplier"]
        tp3_m = self.cfg["tp3_multiplier"]
        if direction == LONG:
            result.update(
                {
                    "sl": price - sl_dist,
                    "tp1": price + sl_dist * tp1_m,
                    "tp2": price + sl_dist * tp2_m,
                    "tp3": price + sl_dist * tp3_m,
                }
            )
        else:
            result.update(
                {
                    "sl": price + sl_dist,
                    "tp1": price - sl_dist * tp1_m,
                    "tp2": price - sl_dist * tp2_m,
                    "tp3": price - sl_dist * tp3_m,
                }
            )
        result["sl_distance"] = sl_dist
        return result

    def analyse(self, df: pd.DataFrame) -> Dict[str, Any]:
        calc_df = df.tail(500).copy()
        close = calc_df["close"]
        ema_f = calculate_ema(close, self.cfg["ema_fast"])
        ema_m = calculate_ema(close, self.cfg["ema_medium"])
        ema_s = calculate_ema(close, self.cfg["ema_slow"])
        rsi = calculate_rsi(close, self.cfg["rsi_period"])
        atr = calculate_atr(calc_df, self.cfg["atr_period"])
        adx = calculate_adx(calc_df, self.cfg.get("adx_period", 14))
        return {
            "ema_fast": ema_f.iloc[-1],
            "ema_medium": ema_m.iloc[-1],
            "ema_slow": ema_s.iloc[-1],
            "rsi": rsi.iloc[-1],
            "atr": atr.iloc[-1],
            "adx": adx.iloc[-1],
            "close": close.iloc[-1],
            "alignment": get_ema_alignment(ema_f.iloc[-1], ema_m.iloc[-1], ema_s.iloc[-1]),
            "df": calc_df,
            "ema_medium_series": ema_m,
            "ema_slow_series": ema_s,
        }

    def get_signal(self, df: pd.DataFrame, spread: float, df_h1: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        data = self.analyse(df)
        filters = {}
        h1_trend = self.check_trend_alignment(df_h1)

        filters["atr_active"] = data["atr"] >= self.cfg["atr_min_threshold"]
        filters["spread_ok"] = spread <= self.cfg.get("max_spread_points", 30)
        filters["ema_aligned"] = data["alignment"] in (BULLISH, BEARISH)
        filters["adx_strong"] = data["adx"] >= self.cfg.get("adx_threshold", 20)

        direction = LONG if data["alignment"] == BULLISH else SHORT if data["alignment"] == BEARISH else NEUTRAL
        if df_h1 is not None and ((direction == LONG and h1_trend != BULLISH) or (direction == SHORT and h1_trend != BEARISH)):
            direction = NEUTRAL

        if direction != NEUTRAL:
            engulfing = check_engulfing(data["df"], direction)
            candle_confirm = check_candle_confirmation(data["df"], direction, self.cfg["candle_body_min_ratio"])
            filters["candle_confirmed"] = engulfing or candle_confirm

            if direction == LONG:
                filters["rsi_zone"] = self.cfg["rsi_long_min"] <= data["rsi"] <= self.cfg["rsi_long_max"]
            else:
                filters["rsi_zone"] = self.cfg["rsi_short_min"] <= data["rsi"] <= self.cfg["rsi_short_max"]
        else:
            filters["candle_confirmed"] = False
            filters["rsi_zone"] = False

        required_filters = ["atr_active", "spread_ok", "ema_aligned", "adx_strong", "candle_confirmed", "rsi_zone"]
        all_pass = all(filters.get(f, False) for f in required_filters)

        result = {
            "direction": direction if all_pass else NEUTRAL,
            "filters": filters,
            "all_pass": all_pass,
            **{k: v for k, v in data.items() if k not in ("ema_medium_series", "ema_slow_series", "df")},
        }
        if all_pass and direction != NEUTRAL:
            self._attach_price_levels(result, data["close"], data["atr"], direction)
        return result