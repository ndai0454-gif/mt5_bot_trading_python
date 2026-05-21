import logging
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Constants
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
    # Sử dụng Wilder's Smoothing (com = period - 1)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    
    # Tránh chia cho 0
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def get_ema_alignment(ema_fast: float, ema_medium: float, ema_slow: float) -> str:
    if ema_fast > ema_medium > ema_slow:
        return BULLISH
    if ema_fast < ema_medium < ema_slow:
        return BEARISH
    return NEUTRAL


def get_ema_slope(ema_series: pd.Series, lookback: int = 3) -> float:
    """Returns the average slope (price change per bar) over the last `lookback` bars."""
    if len(ema_series) < lookback + 1:
        return 0.0
    recent = ema_series.iloc[-lookback:]
    return float(recent.diff().mean())


def check_pullback(df: pd.DataFrame, ema_medium: pd.Series, ema_slow: pd.Series, alignment: str, lookback: int = 3) -> bool:
    """
    True if price has pulled back to touch EMA_M or EMA_S in the specified lookback window
    before the current bar.
    """
    if len(df) < lookback + 1:
        return False

    # Lấy dữ liệu từ nến thứ (lookback + 1) lùi về cho đến trước nến hiện tại
    start_idx = -(lookback + 1)
    end_idx = -1
    
    lows = df["low"].iloc[start_idx:end_idx]
    highs = df["high"].iloc[start_idx:end_idx]
    ema_m = ema_medium.iloc[start_idx:end_idx]
    ema_s = ema_slow.iloc[start_idx:end_idx]

    if alignment == BULLISH:
        # Giá chạm hoặc xuyên qua EMA trong xu hướng tăng
        return (lows <= ema_m).any() or (lows <= ema_s).any()
    elif alignment == BEARISH:
        # Giá chạm hoặc xuyên qua EMA trong xu hướng giảm
        return (highs >= ema_m).any() or (highs >= ema_s).any()

    return False


def check_candle_confirmation(df: pd.DataFrame, direction: str, min_body_ratio: float = 0.5) -> bool:
    """Last closed candle must close in trend direction with body > min_body_ratio of total range."""
    if len(df) < 2:
        return False

    # Lấy nến vừa đóng xong (nến -2), vì nến -1 là nến đang chạy
    candle = df.iloc[-2]
    candle_range = candle["high"] - candle["low"]
    if candle_range == 0:
        return False

    body = abs(candle["close"] - candle["open"])
    body_ratio = body / candle_range

    if body_ratio < min_body_ratio:
        return False

    if direction == LONG:
        return candle["close"] > candle["open"]
    elif direction == SHORT:
        return candle["close"] < candle["open"]
    
    return False


class SignalEngine:
    def __init__(self, config: dict):
        self.cfg = config

    def _attach_price_levels(
        self,
        result: Dict[str, Any],
        price: float,
        atr: float,
        direction: str,
        min_sl_distance: float = 0.0,
    ) -> Dict[str, Any]:
        # Tính SL dựa trên ATR hoặc khoảng cách tối thiểu
        sl_dist = max(atr * self.cfg["atr_sl_multiplier"], min_sl_distance)

        if direction == LONG:
            result["sl"] = price - sl_dist
            result["tp1"] = price + sl_dist * self.cfg["tp1_multiplier"]
            result["tp2"] = price + sl_dist * self.cfg["tp2_multiplier"]
            result["tp3"] = price + sl_dist * self.cfg["tp3_multiplier"]
        else:
            result["sl"] = price + sl_dist
            result["tp1"] = price - sl_dist * self.cfg["tp1_multiplier"]
            result["tp2"] = price - sl_dist * self.cfg["tp2_multiplier"]
            result["tp3"] = price - sl_dist * self.cfg["tp3_multiplier"]

        result["sl_distance"] = sl_dist
        return result

    def analyse(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all indicator calculations on the OHLCV dataframe.
        Optimized to use only the necessary tail of the dataframe.
        """
        # TỐI ƯU: Chỉ lấy 500 nến cuối cùng để tính toán cho nhanh
        # Vẫn đủ cho EMA 200 và các chỉ báo khác ổn định
        calc_df = df.tail(500).copy()
        
        close_ser = calc_df["close"]
        ema_fast = calculate_ema(close_ser, self.cfg["ema_fast"])
        ema_medium = calculate_ema(close_ser, self.cfg["ema_medium"])
        ema_slow = calculate_ema(close_ser, self.cfg["ema_slow"])
        rsi = calculate_rsi(close_ser, self.cfg["rsi_period"])
        atr = calculate_atr(calc_df, self.cfg["atr_period"])

        # Lấy giá trị nến hiện tại
        ef, em, es = float(ema_fast.iloc[-1]), float(ema_medium.iloc[-1]), float(ema_slow.iloc[-1])
        rsi_val, atr_val = float(rsi.iloc[-1]), float(atr.iloc[-1])
        current_close = float(close_ser.iloc[-1])

        alignment = get_ema_alignment(ef, em, es)
        slope = get_ema_slope(ema_fast)

        return {
            "ema_fast": ef,
            "ema_medium": em,
            "ema_slow": es,
            "rsi": rsi_val,
            "atr": atr_val,
            "close": current_close,
            "alignment": alignment,
            "slope": slope,
            "ema_fast_series": ema_fast,
            "ema_medium_series": ema_medium,
            "ema_slow_series": ema_slow,
            "df": calc_df,
        }

    def get_signal(self, df: pd.DataFrame, spread: float) -> Dict[str, Any]:
        """
        Run all entry filters. Returns signal dict with direction and price levels.
        """
        data = self.analyse(df)
        filters = {}

        # Filter 1: ATR activity (Volatility check)
        filters["atr_active"] = data["atr"] >= self.cfg["atr_min_threshold"]

        # Filter 2: Spread check
        filters["spread_ok"] = spread <= self.cfg["max_spread_points"]

        # Filter 3: EMA alignment (Trend direction)
        alignment = data["alignment"]
        filters["ema_aligned"] = alignment in (BULLISH, BEARISH)

        # Filter 4: EMA slope (Trend strength)
        slope = data["slope"]
        filters["slope_ok"] = abs(slope) >= self.cfg["ema_slope_min"]

        # Xác định hướng giao dịch
        direction = LONG if alignment == BULLISH else SHORT if alignment == BEARISH else NEUTRAL

        # Filter 5: Pullback (Sửa: sử dụng lookback từ config)
        filters["pullback"] = check_pullback(
            data["df"],
            data["ema_medium_series"],
            data["ema_slow_series"],
            alignment,
            lookback=self.cfg.get("pullback_lookback", 3)
        ) if direction != NEUTRAL else False

        # Filter 6: RSI zone
        if direction == LONG:
            filters["rsi_zone"] = self.cfg["rsi_long_min"] <= data["rsi"] <= self.cfg["rsi_long_max"]
        elif direction == SHORT:
            filters["rsi_zone"] = self.cfg["rsi_short_min"] <= data["rsi"] <= self.cfg["rsi_short_max"]
        else:
            filters["rsi_zone"] = False

        # Filter 7: Candle confirmation
        filters["candle_confirm"] = check_candle_confirmation(
            data["df"], direction, self.cfg["candle_body_min_ratio"]
        ) if direction != NEUTRAL else False

        # Kiểm tra tất cả filter có vượt qua không
        all_pass = all(filters.values())

        result = {
            "direction": direction if all_pass else NEUTRAL,
            "filters": filters,
            "all_pass": all_pass,
            # Loại bỏ các Series lớn khỏi kết quả trả về để tránh nặng memory
            **{k: v for k, v in data.items() if k not in ("ema_fast_series", "ema_medium_series", "ema_slow_series", "df")},
        }

        if all_pass and direction != NEUTRAL:
            self._attach_price_levels(result, data["close"], data["atr"], direction)

        return result

    def get_forced_signal(self, df: pd.DataFrame, spread: float, preferred_direction: str = "AUTO") -> Dict[str, Any]:
        """
        For paper-test: ignores filters and provides executable levels.
        """
        data = self.analyse(df)
        preferred_direction = (preferred_direction or "AUTO").upper()

        if preferred_direction in (LONG, SHORT):
            direction = preferred_direction
        elif data["alignment"] == BEARISH:
            direction = SHORT
        else:
            direction = LONG

        filters = {k: True for k in ["atr_active", "spread_ok", "ema_aligned", "slope_ok", "pullback", "rsi_zone", "candle_confirm"]}
        
        result = {
            "direction": direction,
            "filters": filters,
            "all_pass": True,
            "forced": True,
            **{k: v for k, v in data.items() if k not in ("ema_fast_series", "ema_medium_series", "ema_slow_series", "df")},
        }
        return self._attach_price_levels(
            result,
            data["close"],
            data["atr"],
            direction,
            self.cfg.get("forced_min_sl_distance", 0.5),
        )

    def check_ema_reversal(self, df: pd.DataFrame, trade_direction: str) -> bool:
        """True if EMA_Fast crosses EMA_Medium against the trade direction (Exit signal)."""
        # Tối ưu: Chỉ tính EMA trên một tập dữ liệu nhỏ
        calc_df = df.tail(100).copy()
        ema_fast = calculate_ema(calc_df["close"], self.cfg["ema_fast"])
        ema_medium = calculate_ema(calc_df["close"], self.cfg["ema_medium"])

        ef_now, em_now = float(ema_fast.iloc[-1]), float(ema_medium.iloc[-1])
        ef_prev, em_prev = float(ema_fast.iloc[-2]), float(ema_medium.iloc[-2])

        if trade_direction == LONG:
            return ef_prev >= em_prev and ef_now < em_now # Cross down
        elif trade_direction == SHORT:
            return ef_prev <= em_prev and ef_now > em_now # Cross up
        
        return False
