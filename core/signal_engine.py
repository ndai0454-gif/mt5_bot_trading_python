import pandas as pd
from typing import Optional, Dict, Any

# ==========================================
# INDICATOR CALCULATIONS
# ==========================================
def calculate_ema(close, period):
    return close.ewm(span=period, adjust=False).mean()

def calculate_rsi(close, period):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, period):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_adx(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    # Đơn giản hóa ADX cho Scalping: dùng độ biến động trung bình làm proxy cho sức mạnh xu hướng
    return atr.rolling(window=period).mean()

# Constants
BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"
LONG = "LONG"
SHORT = "SHORT"

# ==========================================
# HELPER FUNCTIONS (LOGIC BẺ KHÓA)
# ==========================================
def get_ema_alignment(ef, em, es, strict=True):
    """
    Nới lỏng điều kiện EMA: 
    - strict=True: Phải xếp chồng hoàn hảo Fast > Medium > Slow.
    - strict=False: Chỉ cần Fast > Medium là đủ để scalping ngắn.
    """
    if strict:
        if ef > em > es: return BULLISH
        if ef < em < es: return BEARISH
    else:
        if ef > em: return BULLISH
        if ef < em: return BEARISH
    return NEUTRAL

def check_engulfing(df, direction):
    if len(df) < 2:
        return False
    c1, o1 = df["close"].iloc[-2], df["open"].iloc[-2]
    c2, o2 = df["close"].iloc[-1], df["open"].iloc[-1]
    if direction == LONG:
        return (c2 > o2) and (o2 <= c1) and (c2 > o1)
    if direction == SHORT:
        return (c2 < o2) and (o2 >= c1) and (c2 < o1)
    return False

def check_candle_confirmation(df, direction, min_body_ratio=0.4):
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

# ==========================================
# MAIN SIGNAL ENGINE
# ==========================================
class SignalEngine:
    def __init__(self, config: dict):
        self.cfg = config

    def check_trend_alignment(self, df_h1: Optional[pd.DataFrame]) -> str:
        if df_h1 is None or len(df_h1) < 200:
            return NEUTRAL
        ema_200 = calculate_ema(df_h1["close"], self.cfg.get("ema_trend", 200))
        close = df_h1["close"].iloc[-1]
        return BULLISH if close > ema_200.iloc[-1] else BEARISH if close < ema_200.iloc[-1] else NEUTRAL

    def check_ema_reversal(self, df: pd.DataFrame, direction: str) -> bool:
        """
        Kiểm tra xem xu hướng ngắn hạn có bị đảo chiều không.
        Nếu đang LONG mà giá đóng cửa dưới EMA Fast -> Cảnh báo đảo chiều.
        """
        if df is None or len(df) < 2:
            return False
            
        close = df["close"].iloc[-1]
        ema_f = calculate_ema(df["close"], self.cfg["ema_fast"]).iloc[-1]
        
        if direction == LONG:
            # Nếu giá đóng cửa cắt xuống dưới EMA Fast -> Đảo chiều giảm
            return close < ema_f
        elif direction == SHORT:
            # Nếu giá đóng cửa cắt lên trên EMA Fast -> Đảo chiều tăng
            return close > ema_f
            
        return False

    def _attach_price_levels(self, result: dict, price: float, atr: float, direction: str, min_sl: float = 0.0):
        sl_dist = max(atr * self.cfg["atr_sl_multiplier"], min_sl)
        
        # Tỷ lệ R:R tối ưu cho Scalping
        tp1_m = 1.5 
        tp2_m = 3.0
        tp3_m = 5.0

        if direction == LONG:
            result.update({
                "sl": price - sl_dist,
                "tp1": price + sl_dist * tp1_m,
                "tp2": price + sl_dist * tp2_m,
                "tp3": price + sl_dist * tp3_m,
            })
        else:
            result.update({
                "sl": price + sl_dist,
                "tp1": price - sl_dist * tp1_m,
                "tp2": price - sl_dist * tp2_m,
                "tp3": price - sl_dist * tp3_m,
            })
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
        
        # Lấy cấu hình mức độ khắt khe của EMA từ config
        strict_ema = self.cfg.get("strict_ema_alignment", True)

        return {
            "ema_fast": ema_f.iloc[-1],
            "ema_medium": ema_m.iloc[-1],
            "ema_slow": ema_s.iloc[-1],
            "rsi": rsi.iloc[-1],
            "atr": atr.iloc[-1],
            "adx": adx.iloc[-1],
            "close": close.iloc[-1],
            "alignment": get_ema_alignment(ema_f.iloc[-1], ema_m.iloc[-1], ema_s.iloc[-1], strict=strict_ema),
            "df": calc_df,
            "ema_medium_series": ema_m,
            "ema_slow_series": ema_s,
        }

    def get_signal(self, df: pd.DataFrame, spread: float, df_h1: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        data = self.analyse(df)
        filters = {}
        h1_trend = self.check_trend_alignment(df_h1)

        # 1. Bộ lọc cơ bản
        filters["atr_active"] = data["atr"] >= self.cfg.get("atr_min_threshold", 0.0)
        filters["spread_ok"] = spread <= self.cfg.get("max_spread_points", 30)
        filters["ema_aligned"] = data["alignment"] in (BULLISH, BEARISH)
        filters["adx_strong"] = data["adx"] >= self.cfg.get("adx_threshold", 20)

        direction = LONG if data["alignment"] == BULLISH else SHORT if data["alignment"] == BEARISH else NEUTRAL

        # ---------------------------------------------------------------------
        # CƠ CHẾ BẺ KHÓA 1: NỚI LỎNG KHUNG H1
        # Nếu cấu hình strict_h1_trend = False HOẶC ADX cực mạnh (>25), cho phép vào lệnh ngược H1 (Săn sóng hồi)
        # ---------------------------------------------------------------------
        strict_h1 = self.cfg.get("strict_h1_trend", True)
        bypass_h1_by_adx = data["adx"] >= self.cfg.get("adx_bypass_h1_threshold", 25)

        if strict_h1 and not bypass_h1_by_adx:
            if df_h1 is not None:
                if (direction == LONG and h1_trend != BULLISH) or (direction == SHORT and h1_trend != BEARISH):
                    direction = NEUTRAL

        if direction != NEUTRAL:
            # -----------------------------------------------------------------
            # CƠ CHẾ BẺ KHÓA 2: NỚI LỎNG XÁC NHẬN NẾN
            # Nếu ADX > 25 (Trend mạnh), chấp nhận lệnh kể cả khi nến không phải Engulfing hoặc thân nhỏ
            # -----------------------------------------------------------------
            engulfing = check_engulfing(data["df"], direction)
            min_body = self.cfg.get("candle_body_min_ratio", 0.4)
            candle_confirm = check_candle_confirmation(data["df"], direction, min_body)
            
            if self.cfg.get("easy_candle_confirm_on_trend", True) and data["adx"] >= 25:
                filters["candle_confirmed"] = True # Bypass xác nhận nến khi trend cực mạnh
            else:
                filters["candle_confirmed"] = engulfing or candle_confirm

            # RSI Logic (Soft filter: chỉ chặn nếu quá cực đoan)
            if direction == LONG:
                filters["rsi_zone"] = data["rsi"] >= self.cfg.get("rsi_long_min", 30)
            else:
                filters["rsi_zone"] = data["rsi"] <= self.cfg.get("rsi_short_max", 70)
        else:
            filters["candle_confirmed"] = False
            filters["rsi_zone"] = False

        # Danh sách lọc bắt buộc để kích hoạt lệnh
        required_filters = ["atr_active", "spread_ok", "ema_aligned", "candle_confirmed"]
        all_pass = all(filters.get(f, False) for f in required_filters)

        result = {
            "direction": direction if all_pass else NEUTRAL,
            "filters": filters,
            "all_pass": all_pass,
            "h1_trend": h1_trend,
            **{k: v for k, v in data.items() if k not in ("ema_medium_series", "ema_slow_series", "df")},
        }

        if all_pass and direction != NEUTRAL:
            self._attach_price_levels(result, data["close"], data["atr"], direction)
        
        return result
