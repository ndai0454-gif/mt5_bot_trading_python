import pandas as pd
from typing import Optional, Dict, Any
import MetaTrader5 as mt5
import numpy as np

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
    """Tính toán ADX chuẩn để đo sức mạnh xu hướng"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    # True Range
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # +DM và -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
    
    # Smoothing (Wilder's Smoothing approximation)
    smooth_tr = tr.rolling(window=period).sum()
    smooth_plus_dm = plus_dm.rolling(window=period).sum()
    smooth_minus_dm = minus_dm.rolling(window=period).sum()
    
    plus_di = 100 * (smooth_plus_dm / smooth_tr)
    minus_di = 100 * (smooth_minus_dm / smooth_tr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1e-9)
    adx = dx.rolling(window=period).mean()
    return adx

def calculate_price_clusters(df, window=100, cluster_size=0.001):
    """
    Xác định các vùng giá (Clusters) nơi giá phản ứng nhiều lần.
    Trả về: (upper_zone, lower_zone) - Mỗi zone là một tuple (top, bottom)
    """
    if len(df) < window: return None, None
    
    recent = df.tail(window)
    highs = recent["high"]
    lows = recent["low"]
    
    def find_cluster_zone(series):
        counts, bins = np.histogram(series, bins=20)
        max_idx = np.argmax(counts)
        mid = (bins[max_idx] + bins[max_idx+1]) / 2
        half_width = (bins[max_idx+1] - bins[max_idx]) / 2
        # Trả về vùng biên rộng hơn một chút để bao quát phản ứng giá
        return (mid + half_width * 1.2, mid - half_width * 1.2)

    res_high_zone = find_cluster_zone(highs)
    res_low_zone = find_cluster_zone(lows)
    
    return res_high_zone, res_low_zone

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """Tính toán dải Bollinger Bands"""
    sma = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    bandwidth = (upper_band - lower_band) / sma
    return upper_band, lower_band, bandwidth

def calculate_ema_slope(df, period=20, lookback=5):
    """Tính độ dốc EMA: (EMA_now - EMA_prev) / lookback"""
    ema = calculate_ema(df["close"], period)
    if len(ema) < lookback: return 0
    slope = (ema.iloc[-1] - ema.iloc[-lookback]) / lookback
    return slope

def detect_market_structure(df, window=20):
    """
    Phân tích cấu trúc thị trường chuyên sâu:
    - TREND: Tạo Higher Highs & Higher Lows (Tăng) hoặc Lower Highs & Lower Lows (Giảm)
    - SIDEWAY: Khi giá ngừng tạo đỉnh/đáy mới (Flat structure)
    """
    if len(df) < window * 3: return SIDEWAY
    
    # Lấy 3 cụm nến để so sánh sự phát triển của cấu trúc
    recent = df.iloc[-window:]
    mid = df.iloc[-window*2 : -window]
    previous = df.iloc[-window*3 : -window*2]
    
    curr_max, curr_min = recent["high"].max(), recent["low"].min()
    mid_max, mid_min = mid["high"].max(), mid["low"].min()
    prev_max, prev_min = previous["high"].max(), previous["low"].min()
    
    # Xu hướng tăng: Đỉnh sau cao hơn đỉnh trước VÀ đáy sau cao hơn đáy trước
    if curr_max > mid_max > prev_max and curr_min > mid_min > prev_min:
        return TREND # Bullish Trend
    # Xu hướng giảm: Đỉnh sau thấp hơn đỉnh trước VÀ đáy sau thấp hơn đáy trước
    if curr_max < mid_max < prev_max and curr_min < mid_min < prev_min:
        return TREND # Bearish Trend
        
    return SIDEWAY

def detect_momentum_candle(df, window=10, multiplier=2.0):
    """
    Nhận diện nến Momentum: Thân nến lớn hơn multiplier lần trung bình 10 nến trước.
    Dùng để xác nhận phá vỡ cấu trúc hoặc hủy lệnh Mean Reversion.
    """
    if len(df) < window + 1: return False
    
    bodies = (df["close"] - df["open"]).abs()
    avg_body = bodies.iloc[-window-1:-1].mean()
    current_body = bodies.iloc[-1]
    
    return current_body > (avg_body * multiplier)

def detect_wedge_pattern(df, window=15):
    """
    Tìm kiếm sự hội tụ giá (Wedge):
    - Đỉnh sau thấp hơn đỉnh trước (Highs falling)
    - Đáy sau cao hơn đáy trước (Lows rising)
    """
    if len(df) < window: return False
    
    # Lấy các đỉnh/đáy cục bộ
    highs = df["high"].rolling(window=3).max()
    lows = df["low"].rolling(window=3).min()
    
    # Kiểm tra xu hướng hội tụ trong window
    is_highs_falling = highs.iloc[-1] < highs.iloc[-window]
    is_lows_rising = lows.iloc[-1] > lows.iloc[-window]
    
    return is_highs_falling and is_lows_rising

def detect_channel_pattern(df, window=15):
    """
    Tìm kiếm kênh giá (Channel):
    - Đỉnh và đáy di chuyển song song (cùng hướng)
    """
    if len(df) < window: return False
    
    highs = df["high"].rolling(window=3).max()
    lows = df["low"].rolling(window=3).min()
    
    # Kênh tăng: Cả đỉnh và đáy đều tăng
    is_bullish_channel = highs.iloc[-1] > highs.iloc[-window] and lows.iloc[-1] > lows.iloc[-window]
    # Kênh giảm: Cả đỉnh và đáy đều giảm
    is_bearish_channel = highs.iloc[-1] < highs.iloc[-window] and lows.iloc[-1] < lows.iloc[-window]
    
    return is_bullish_channel or is_bearish_channel

def identify_range_bound(df, lookback=20):
    """Xác định vùng biên cao nhất và thấp nhất trong khoảng thời gian gần đây"""
    recent_high = df["high"].rolling(window=lookback).max().iloc[-1]
    recent_low = df["low"].rolling(window=lookback).min().iloc[-1]
    range_size = recent_high - recent_low
    return recent_high, recent_low, range_size

# Constants
BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"
LONG = "LONG"
SHORT = "SHORT"
TREND = "TREND"
SIDEWAY = "SIDEWAY"

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

def check_pinbar(df):
    """Kiểm tra nến Pinbar (râu dài, thân nhỏ)"""
    if len(df) < 1: return False
    candle = df.iloc[-1]
    body = abs(candle["close"] - candle["open"])
    total_range = candle["high"] - candle["low"]
    if total_range == 0: return False
    
    # Pinbar tăng: râu dưới dài, thân nằm ở trên
    is_bull_pin = (candle["close"] - candle["low"]) > (total_range * 0.6) and body < (total_range * 0.3)
    # Pinbar giảm: râu trên dài, thân nằm ở dưới
    is_bear_pin = (candle["high"] - candle["close"]) > (total_range * 0.6) and body < (total_range * 0.3)
    
    return "LONG" if is_bull_pin else "SHORT" if is_bear_pin else None

# Thêm hàm tính phân kỳ RSI
def check_rsi_divergence(df, direction):
    if len(df) < 20: return False
    close = df["close"]
    rsi = calculate_rsi(close, 14)
    
    # Tìm 2 đáy/đỉnh gần nhất của giá và RSI
    # Đơn giản hóa: So sánh nến hiện tại với nến cách đây 5-10 phiên
    price_now = close.iloc[-1]
    rsi_now = rsi.iloc[-1]
    
    price_prev = close.iloc[-10:-1].min() if direction == LONG else close.iloc[-10:-1].max()
    rsi_prev = rsi.iloc[-10:-1].min() if direction == LONG else rsi.iloc[-10:-1].max()
    
    if direction == LONG:
        return (price_now < price_prev) and (rsi_now > rsi_prev)
    else:
        return (price_now > price_prev) and (rsi_now < rsi_prev)

# ==========================================
# MAIN SIGNAL ENGINE
# ==========================================
class SignalEngine:
    def __init__(self, config: dict):
        self.cfg = config

    def detect_momentum_candle(self, df, window=10, multiplier=2.0):
        """
        Nhận diện nến Momentum: Thân nến lớn hơn multiplier lần trung bình 10 nến trước.
        """
        return detect_momentum_candle(df, window, multiplier)

    def detect_wedge_pattern(self, df, window=15):
        """
        Tìm kiếm sự hội tụ giá (Wedge).
        """
        return detect_wedge_pattern(df, window)

    def detect_channel_pattern(self, df, window=15):
        """
        Tìm kiếm kênh giá (Channel).
        """
        return detect_channel_pattern(df, window)

    def calculate_price_clusters(self, df, window=100, cluster_size=0.001):
        """
        Xác định các vùng giá (Clusters) nơi giá phản ứng nhiều lần.
        """
        return calculate_price_clusters(df, window, cluster_size)

    def calculate_all_regimes(self, df: pd.DataFrame) -> pd.Series:
        """
        Nâng cấp Giai đoạn 3: Xác định Sideway dựa trên Cấu trúc & Cụm giá
        """
        # 1. ADX & BB Squeeze (Nền tảng)
        adx = calculate_adx(df)
        upper, lower, bandwidth = calculate_bollinger_bands(df)
        avg_bandwidth = bandwidth.rolling(window=100).mean()
        is_squeeze = bandwidth < avg_bandwidth
        
        # 2. Market Structure Analysis (Đỉnh/Đáy)
        structure_regimes = []
        for i in range(len(df)):
            if i < 40:
                structure_regimes.append(TREND)
            else:
                structure_regimes.append(detect_market_structure(df.iloc[:i+1]))
        
        regimes_series = pd.Series(structure_regimes, index=df.index)
        
        # 3. Cluster Stability (Kiểm tra xem giá có đang tôn trọng vùng cụm không)
        # Chỉ tính cho nến hiện tại trong loop, nhưng ở đây ta vectorize bằng cách 
        # kiểm tra xem range hiện tại có nằm trong vùng cluster lịch sử không
        # (Đơn giản hóa cho vectorized: dùng độ lệch chuẩn thấp)
        volatility = df["close"].rolling(window=20).std()
        is_stable = volatility < (df["close"].mean() * 0.002) # Biến động < 0.2% giá
        
        # LOGIC KẾT HỢP:
        # Phải là SIDEWAY về cấu trúc VÀ (ADX thấp HOẶC BB Squeeze) VÀ Biến động ổn định
        is_sideway = (regimes_series == SIDEWAY) & ((adx < 25) | is_squeeze) & is_stable
        
        final_regimes = pd.Series(TREND, index=df.index)
        final_regimes[is_sideway] = SIDEWAY
        
        return final_regimes

    def get_market_regime(self, df: pd.DataFrame) -> str:
        """
        Phân tích đa yếu tố để xác định thị trường TREND hay SIDEWAY
        """
        if len(df) < 30:
            return TREND

        # 1. ADX Check (Sức mạnh xu hướng)
        adx_series = calculate_adx(df)
        current_adx = adx_series.iloc[-1]
        
        # 2. EMA Slope Check (Độ dốc)
        slope = calculate_ema_slope(df)
        
        # 3. Range Bound Check (Vùng biên)
        _, _, r_size = identify_range_bound(df)
        
        # LOGIC QUYẾT ĐỊNH:
        # Ngưỡng ADX chuẩn là 25. Dưới 25 thường là sideway.
        is_adx_low = current_adx < 25 if not pd.isna(current_adx) else False
        
        # Độ dốc phẳng: so sánh với 10% độ lệch chuẩn của giá để chuẩn hóa mọi cặp tiền
        volatility = df["close"].std()
        is_slope_flat = abs(slope) < (volatility * 0.1) if not pd.isna(volatility) else False
        
        if is_adx_low and is_slope_flat:
            return SIDEWAY
        
        return TREND

    def check_sideway_signals(self, df: pd.DataFrame) -> Optional[str]:
        """
        Nâng cấp Logic Vào Lệnh:
        - Lọc Momentum cực mạnh (Hủy Mean Reversion)
        - Yêu cầu xác nhận Wedge/Channel
        - Yêu cầu RSI Divergence hoặc nến đảo chiều mạnh
        """
        if len(df) < 30: return None
        
        # 1. Lọc Momentum: Nếu nến hiện tại quá mạnh -> Breakout -> Hủy Mean Reversion
        if detect_momentum_candle(df):
            return None
        
        high_b, low_b, r_size = identify_range_bound(df)
        close = df["close"].iloc[-1]
        rsi = calculate_rsi(df["close"], 14).iloc[-1]
        
        # Kiểm tra mô hình hội tụ (Wedge)
        is_wedge = detect_wedge_pattern(df)
        
        # --- PHƯƠNG ÁN A: MEAN REVERSION (Đánh biên) ---
        # Buy: Chạm hỗ trợ + (RSI < 30 hoặc Nến đảo chiều) + (Bắt buộc Wedge hoặc RSI cực thấp)
        if close <= low_b * 1.001:
            pin = check_pinbar(df)
            # Thắt chặt: Phải có Wedge HOẶC RSI cực đoan (< 20)
            if (rsi < 30 or pin == "LONG" or check_engulfing(df, LONG)) and (is_wedge or rsi < 20):
                return LONG
                
        # Sell: Chạm kháng cự + (RSI > 70 hoặc Nến đảo chiều) + (Bắt buộc Wedge hoặc RSI cực cao)
        if close >= high_b * 0.999:
            pin = check_pinbar(df)
            if (rsi > 70 or pin == "SHORT" or check_engulfing(df, SHORT)) and (is_wedge or rsi > 80):
                return SHORT

        return None

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
