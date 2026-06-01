import MetaTrader5 as mt5 
import numpy as np
import pandas as pd
import csv
import json
import logging
from tqdm import tqdm
from datetime import datetime
from core.signal_engine import SignalEngine, NEUTRAL, TREND, SIDEWAY, LONG, SHORT
from core.trade_manager import TradeManager
from core.risk_manager import RiskManager
from core.mt5_connector import MT5Connector

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SidewayBacktest")

# --- CẤU HÌNH BACKTEST ---
SYMBOL = "XAUUSD"
INITIAL_BALANCE = 10000.0
DATE_FROM = datetime(2022, 1, 31)
DATE_TO = datetime(2026, 5, 31)

def run_sideway_backtest():
    connector = MT5Connector(paper_mode=True)
    if not connector.connect():
        print("❌ MT5 Initialize failed!")
        return

    if not connector.ensure_symbol(SYMBOL):
        print(f"❌ Symbol {SYMBOL} not available.")
        return

    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ Không tìm thấy file config.json")
        return

    signal_engine = SignalEngine(config)
    risk_manager = RiskManager(config)
    trade_manager = TradeManager(config, connector, risk_manager)

    print(f"📥 Downloading data for {SYMBOL} from {DATE_FROM} to {DATE_TO}...")
    rates_m5 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M5, DATE_FROM, DATE_TO)
    rates_h1 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H1, DATE_FROM, DATE_TO)
    rates_h4 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H4, DATE_FROM, DATE_TO)

    if rates_m5 is None or rates_h1 is None or rates_h4 is None:
        print("❌ Failed to get data from MT5.")
        return

    df_m5 = pd.DataFrame(rates_m5)
    df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
    df_m5.set_index('time', inplace=True)

    df_h1 = pd.DataFrame(rates_h1)
    df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
    df_h1.set_index('time', inplace=True)

    df_h4 = pd.DataFrame(rates_h4)
    df_h4['time'] = pd.to_datetime(df_h4['time'], unit='s')
    df_h4.set_index('time', inplace=True)

    risk_manager.set_session_balance(INITIAL_BALANCE)
    
    results_file = "sideway_backtest_results.csv"
    with open(results_file, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Ticket", "Time", "Direction", "Entry", "SL", "TP1", "Result", "PnL", "Regime"])

        total_bars = len(df_m5)
        print(f"⚙️ Testing Sideway Strategy with {total_bars} bars...")
        
        # TỐI ƯU: Tính toán regime và ATR cho toàn bộ dữ liệu trước khi vào vòng lặp
        df_m5['regime'] = signal_engine.calculate_all_regimes(df_m5)
        
        # Tính ATR 14 cho toàn bộ dữ liệu để tránh tính lại trong vòng lặp
        df_m5['atr'] = (df_m5["high"].rolling(14).max() - df_m5["low"].rolling(14).min())
        
        # Biến thống kê
        stats = {"total_trades": 0, "wins": 0, "total_pnl": 0.0, "max_dd": 0.0, "balance_history": [INITIAL_BALANCE]}
        current_balance = INITIAL_BALANCE

        # Sử dụng tqdm để theo dõi tiến độ
        for i in tqdm(range(200, total_bars), desc="Backtesting Professional Sideway Strategy"):
            current_time = df_m5.index[i]
            
            # 1. Lọc tin tức (Sử dụng ATR)
            atr_val = df_m5['atr'].iloc[i]
            candle_range = df_m5["high"].iloc[i] - df_m5["low"].iloc[i]
            if candle_range > atr_val * 3:
                continue 

            # 2. XÁC NHẬN ĐA KHUNG THỜI GIAN (4H -> 1H -> 5M)
            try:
                df_h4_slice = df_h4.loc[:current_time].tail(100)
                df_h1_slice = df_h1.loc[:current_time].tail(100)
            except Exception:
                continue

            if len(df_h4_slice) < 50 or len(df_h1_slice) < 50:
                continue

            # --- NÂNG CẤP: Nới lỏng vùng Cluster 4H để tăng số lượng cơ hội ---
            res_high_zone, res_low_zone = signal_engine.calculate_price_clusters(df_h4_slice)
            if res_high_zone is None or res_low_zone is None:
                continue
                
            price_now = df_m5["close"].iloc[i]
            
            # Nới lỏng: Cho phép giá tiệm cận vùng biên (thêm 0.1% buffer)
            buffer = price_now * 0.001 
            at_upper_4h = price_now >= (res_high_zone[1] - buffer)
            at_lower_4h = price_now <= (res_low_zone[0] + buffer)

            if not (at_upper_4h or at_lower_4h):
                continue

            # Kiểm tra mô hình trên 1H (Wedge hoặc Channel)
            is_wedge = signal_engine.detect_wedge_pattern(df_h1_slice)
            is_channel = signal_engine.detect_channel_pattern(df_h1_slice)
            
            # Nới lỏng: Nếu không có Wedge/Channel nhưng ADX cực thấp (< 20) thì vẫn coi là Sideway mạnh
            adx_h1 = signal_engine.calculate_adx(df_h1_slice).iloc[-1] if hasattr(signal_engine, 'calculate_adx') else 25
            if not (is_wedge or is_channel or adx_h1 < 20):
                continue

            # 3. ĐIỂM KÍCH HOẠT (TRIGGER) & MOMENTUM
            df_m5_slice = df_m5.iloc[max(0, i-30) : i+1]
            signal = signal_engine.check_sideway_signals(df_m5_slice)
            
            if not signal:
                continue

            # Lọc nến Momentum: Điều chỉnh hệ số nhân xuống 1.5 để bắt được nhiều sóng hơn
            if not signal_engine.detect_momentum_candle(df_m5_slice, multiplier=1.5):
                continue

            entry_price = df_m5["close"].iloc[i]
            
            # 4. QUẢN LÝ RỦI RO DỰA TRÊN CẤU TRÚC (SL/TP)
            # SL đặt trên đỉnh/đáy của mô hình 1H
            h1_recent_high = df_h1_slice["high"].max()
            h1_recent_low = df_h1_slice["low"].min()
            
            if signal == LONG:
                sl = h1_recent_low - (entry_price * 0.0002)
                # TP1: Chốt lời sớm tại 75% khoảng cách đến biên đối diện
                tp = entry_price + (res_high_zone[0] - entry_price) * 0.75
            else: # SHORT
                sl = h1_recent_high + (entry_price * 0.0002)
                tp = entry_price - (entry_price - res_low_zone[1]) * 0.75
            
            sl_dist = abs(entry_price - sl)
            tp_dist = abs(entry_price - tp)
            
            # Mô phỏng kết quả với Break-even
            trade_result = "OPEN"
            pnl = 0.0
            be_reached = False
            
            for j in range(i + 1, min(i + 500, total_bars)):
                high_j = df_m5["high"].iloc[j]
                low_j = df_m5["low"].iloc[j]
                
                if signal == LONG:
                    if low_j <= sl:
                        trade_result = "LOSS"
                        pnl = -sl_dist
                        break
                    if high_j >= tp:
                        trade_result = "WIN"
                        pnl = tp_dist
                        break
                    if not be_reached and high_j >= (entry_price + sl_dist):
                        be_reached = True
                        sl = entry_price 
                else: # SHORT
                    if high_j >= sl:
                        trade_result = "LOSS"
                        pnl = -sl_dist
                        break
                    if low_j <= tp:
                        trade_result = "WIN"
                        pnl = tp_dist
                        break
                    if not be_reached and low_j <= (entry_price - sl_dist):
                        be_reached = True
                        sl = entry_price
            
            if trade_result == "OPEN": trade_result = "EXPIRED"

            stats["total_trades"] += 1
            if trade_result == "WIN": stats["wins"] += 1
            
            trade_pnl_usd = (pnl / sl_dist) * (INITIAL_BALANCE * 0.01) if sl_dist != 0 else 0
            stats["total_pnl"] += trade_pnl_usd
            current_balance += trade_pnl_usd
            stats["balance_history"].append(current_balance)
            
            writer.writerow([
                f"S-{stats['total_trades']}", current_time, signal, entry_price, 
                round(sl, 2), round(tp, 2), trade_result, round(trade_pnl_usd, 2), "SIDEWAY_ADV"
            ])

        # Tính Max Drawdown
        peak = stats["balance_history"][0]
        max_dd = 0
        for b in stats["balance_history"]:
            if b > peak: peak = b
            dd = (peak - b) / peak * 100
            if dd > max_dd: max_dd = dd
        stats["max_dd"] = max_dd

    # IN KẾT QUẢ CUỐI CÙNG
    win_rate = (stats["wins"] / stats["total_trades"] * 100) if stats["total_trades"] > 0 else 0
    print("\n" + "="*30)
    print(" 🏆 BACKTEST COMPLETED ")
    print("="*30)
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total PnL: ${stats['total_pnl']:.2f}")
    print(f"Max Drawdown: {stats['max_dd']:.2f}%")
    print(f"Detailed data saved to: {results_file}")
    print("="*30)

if __name__ == "__main__":
    run_sideway_backtest()
