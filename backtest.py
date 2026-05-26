import MetaTrader5 as mt5  # SỬA LỖI: Phải là MetaTrader5
import numpy as np
import pandas as pd
import csv
import json
import logging
from datetime import datetime
from core.signal_engine import SignalEngine
from core.trade_manager import TradeManager, ManagedTrade
from core.risk_manager import RiskManager
from core.mt5_connector import MT5Connector
from core.session_filter import is_trading_session

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backtest")

# --- CẤU HÌNH BACKTEST ---
SYMBOL = "XAUUSD"
NUM_BARS_M5 = 320000    
NUM_BARS_H1 = 15000      
INITIAL_BALANCE = 10000.0

def run_backtest():
    # 1. Khởi tạo Connector và MT5
    connector = MT5Connector(paper_mode=True)
    if not connector.connect():
        print("❌ MT5 Initialize failed!")
        return

    # Đảm bảo Symbol sẵn sàng
    if not connector.ensure_symbol(SYMBOL):
        print(f"❌ Symbol {SYMBOL} not available.")
        return

    # Load config
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ Không tìm thấy file config.json")
        return

    # Khởi tạo các Engine
    signal_engine = SignalEngine(config)
    risk_manager = RiskManager(config)
    trade_manager = TradeManager(config, connector, risk_manager)

    # 2. Tải dữ liệu
    print(f"📥 Downloading 3 years of data for {SYMBOL}...")
    df_m5 = connector.get_ohlcv(SYMBOL, "M5", count=NUM_BARS_M5)
    df_h1 = connector.get_ohlcv(SYMBOL, "H1", count=NUM_BARS_H1)
    
    if df_m5 is None or df_h1 is None:
        print("❌ Failed to get data from MT5. Please check 'Max bars in chart' in MT5 settings.")
        return

    # HIỂN THỊ KHOẢNG THỜI GIAN BACKTEST
    start_date = df_m5.index[0]
    end_date = df_m5.index[-1]
    print(f"📅 Backtest Period: {start_date}  --->  {end_date}")

    # Lấy thông tin tick để tính Lot size chính xác
    symbol_info = connector.get_symbol_info(SYMBOL)
    tick_value = symbol_info['trade_tick_value']
    tick_size = symbol_info['trade_tick_size']
    vol_step = symbol_info['volume_step']
    vol_min = symbol_info['volume_min']

    # Thiết lập vốn ban đầu
    risk_manager.set_session_balance(INITIAL_BALANCE)
    
    # Mở file kết quả
    with open("backtest_results.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Ticket", "Time", "Direction", "Entry", "SL", "TP1", "Result", "PnL", "H1_Trend"])

        total_bars = len(df_m5)
        print(f"⚙️ Simulation started with {total_bars} bars...")
        
        # --- VÒNG LẶP BACKTEST CHÍNH ---
        for i in range(200, total_bars):
            # Thông báo tiến độ
            if i % 10000 == 0:
                progress = (i / total_bars) * 100
                print(f"⏳ Progress: {progress:.1f}% | Current Time: {df_m5.index[i]}")

            row = df_m5.iloc[i]
            current_time = row.name 
            price_close = row['close']
            price_high = row['high']
            price_low = row['low']
            
            sessions = config.get("sessions", [])

            # ---------------------------------------------------------------------
            # BƯỚC 1: THEO DÕI VÀ ĐÓNG LỆNH (SL/TP chạy 24/5)
            # ---------------------------------------------------------------------
            active_trades = trade_manager.get_trades()
            for t in list(active_trades):
                closed = False
                res_pnl = 0

                if t.direction == "LONG":
                    if price_low <= t.sl: 
                        res_pnl = t.pnl_for_volume(t.sl, t.lot_remaining)
                        closed = True
                    elif price_high >= t.tp1: 
                        res_pnl = t.pnl_for_volume(t.tp1, t.lot_remaining)
                        closed = True
                elif t.direction == "SHORT":
                    if price_high >= t.sl: 
                        res_pnl = t.pnl_for_volume(t.sl, t.lot_remaining)
                        closed = True
                    elif price_low <= t.tp1: 
                        res_pnl = t.pnl_for_volume(t.tp1, t.lot_remaining)
                        closed = True

                if closed:
                    # GHI CSV CHỈ KHI ĐÓNG LỆNH
                    writer.writerow([
                        t.ticket, 
                        current_time, 
                        t.direction, 
                        t.entry_price, 
                        t.sl, 
                        t.tp1, 
                        "WIN" if res_pnl > 0 else "LOSS", 
                        round(res_pnl, 2), 
                        getattr(t, 'h1_trend', 'UNKNOWN')
                    ])
                    trade_manager.remove_trade(t.ticket)
                    risk_manager.record_trade_result(res_pnl)

            # ---------------------------------------------------------------------
            # BƯỚC 2: TÌM TÍN HIỆU VÀO LỆNH MỚI (Chỉ trong session)
            # ---------------------------------------------------------------------
            if is_trading_session(current_time, sessions):
                window_m5 = df_m5.iloc[max(0, i-500):i+1]
                current_h1_window = df_h1[df_h1.index <= current_time].tail(200)
                spread = connector.get_current_spread(SYMBOL) or 20 
                
                signal = signal_engine.get_signal(window_m5, spread, current_h1_window)
                
                if signal["direction"] != "NEUTRAL":
                    direction = signal["direction"]
                    sl_dist = abs(price_close - signal["sl"])
                    h1_trend = signal.get("h1_trend", "UNKNOWN")
                    
                    # Sử dụng RiskManager để tính Lot size động theo Trend
                    current_balance = risk_manager.get_current_balance()
                    lot = risk_manager.calculate_lot_size(
                        balance=current_balance, 
                        sl_distance=sl_dist, 
                        trend=h1_trend, 
                        tick_value=tick_value,
                        volume_step=vol_step,
                        tick_size=tick_size,
                        volume_min=vol_min
                    )
                    
                    trade = ManagedTrade(
                        ticket=i, symbol=SYMBOL, direction=direction,
                        entry_price=price_close, lot_total=lot,
                        sl=signal["sl"], tp1=signal["tp1"], tp2=signal["tp2"], tp3=signal["tp3"],
                        tick_size=tick_size, tick_value=tick_value
                    )
                    trade.h1_trend = h1_trend
                    trade_manager.add_trade(trade)
                    # TUYỆT ĐỐI KHÔNG GHI CSV "OPEN" Ở ĐÂY

    # 5. Tổng kết kết quả
    stats = risk_manager.get_daily_stats()
    print("\n" + "="*30 + "\n 🏆 BACKTEST COMPLETED \n" + "="*30)
    print(f"📅 Period: {df_m5.index[0]} to {df_m5.index[-1]}")
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Win Rate: {stats['win_rate']:.2f}%")
    print(f"Total PnL: ${stats['total_pnl']:.2f}")
    print(f"Best Trade: ${stats['best_trade']:.2f}")
    print(f"Worst Trade: ${stats['worst_trade']:.2f}")
    print(f"Detailed data saved to: backtest_results.csv")

if __name__ == "__main__":
    try:
        run_backtest()
    finally:
        mt5.shutdown()
