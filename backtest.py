# backtest.py
import MetaTrader5 as mt5 
import numpy as np
import pandas as pd
import csv
import json
import logging
import math
from datetime import datetime # Chỉ sử dụng 1 cách import duy nhất để tránh lỗi AttributeError
from core.signal_engine import SignalEngine, NEUTRAL
from core.trade_manager import TradeManager, ManagedTrade
from core.risk_manager import RiskManager
from core.mt5_connector import MT5Connector
from core.session_filter import is_trading_session

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Backtest")

# --- CẤU HÌNH BACKTEST ---
SYMBOL = "XAUUSD"
INITIAL_BALANCE = 10000.0
# Định nghĩa khoảng thời gian backtest (Sửa lỗi: dùng datetime(...) thay vì datetime.datetime(...))
DATE_FROM = datetime(2024, 5, 27)
DATE_TO = datetime(2026, 5, 27)

def run_backtest():
    # 1. Khởi tạo Connector và MT5
    connector = MT5Connector(paper_mode=True)
    if not connector.connect():
        print("❌ MT5 Initialize failed!")
        return

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

    # ---------------------------------------------------------------------
    # 2. TẢI DỮ LIỆU THEO KHOẢNG THỜI GIAN (DATE RANGE)
    # ---------------------------------------------------------------------
    print(f"📥 Downloading data for {SYMBOL} from {DATE_FROM} to {DATE_TO}...")
    
    # Lấy dữ liệu M5 từ MT5
    rates_m5 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M5, DATE_FROM, DATE_TO)
    # Lấy dữ liệu H1 từ MT5
    rates_h1 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H1, DATE_FROM, DATE_TO)

    if rates_m5 is None or rates_h1 is None:
        print("❌ Failed to get data from MT5. Please check if the terminal is connected.")
        return

    # Chuyển sang DataFrame và thiết lập Index là thời gian cho M5
    df_m5 = pd.DataFrame(rates_m5)
    df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
    df_m5.set_index('time', inplace=True)

    # Chuyển sang DataFrame và thiết lập Index là thời gian cho H1
    df_h1 = pd.DataFrame(rates_h1)
    df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
    df_h1.set_index('time', inplace=True)

    # HIỂN THỊ KHOẢNG THỜI GIAN THỰC TẾ LẤY ĐƯỢC
    start_date = df_m5.index[0]
    end_date = df_m5.index[-1]
    print(f"📅 Backtest Period: {start_date}  --->  {end_date}")

    # Lấy thông tin symbol để tính toán lot/tick
    symbol_info = connector.get_symbol_info(SYMBOL)
    tick_value = symbol_info['trade_tick_value']
    tick_size = symbol_info['trade_tick_size']
    vol_step = symbol_info['volume_step']
    vol_min = symbol_info['volume_min']
    vol_max = symbol_info['volume_max']

    risk_manager.set_session_balance(INITIAL_BALANCE)
    
    # Mở file kết quả
    with open("backtest_results.csv", "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # Header chi tiết
        writer.writerow(["Ticket", "Time", "Direction", "Entry", "SL", "TP1", "Result", "PnL", "H1_Trend", "Regime"])

        total_bars = len(df_m5)
        print(f"⚙️ Simulation started with {total_bars} bars...")
        
        # Bắt đầu vòng lặp mô phỏng (bỏ qua 200 nến đầu để tính toán chỉ báo)
        for i in range(200, total_bars):
            if i % 10000 == 0:
                progress = (i / total_bars) * 100
                print(f"⏳ Progress: {progress:.1f}% | Time: {df_m5.index[i]}")

            row = df_m5.iloc[i]
            current_time = row.name 
            price_close = row['close']
            price_high = row['high']
            price_low = row['low']
            
            sessions = config.get("sessions", [])

            # ---------------------------------------------------------------------
            # BƯỚC 1: QUẢN LÝ LỆNH (MONITORING)
            # ---------------------------------------------------------------------
            window_m5 = df_m5.iloc[max(0, i-500):i+1]
            
            # TradeManager kiểm tra đóng lệnh (SL, TP hoặc tín hiệu đảo chiều)
            events = trade_manager.monitor(price_close, price_high, price_low, window_m5, signal_engine)
            
            for ev in events:
                res_pnl = ev.get('pnl', 0)
                ticket = ev.get('ticket')
                
                # Truy xuất thông tin trade trước khi bị xóa
                trade_info = trade_manager.get_trade(ticket) 
                
                # Tính toán Market Regime đơn giản tại thời điểm đóng
                regime = "TREND" if abs(price_close - df_m5.iloc[max(0, i-20)]['close']) > (tick_size * 100) else "SIDEWAY"
                h1_trend = getattr(trade_info, 'h1_trend', 'UNKNOWN') if trade_info else 'UNKNOWN'
                
                writer.writerow([
                    ticket, 
                    current_time, 
                    trade_info.direction if trade_info else "N/A", 
                    trade_info.entry_price if trade_info else "N/A", 
                    trade_info.sl if trade_info else "N/A", 
                    trade_info.tp1 if trade_info else "N/A", 
                    "WIN" if res_pnl > 0 else "LOSS", 
                    round(res_pnl, 2), 
                    h1_trend, 
                    regime
                ])
                
                # Cập nhật vốn vào RiskManager
                risk_manager.record_trade_result(res_pnl)

            # ---------------------------------------------------------------------
            # BƯỚC 2: TÌM TÍN HIỆU VÀO LỆNH
            # ---------------------------------------------------------------------
            if is_trading_session(current_time, sessions):
                # Lấy dữ liệu H1 tương ứng với thời điểm hiện tại (không nhìn trước tương lai)
                current_h1_window = df_h1[df_h1.index <= current_time].tail(200)
                spread = connector.get_current_spread(SYMBOL) or 20 
                
                signal = signal_engine.get_signal(window_m5, spread, current_h1_window)
                
                if signal["direction"] != NEUTRAL:
                    direction = signal["direction"]
                    h1_trend = signal.get("h1_trend", "UNKNOWN")
                    
                    # Kiểm tra điều kiện DCA Dương (Nhồi lệnh)
                    if trade_manager.trade_count() == 0 or trade_manager.can_open_dca_trade(direction, price_close):
                        
                        sl_dist = abs(price_close - signal["sl"])
                        current_balance = risk_manager.get_current_balance()
                        
                        # Tính toán khối lượng lệnh
                        lot = risk_manager.calculate_lot_size(
                            balance=current_balance, 
                            sl_distance=sl_dist, 
                            trend=h1_trend, 
                            tick_value=tick_value, 
                            volume_step=vol_step, 
                            tick_size=tick_size, 
                            volume_min=vol_min, 
                            volume_max=vol_max
                        )
                        
                        # Áp dụng Lot Multiplier cho lệnh DCA
                        existing_trades = [t for t in trade_manager.get_trades() if t.direction == direction]
                        if len(existing_trades) > 0:
                            lot = round(lot * config.get("dca_lot_multiplier", 1.0), 2)

                        if lot > 0:
                            trade = ManagedTrade(
                                ticket=i, symbol=SYMBOL, direction=direction,
                                entry_price=price_close, lot_total=lot,
                                sl=signal["sl"], tp1=signal["tp1"], tp2=signal["tp2"], tp3=signal["tp3"],
                                tick_size=tick_size, tick_value=tick_value
                            )
                            trade.h1_trend = h1_trend # Lưu trend để track kết quả
                            
                            trade_manager.add_trade(trade)
                            
                            # Ghi nhận lệnh MỞ vào CSV
                            writer.writerow([
                                trade.ticket, current_time, direction, price_close, 
                                signal["sl"], signal["tp1"], "OPEN", 0, h1_trend, signal.get("market_regime", "TREND")
                            ])

        # ĐÓNG CÁC LỆNH CÒN SÓT TẠI CUỐI KỲ BACKTEST
        final_price = df_m5.iloc[-1]['close']
        remaining = trade_manager.get_trades()
        for t in list(remaining):
            pnl = t.pnl_for_volume(final_price, t.lot_remaining)
            writer.writerow([
                t.ticket, df_m5.index[-1], t.direction, t.entry_price, 
                t.sl, t.tp1, "FORCE_CLOSE", round(pnl, 2), getattr(t, 'h1_trend', 'UNKNOWN'), "FINAL_CLOSE"
            ])
            risk_manager.record_trade_result(pnl)
            trade_manager.remove_trade(t.ticket)

    # 3. Kết quả thống kê cuối cùng
    stats = risk_manager.get_daily_stats()
    print("\n" + "="*30 + "\n 🏆 BACKTEST COMPLETED \n" + "="*30)
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Win Rate: {stats['win_rate']:.2f}%")
    print(f"Total PnL: ${stats['total_pnl']:.2f}")
    print(f"Max Drawdown: {stats.get('current_drawdown', 0):.2f}%")
    print(f"Detailed data saved to: backtest_results.csv")

if __name__ == "__main__":
    try:
        run_backtest()
    finally:
        mt5.shutdown()
