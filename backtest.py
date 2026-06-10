# backtest.py
import MetaTrader5 as mt5 
import numpy as np
import pandas as pd
import csv
import json
import logging
from datetime import datetime
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
# Dates will be loaded from config.json

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
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("❌ Không tìm thấy file config.json")
        return

    # Khởi tạo các Engine
    signal_engine = SignalEngine(config)
    risk_manager = RiskManager(config)
    trade_manager = TradeManager(config, connector, risk_manager)

    # ---------------------------------------------------------------------
    # 2. TẢI DỮ LIỆU THEO KHOẢNG THỜI GIAN
    # ---------------------------------------------------------------------
    date_from_str = config.get("backtest_date_from", "2024-05-27")
    date_to_str = config.get("backtest_date_to", "2026-05-27")
    
    date_from = datetime.strptime(date_from_str, "%Y-%m-%d")
    date_to = datetime.strptime(date_to_str, "%Y-%m-%d")

    print(f"📥 Downloading data for {SYMBOL} from {date_from} to {date_to}...")
    
    rates_m5 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_M5, date_from, date_to)
    rates_h1 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H1, date_from, date_to)

    if rates_m5 is None or rates_h1 is None:
        print("❌ Failed to get data from MT5. Please check if the terminal is connected.")
        return

    # Gán tên cột chuẩn
    columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume', 'spread', 'real_volume']
    
    df_m5 = pd.DataFrame(rates_m5, columns=columns)
    df_m5['time'] = pd.to_datetime(df_m5['time'], unit='s')
    df_m5.set_index('time', inplace=True)

    df_h1 = pd.DataFrame(rates_h1, columns=columns)
    df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
    df_h1.set_index('time', inplace=True)

    start_date = df_m5.index[0]
    end_date = df_m5.index[-1]
    print(f"📅 Backtest Period: {start_date}  --->  {end_date}")

    # Lấy thông tin symbol
    symbol_info = connector.get_symbol_info(SYMBOL)
    tick_value = symbol_info['trade_tick_value']
    tick_size = symbol_info['trade_tick_size']
    vol_step = symbol_info['volume_step']
    vol_min = symbol_info['volume_min']
    vol_max = symbol_info['volume_max']

    risk_manager.set_session_balance(INITIAL_BALANCE)
    
    with open("backtest_results.csv", "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Ticket", "Time", "Direction", "Entry", "SL", "TP1", "Result", "PnL", "H1_Trend", "Regime"])

        total_bars = len(df_m5)
        print(f"⚙️ Simulation started with {total_bars} bars...")
        
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
            # BƯỚC 1: QUẢN LÝ LỆNH (MONITORING) - ĐÃ CẬP NHẬT
            # ---------------------------------------------------------------------
            window_m5 = df_m5.iloc[max(0, i-500):i+1]
            events = trade_manager.monitor(price_close, price_high, price_low, window_m5, signal_engine)
            
            for ev in events:
                res_pnl = ev.get('pnl', 0)
                ticket = ev.get('ticket')
                trade_info = trade_manager.get_trade(ticket) 
                
                # Xác định regime tại thời điểm đóng/chốt một phần
                regime = "TREND" if abs(price_close - df_m5.iloc[max(0, i-20)]['close']) > (tick_size * 100) else "SIDEWAY"
                h1_trend = getattr(trade_info, 'h1_trend', 'UNKNOWN') if trade_info else 'UNKNOWN'
                
                # Ghi kết quả vào CSV (Sử dụng type của event: SL, PARTIAL_TP, REVERSAL)
                writer.writerow([
                    ticket, current_time, 
                    trade_info.direction if trade_info else "N/A", 
                    trade_info.entry_price if trade_info else "N/A", 
                    trade_info.sl if trade_info else "N/A", 
                    trade_info.tp1 if trade_info else "N/A", 
                    f"{ev.get('type')}" if res_pnl > 0 else f"LOSS_{ev.get('type')}", 
                    round(res_pnl, 2), 
                    h1_trend, 
                    regime
                ])
                
                # Ghi nhận PnL theo nhóm (Group ID)
                risk_manager.record_group_result(ev.get('group_id'), res_pnl)

            # ---------------------------------------------------------------------
            # BƯỚC 2: TÌM TÍN HIỆU VÀO LỆNH
            # ---------------------------------------------------------------------
            if is_trading_session(current_time, sessions):
                current_h1_window = df_h1[df_h1.index <= current_time].tail(200)
                spread = connector.get_current_spread(SYMBOL) or 20 
                
                signal = signal_engine.get_signal(window_m5, spread, current_h1_window)
                
                if signal["direction"] != NEUTRAL:
                    direction = signal["direction"]
                    h1_trend = signal.get("h1_trend", "UNKNOWN")
                    
                    if trade_manager.trade_count() == 0 or trade_manager.can_open_dca_trade(direction, price_close):
                        sl_dist = abs(price_close - signal["sl"])
                        current_balance = risk_manager.get_current_balance()
                        
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
                        
                        existing_trades = [t for t in trade_manager.get_trades() if t.direction == direction]
                        if len(existing_trades) > 0:
                            lot = round(lot * config.get("dca_lot_multiplier", 1.0), 2)

                        if lot > 0:
                            # FIX LỖI TÊN THAM SỐ: tp1_price -> tp1
                            trade = ManagedTrade(
                                ticket=i, symbol=SYMBOL, direction=direction,
                                entry_price=price_close, lot_total=lot,
                                sl=signal["sl"], initial_sl=signal["sl"], risk_distance=sl_dist,
                                tp1=signal["tp1"], 
                                tp2=signal.get("tp2", signal["tp1"]), 
                                tp3=signal.get("tp3", signal["tp1"]),
                                tick_size=tick_size, tick_value=tick_value
                            )
                            trade.h1_trend = h1_trend 
                            trade_manager.add_trade(trade)
                            
                            writer.writerow([
                                trade.ticket, current_time, direction, price_close, 
                                signal["sl"], signal["tp1"], "OPEN", 0, h1_trend, signal.get("market_regime", "TREND")
                            ])

        # ĐÓNG LỆNH CUỐI KỲ
        final_price = df_m5.iloc[-1]['close']
        remaining = trade_manager.get_trades()
        for t in list(remaining):
            pnl = t.pnl_for_volume(final_price, t.lot_remaining)
            writer.writerow([
                t.ticket, df_m5.index[-1], t.direction, t.entry_price, 
                t.sl, t.tp1, "FORCE_CLOSE", round(pnl, 2), getattr(t, 'h1_trend', 'UNKNOWN'), "FINAL_CLOSE"
            ])
            risk_manager.record_group_result(t.group_id, pnl)
            trade_manager.remove_trade(t.ticket)

    stats = risk_manager.get_daily_stats()
    print("\n" + "="*30 + "\n 🏆 BACKTEST COMPLETED \n" + "="*30)
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Win Rate: {stats['win_rate']:.2f}%")
    print(f"Total PnL: ${stats['total_pnl']:.2f}")
    print(f"Detailed data saved to: backtest_results.csv")

if __name__ == "__main__":
    try:
        run_backtest()
    finally:
        mt5.shutdown()
