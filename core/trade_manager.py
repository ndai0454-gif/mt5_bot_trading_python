import logging
import math
import time
from typing import List, Dict, Any, Optional

# core/trade_manager.py
class ManagedTrade:
    def __init__(self, ticket, symbol, direction, entry_price, lot_total, sl, initial_sl, risk_distance, tp1, tp2, tp3, tick_size, tick_value, group_id=None):
        self.ticket = ticket
        self.group_id = group_id if group_id else ticket # Nếu không có group_id, dùng chính ticket làm id nhóm
        self.symbol = symbol
        self.direction = direction
        self.entry_price = entry_price
        self.lot_total = lot_total
        self.lot_remaining = lot_total
        self.sl = sl
        self.initial_sl = initial_sl
        self.risk_distance = risk_distance
        self.tp1 = tp1
        self.tp2 = tp2
        self.tp3 = tp3
        self.tick_size = tick_size
        self.tick_value = tick_value
        self.tp1_hit = False
        self.h1_trend = "UNKNOWN"
        self.market_regime = "TREND"

    def pnl_for_volume(self, current_price, volume):
        if volume <= 0: return 0.0
        diff = current_price - self.entry_price if self.direction == "LONG" else self.entry_price - current_price
        return diff * volume * (self.tick_value / self.tick_size)

class TradeManager:
    def __init__(self, config, connector, risk_manager):
        self.config = config
        self.connector = connector
        self.risk_manager = risk_manager
        self.active_trades = {}
        self.group_pnl = {} # Lưu trữ PnL tích lũy theo group_id: {group_id: total_pnl}
        self.logger = logging.getLogger("TradeManager")

    def add_trade(self, trade: ManagedTrade):
        self.active_trades[trade.ticket] = trade
        # Khởi tạo PnL cho nhóm nếu là lệnh đầu tiên
        if trade.group_id not in self.group_pnl:
            self.group_pnl[trade.group_id] = 0.0
        self.logger.info(f"Trade Added: #{trade.ticket} | Group: {trade.group_id} | {trade.direction}")

    def remove_trade(self, ticket):
        if ticket in self.active_trades:
            del self.active_trades[ticket]

    def get_trade(self, ticket):
        return self.active_trades.get(ticket)

    def get_trades(self):
        return list(self.active_trades.values())

    def trade_count(self):
        return len(self.active_trades)

    def can_open_dca_trade(self, direction, current_price):
        trades = [t for t in self.active_trades.values() if t.direction == direction]
        if not trades: return False
        last_trade = trades[-1]
        dist = abs(current_price - last_trade.entry_price)
        return dist > (last_trade.risk_distance * 0.5)

    def monitor(self, price_close, price_high, price_low, df, signal_engine):
        events = []
        for ticket, trade in list(self.active_trades.items()):
            event = self._check_trade(trade, price_close, price_high, price_low, df, signal_engine)
            if event:
                # Cập nhật PnL vào nhóm
                self.group_pnl[trade.group_id] += event['pnl']
                events.append(event)

                # CHỈ xóa trade nếu không phải là chốt lời một phần (PARTIAL_TP)
                if event.get('type') != "PARTIAL_TP":
                    self.remove_trade(ticket)
        
        return events

    def _check_trade(self, trade, current_price, phigh, plow, df, signal_engine):
        # --- BƯỚC 1: KIỂM TRA STOP LOSS (SL) ---
        if (trade.direction == "LONG" and plow <= trade.sl) or \
           (trade.direction == "SHORT" and phigh >= trade.sl):
            return {"ticket": trade.ticket, "group_id": trade.group_id, "type": "SL",
                    "pnl": trade.pnl_for_volume(trade.sl, trade.lot_remaining)}

        # --- BƯỚC 2 & 3: CHỐT TP1 VÀ DỜI BE+ LẬP TỨC ---
        tp1_val = getattr(trade, 'tp1', 0)
        if tp1_val != 0 and not trade.tp1_hit:
            # Kiểm tra xem giá đã chạm TP1 chưa
            if (trade.direction == "LONG" and phigh >= tp1_val) or \
               (trade.direction == "SHORT" and plow <= tp1_val):

                # A. Chốt lúa: Đóng 80% Volume
                closed_vol = trade.lot_total * 0.8
                pnl_partial = trade.pnl_for_volume(tp1_val, closed_vol)
                trade.lot_remaining -= closed_vol

                # B. Dời SL về BE+ (Entry + Phí + 1 tick)
                # be_offset đảm bảo không bị âm tiền phí sàn
                be_offset = trade.tick_size * 5 # Tăng offset lên 5 ticks cho an toàn
                if trade.direction == "LONG":
                    trade.sl = trade.entry_price + be_offset
                else:
                    trade.sl = trade.entry_price - be_offset

                trade.tp1_hit = True
                self.logger.info(f"🎯 TP1 HIT #{trade.ticket}: Closed 80% | SL moved to BE+")

                # Trả về event chốt lời một phần (không xóa trade khỏi active_trades)
                return {"ticket": trade.ticket, "group_id": trade.group_id, "type": "PARTIAL_TP", "pnl": pnl_partial, "keep_trade": True}

        # --- BƯỚC 4: KIỂM TRA ĐẢO CHIỀU EMA (EXIT SỚM) ---
        if signal_engine.check_ema_reversal(df, trade.direction):
            return {"ticket": trade.ticket, "group_id": trade.group_id, "type": "REVERSAL",
                    "pnl": trade.pnl_for_volume(current_price, trade.lot_remaining)}

        return None

