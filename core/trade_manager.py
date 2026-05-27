import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

@dataclass
class ManagedTrade:
    ticket: int
    symbol: str
    direction: str
    entry_price: float
    lot_total: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    tick_size: float = 0.01
    tick_value: float = 1.0
    contract_size: float = 100.0
    opened_at: float = field(default_factory=time.time)

    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    breakeven_set: bool = False
    realized_pnl: float = 0.0
    closed_lots: float = 0.0

    lot_remaining: float = field(init=False)

    def __post_init__(self):
        self.lot_remaining = self.lot_total

    def unrealized_pnl(self, current_price: float) -> float:
        return self.realized_pnl + self.pnl_for_volume(current_price, self.lot_remaining)

    def pnl_for_volume(self, exit_price: float, volume: float) -> float:
        if volume <= 0: return 0.0
        delta = (exit_price - self.entry_price) if self.direction == "LONG" else (self.entry_price - exit_price)
        if self.tick_size > 0 and self.tick_value > 0:
            return (delta / self.tick_size) * self.tick_value * volume
        return delta * volume * self.contract_size

    def book_close(self, volume: float, exit_price: float) -> float:
        volume = min(volume, self.lot_remaining)
        if volume <= 0: return 0.0
        pnl = self.pnl_for_volume(exit_price, volume)
        self.realized_pnl = round(self.realized_pnl + pnl, 2)
        self.closed_lots = round(self.closed_lots + volume, 4) 
        self.lot_remaining = round(max(0.0, self.lot_remaining - volume), 4)
        return pnl

class TradeManager:
    def __init__(self, config: dict, connector: Any, risk_manager: Any):
        self.cfg = config
        self.connector = connector
        self.risk_manager = risk_manager
        self._trades: List[ManagedTrade] = []
        self.lot_step = self.cfg.get("lot_step", 0.01)

    def add_trade(self, trade: ManagedTrade):
        self._trades.append(trade)
        logger.info(f"Tracking trade: ticket={trade.ticket} {trade.direction} {trade.lot_total:.2f}L")
        
        if self.cfg.get("dca_positive_enabled", False):
            self._protect_previous_trades(trade)

    def _protect_previous_trades(self, new_trade: ManagedTrade):
        """Dời SL của các lệnh cũ lên điểm vào của lệnh mới để khóa lợi nhuận (Lock Profit)"""
        for t in self._trades:
            if t.ticket != new_trade.ticket and t.direction == new_trade.direction:
                if (new_trade.direction == "LONG" and new_trade.entry_price > t.sl) or \
                   (new_trade.direction == "SHORT" and new_trade.entry_price < t.sl):
                    
                    if self.connector.modify_sl(t.ticket, new_trade.entry_price):
                        t.sl = new_trade.entry_price
                        logger.info(f"🛡️ Protected trade {t.ticket}: SL moved to {new_trade.entry_price:.2f}")

    def can_open_dca_trade(self, direction: str, current_price: float) -> bool:
        """Kiểm tra xem có đủ điều kiện nhồi lệnh DCA Dương hay không."""
        if not self.cfg.get("dca_positive_enabled", False):
            return False

        trades_same_dir = [t for t in self._trades if t.direction == direction]
        
        if len(trades_same_dir) >= self.cfg.get("dca_max_layers", 5):
            return False

        if not trades_same_dir:
            return True 

        last_trade = trades_same_dir[-1]
        required_dist = self.cfg.get("dca_positive_distance_pips", 20) * 0.01 
        
        if direction == "LONG" and current_price >= last_trade.entry_price + required_dist:
            return True
        if direction == "SHORT" and current_price <= last_trade.entry_price - required_dist:
            return True

        return False

    def remove_trade(self, ticket: int):
        self._trades = [t for t in self._trades if t.ticket != ticket]

    def get_trades(self) -> List[ManagedTrade]:
        """Trả về danh sách tất cả các lệnh đang mở"""
        return list(self._trades)

    def get_trade(self, ticket: int) -> Optional[ManagedTrade]:
        """
        Lấy thông tin chi tiết của một lệnh cụ thể qua ticket.
        Trả về đối tượng ManagedTrade nếu tìm thấy, ngược lại trả về None.
        """
        for trade in self._trades:
            if trade.ticket == ticket:
                return trade
        return None

    def trade_count(self) -> int:
        return len(self._trades)

    def _close_and_book(self, trade: ManagedTrade, volume: float, price: float) -> Optional[float]:
        volume = min(volume, trade.lot_remaining)
        if volume <= 0: return None
        res = self.connector.close_partial(trade.ticket, volume, trade.symbol, trade.direction)
        if not res: return None
        fill_price = res.get("fill_price", price) if isinstance(res, dict) else price
        pnl = trade.book_close(volume, fill_price)
        logger.info(f"Closed {volume:.2f}L ticket={trade.ticket} at {fill_price:.2f} | pnl={pnl:.2f}")
        return pnl

    def monitor(self, current_price: float, price_high: float, price_low: float, df=None, signal_engine=None) -> List[dict]:
        events = []
        for trade in list(self._trades):
            trade_events = self._check_trade(trade, current_price, price_high, price_low, df, signal_engine)
            events.extend(trade_events)
        return events

    def _check_trade(self, trade: ManagedTrade, price: float, phigh: float, plow: float, df: Any, signal_engine: Any) -> List[dict]:
        events = []
        direction = trade.direction

        # 1. STOP LOSS
        sl_hit = (direction == "LONG" and plow <= trade.sl) or (direction == "SHORT" and phigh >= trade.sl)
        if sl_hit:
            close_vol = trade.lot_remaining
            pnl = self._close_and_book(trade, close_vol, trade.sl)
            if pnl is not None:
                events.append({"type": "SL", "ticket": trade.ticket, "volume": close_vol, "pnl": pnl})
                self._finalize_trade(trade, trade.sl)
                return events

        # 2. TAKE PROFIT
        tp_trigger_price = price
        if direction == "LONG":
            if phigh >= trade.tp1 and not trade.tp1_hit: tp_trigger_price = trade.tp1
            elif phigh >= trade.tp2 and not trade.tp2_hit: tp_trigger_price = trade.tp2
            elif phigh >= trade.tp3 and not trade.tp3_hit: tp_trigger_price = trade.tp3
        else:
            if plow <= trade.tp1 and not trade.tp1_hit: tp_trigger_price = trade.tp1
            elif plow <= trade.tp2 and not trade.tp2_hit: tp_trigger_price = trade.tp2
            elif plow <= trade.tp3 and not trade.tp3_hit: tp_trigger_price = trade.tp3

        # TP1
        if not trade.tp1_hit and self._tp_hit(tp_trigger_price, trade.tp1, direction):
            close_vol = self._partial_volume(trade, self.cfg.get("tp1_close_percent", 30))
            pnl = self._close_and_book(trade, close_vol, tp_trigger_price)
            if pnl is not None:
                trade.tp1_hit = True
                events.append({"type": "TP1", "ticket": trade.ticket, "volume": close_vol, "pnl": pnl})

        # TP2 + Breakeven
        if trade.tp1_hit and not trade.tp2_hit and self._tp_hit(tp_trigger_price, trade.tp2, direction):
            close_vol = self._partial_volume(trade, self.cfg.get("tp2_close_percent", 30))
            pnl = self._close_and_book(trade, close_vol, tp_trigger_price)
            if pnl is not None:
                trade.tp2_hit = True
                if not trade.breakeven_set:
                    if self.connector.modify_sl(trade.ticket, trade.entry_price):
                        trade.sl = trade.entry_price
                        trade.breakeven_set = True
                        events.append({"type": "BREAKEVEN", "ticket": trade.ticket})
                events.append({"type": "TP2", "ticket": trade.ticket, "volume": close_vol, "pnl": pnl})

        # TP3
        if trade.tp2_hit and not trade.tp3_hit and self._tp_hit(tp_trigger_price, trade.tp3, direction):
            if trade.lot_remaining > 0:
                close_vol = trade.lot_remaining
                pnl = self._close_and_book(trade, close_vol, tp_trigger_price)
                if pnl is not None:
                    trade.tp3_hit = True
                    events.append({"type": "TP3", "ticket": trade.ticket, "volume": close_vol, "pnl": pnl})

        # 3. EMA REVERSAL
        if df is not None and signal_engine is not None and not trade.tp3_hit:
            if signal_engine.check_ema_reversal(df, direction):
                if trade.lot_remaining > 0:
                    close_vol = trade.lot_remaining
                    pnl = self._close_and_book(trade, close_vol, price)
                    if pnl is not None:
                        events.append({"type": "EMA_REVERSAL", "ticket": trade.ticket, "volume": close_vol, "pnl": pnl})

        if trade.lot_remaining <= 0:
            self._finalize_trade(trade, price)

        return events

    def _tp_hit(self, price: float, tp: float, direction: str) -> bool:
        return (price >= tp) if direction == "LONG" else (price <= tp)

    def _partial_volume(self, trade: ManagedTrade, percent: int) -> float:
        raw_vol = (trade.lot_total * percent / 100)
        vol = math.floor(raw_vol / self.lot_step) * self.lot_step
        if vol <= 0 and trade.lot_remaining > 0: vol = self.lot_step
        return min(round(vol, 4), trade.lot_remaining)

    def _finalize_trade(self, trade: ManagedTrade, exit_price: float):
        if trade.lot_remaining > 0:
            trade.book_close(trade.lot_remaining, exit_price)
        self.risk_manager.record_trade_result(
            trade.realized_pnl, 
            direction=trade.direction, 
            entry=trade.entry_price, 
            exit_price=exit_price, 
            lots=trade.lot_total
        )
        self.remove_trade(trade.ticket)

    def close_all(self):
        for trade in list(self._trades):
            if trade.lot_remaining > 0:
                tick = self.connector.get_current_price(trade.symbol)
                exit_price = (tick["bid"] + tick["ask"]) / 2 if tick else trade.entry_price
                self._close_and_book(trade, trade.lot_remaining, exit_price)
                self._finalize_trade(trade, exit_price)
