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
        if volume <= 0:
            return 0.0

        if self.direction == "LONG":
            delta = exit_price - self.entry_price
        else:
            delta = self.entry_price - exit_price

        # Tính PnL chuẩn theo Tick Value (Phù hợp cho Gold, Forex, Indices)
        if self.tick_size > 0 and self.tick_value > 0:
            return (delta / self.tick_size) * self.tick_value * volume

        return delta * volume * self.contract_size

    def book_close(self, volume: float, exit_price: float) -> float:
        """Ghi nhận kết quả đóng lệnh vào sổ sách."""
        volume = min(volume, self.lot_remaining)
        if volume <= 0:
            return 0.0

        pnl = self.pnl_for_volume(exit_price, volume)
        self.realized_pnl = round(self.realized_pnl + pnl, 2)
        self.closed_lots = round(self.closed_lots + volume, 4) # Tăng độ chính xác
        self.lot_remaining = round(max(0.0, self.lot_remaining - volume), 4)
        return pnl


class TradeManager:
    def __init__(self, config: dict, connector: Any, risk_manager: Any):
        self.cfg = config
        self.connector = connector
        self.risk_manager = risk_manager
        self._trades: List[ManagedTrade] = []
        # Lấy lot_step từ config (ví dụ: 0.01 cho Standard, 0.1 cho một số chỉ số)
        self.lot_step = self.cfg.get("lot_step", 0.01)

    def add_trade(self, trade: ManagedTrade):
        self._trades.append(trade)
        logger.info(f"Tracking trade: ticket={trade.ticket} {trade.direction} {trade.lot_total:.2f}L")

    def remove_trade(self, ticket: int):
        self._trades = [t for t in self._trades if t.ticket != ticket]

    def get_trades(self) -> List[ManagedTrade]:
        return list(self._trades)

    def trade_count(self) -> int:
        return len(self._trades)

    def _close_and_book(self, trade: ManagedTrade, volume: float, price: float) -> Optional[float]:
        """Thực hiện đóng lệnh trên sàn và ghi nhận PnL."""
        volume = min(volume, trade.lot_remaining)
        if volume <= 0:
            return None

        # Gửi lệnh đóng lên sàn
        res = self.connector.close_partial(trade.ticket, volume, trade.symbol, trade.direction)
        if not res:
            return None

        # Nếu connector trả về giá khớp thực tế (fill_price), ưu tiên dùng giá đó
        fill_price = res.get("fill_price", price) if isinstance(res, dict) else price

        pnl = trade.book_close(volume, fill_price)
        logger.info(
            "Closed %.2fL ticket=%s at %.2f | pnl=%+.2f realized=%+.2f remaining=%.2fL",
            volume, trade.ticket, fill_price, pnl, trade.realized_pnl, trade.lot_remaining,
        )
        return pnl

    def monitor(self, current_price: float, df=None, signal_engine=None, price_by_ticket: Optional[dict] = None) -> List[dict]:
        """
        Check all managed trades against current price.
        Returns list of events (partial closes, SL moves, reversals).
        """
        events = []
        price_by_ticket = price_by_ticket or {}
        for trade in list(self._trades):
            trade_price = price_by_ticket.get(trade.ticket, current_price)
            trade_events = self._check_trade(trade, trade_price, df, signal_engine)
            events.extend(trade_events)
        return events

    def _check_trade(self, trade: ManagedTrade, price: float, df: Any, signal_engine: Any) -> List[dict]:
        events = []
        direction = trade.direction

        # --- TP1 ---
        if not trade.tp1_hit and self._tp_hit(price, trade.tp1, direction):
            close_vol = self._partial_volume(trade, self.cfg["tp1_close_percent"])
            pnl = self._close_and_book(trade, close_vol, price)
            if pnl is not None:
                trade.tp1_hit = True
                logger.info(f"TP1 hit ticket={trade.ticket} closed={close_vol:.2f}L")
                events.append({"type": "TP1", "ticket": trade.ticket, "volume": close_vol, "pnl": pnl})

        # --- TP2 + Breakeven ---
        if trade.tp1_hit and not trade.tp2_hit and self._tp_hit(price, trade.tp2, direction):
            close_vol = self._partial_volume(trade, self.cfg["tp2_close_percent"])
            pnl = self._close_and_book(trade, close_vol, price)
            if pnl is not None:
                trade.tp2_hit = True
                logger.info(f"TP2 hit ticket={trade.ticket} closed={close_vol:.2f}L")
                events.append({"type": "TP2", "ticket": trade.ticket, "volume": close_vol, "pnl": pnl})

            if not trade.breakeven_set:
                if self.connector.modify_sl(trade.ticket, trade.entry_price):
                    trade.sl = trade.entry_price
                    trade.breakeven_set = True
                    logger.info(f"Breakeven set ticket={trade.ticket} SL={trade.entry_price:.2f}")
                    events.append({"type": "BREAKEVEN", "ticket": trade.ticket})

        # --- TP3 (Close All Remaining) ---
        if trade.tp2_hit and not trade.tp3_hit and self._tp_hit(price, trade.tp3, direction):
            if trade.lot_remaining > 0:
                close_vol = trade.lot_remaining
                pnl = self._close_and_book(trade, close_vol, price)
                if pnl is not None:
                    trade.tp3_hit = True
                    logger.info(f"TP3 hit ticket={trade.ticket} — fully closed")
                    events.append({"type": "TP3", "ticket": trade.ticket, "volume": close_vol, "pnl": pnl})

        # --- EMA Reversal Exit ---
        if df is not None and signal_engine is not None and not trade.tp3_hit:
            if signal_engine.check_ema_reversal(df, direction):
                if trade.lot_remaining > 0:
                    close_vol = trade.lot_remaining
                    pnl = self._close_and_book(trade, close_vol, price)
                    if pnl is not None:
                        logger.info(f"EMA reversal exit ticket={trade.ticket} price={price:.2f}")
                        events.append({"type": "EMA_REVERSAL", "ticket": trade.ticket, "volume": close_vol, "pnl": pnl})

        # Nếu lệnh đã đóng hết volume, thực hiện finalize (ghi nhận vào RiskManager)
        if trade.lot_remaining <= 0:
            self._finalize_trade(trade, price)

        return events

    def _tp_hit(self, price: float, tp: float, direction: str) -> bool:
        if direction == "LONG":
            return price >= tp
        return price <= tp

    def _partial_volume(self, trade: ManagedTrade, percent: int) -> float:
        """Tính toán volume chốt lời theo % lot ban đầu, làm tròn theo lot_step."""
        # Tính toán dựa trên lot tổng, làm tròn xuống theo lot_step
        raw_vol = (trade.lot_total * percent / 100)
        vol = math.floor(raw_vol / self.lot_step) * self.lot_step
        
        # Nếu volume tính ra quá nhỏ (< step), nhưng vẫn còn lot, thì chốt tối thiểu 1 step
        if vol <= 0 and trade.lot_remaining > 0:
            vol = self.lot_step
            
        # Đảm bảo không đóng quá số lot còn lại và làm tròn 4 số thập phân
        return min(round(vol, 4), trade.lot_remaining)

    def _finalize_trade(self, trade: ManagedTrade, exit_price: float):
        """Kết thúc theo dõi lệnh và gửi dữ liệu sang RiskManager."""
        if trade.lot_remaining > 0:
            trade.book_close(trade.lot_remaining, exit_price)

        pnl = trade.realized_pnl
        self.risk_manager.record_trade_result(
            pnl,
            direction=trade.direction,
            entry=trade.entry_price,
            exit_price=exit_price,
            lots=trade.lot_total,
        )
        self.remove_trade(trade.ticket)

    def handle_sl_hit(self, ticket: int, exit_price: float):
        """Xử lý khi StopLoss được hit (thường gọi từ Event của Broker)."""
        for trade in list(self._trades):
            if trade.ticket == ticket:
                self._finalize_trade(trade, exit_price)
                break

    def close_all(self):
        """Đóng khẩn cấp toàn bộ các lệnh đang quản lý."""
        for trade in list(self._trades):
            if trade.lot_remaining > 0:
                tick = self.connector.get_current_price(trade.symbol)
                exit_price = (tick["bid"] + tick["ask"]) / 2 if tick else trade.entry_price
                close_vol = trade.lot_remaining
                pnl = self._close_and_book(trade, close_vol, exit_price)
                if pnl is not None:
                    logger.info(f"Emergency close: ticket={trade.ticket}")
                    self._finalize_trade(trade, exit_price)
