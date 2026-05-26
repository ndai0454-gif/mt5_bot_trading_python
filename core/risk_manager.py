import math
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

@dataclass
class TradeRecord:
    pnl: float
    direction: str
    entry: float
    exit: float
    lots: float

class RiskManager:
    def __init__(self, config: dict):
        # --- CẤU HÌNH RỦI RO ĐỘNG (Tối ưu 3) ---
        # Lấy các mức % rủi ro từ config, nếu không có thì dùng giá trị mặc định
        self.risk_percent_default = config.get("risk_percent_default", config.get("risk_percent", 1.0)) / 100.0
        self.risk_percent_bullish = config.get("risk_percent_bullish", 1.5) / 100.0
        self.risk_percent_bearish = config.get("risk_percent_bearish", 1.0) / 100.0
        
        # --- CHỐT CHẶN BẢO VỆ VỐN ---
        self.max_daily_loss_pct = config.get("max_daily_loss_percent", 5.0) / 100.0
        self.max_consecutive_losses = config.get("max_consecutive_losses", 5)
        self.max_concurrent_positions = config.get("max_concurrent_positions", 3)
        self.max_equity_drawdown_pct = config.get("max_equity_drawdown_percent", 10.0) / 100.0
        self.max_lot_size = config.get("max_lot_size", 1.0)

        # --- BIẾN THEO DÕI TRẠNG THÁI ---
        self._daily_loss = 0.0
        self._daily_profit = 0.0
        self._consecutive_losses = 0
        self._session_start_balance = None
        self._trade_records: List[TradeRecord] = []

    def set_session_balance(self, balance: float):
        """Thiết lập số dư đầu ngày để tính toán drawdown."""
        if self._session_start_balance is None:
            self._session_start_balance = balance

    def get_current_balance(self) -> float:
        """Trả về số dư hiện tại (Vốn ban đầu + PnL thực tế)."""
        if self._session_start_balance is None:
            return 10000.0 # Fallback
        return self._session_start_balance + self.daily_pnl

    def check_equity_hard_stop(self, current_equity: float) -> Tuple[bool, str]:
        """
        Kiểm tra sụt giảm vốn (Equity Drawdown).
        Trả về (False, reason) nếu chạm ngưỡng ngắt khẩn cấp.
        """
        if self._session_start_balance is None:
            return True, ""
            
        if current_equity < 0:
            return False, "Equity is negative!"

        # Tính % sụt giảm so với vốn đầu ngày
        drawdown = (self._session_start_balance - current_equity) / self._session_start_balance
        if drawdown >= self.max_equity_drawdown_pct:
            return False, f"Equity Hard Stop: Drawdown {drawdown*100:.1f}% exceeds limit!"
            
        return True, ""

    def calculate_lot_size(
        self,
        balance: float,
        sl_distance: float,
        trend: Optional[str] = None, # Tối ưu 3: Nhận diện xu hướng H1
        tick_value: float = 1.0,
        volume_step: float = 0.01,
        tick_size: float = 0.01,
        volume_min: float = 0.01,
        volume_max: Optional[float] = None,
    ) -> float:
        """
        Tính toán lot size dựa trên rủi ro % động theo xu hướng và khoảng cách SL.
        """
        if sl_distance <= 0 or tick_value <= 0 or tick_size <= 0 or volume_step <= 0:
            return max(volume_min, volume_step)

        # --- LỰA CHỌN % RỦI RO DỰA TRÊN TREND (Tối ưu 3) ---
        if trend == "BULLISH":
            risk_pct = self.risk_percent_bullish
        elif trend == "BEARISH":
            risk_pct = self.risk_percent_bearish
        else:
            risk_pct = self.risk_percent_default

        risk_amount = balance * risk_pct
        
        # Tính số tiền rủi ro trên 1 lot
        # Ví dụ Gold: SL 2.0 giá, tick_size 0.01 -> 200 ticks. 200 * tick_value = số tiền lỗ/lot
        loss_per_lot = (sl_distance / tick_size) * tick_value
        if loss_per_lot <= 0:
            return max(volume_min, volume_step)

        raw_lot = risk_amount / loss_per_lot
        
        # Làm tròn xuống theo volume_step của sàn (ví dụ 0.01)
        lot = math.floor(raw_lot / volume_step) * volume_step
        lot = max(volume_min, lot)
        
        # Giới hạn lot tối đa
        if volume_max:
            lot = min(volume_max, lot)
        lot = min(lot, self.max_lot_size)

        # Làm tròn cuối cùng để tránh lỗi số thập phân của Python
        step_text = f"{volume_step:.8f}".rstrip("0")
        step_decimals = len(step_text.split(".")[1]) if "." in step_text else 0
        lot = round(lot, step_decimals)

        return lot

    def check_daily_limits(self) -> Tuple[bool, str]:
        """Kiểm tra giới hạn lỗ ngày và số lệnh thua liên tiếp."""
        if self._session_start_balance and self._daily_loss > 0:
            daily_loss_pct = self._daily_loss / self._session_start_balance
            if daily_loss_pct >= self.max_daily_loss_pct:
                return False, f"Daily loss limit reached: {daily_loss_pct*100:.1f}% >= {self.max_daily_loss_pct*100:.1f}%"

        if self._consecutive_losses >= self.max_consecutive_losses:
            return False, f"Consecutive loss limit reached: {self._consecutive_losses}"

        return True, ""

    def record_trade_result(self, pnl: float, direction: str = "",
                             entry: float = 0.0, exit_price: float = 0.0,
                             lots: float = 0.0):
        """Ghi nhận kết quả giao dịch vào sổ sách."""
        record = TradeRecord(pnl=pnl, direction=direction,
                             entry=entry, exit=exit_price, lots=lots)
        self._trade_records.append(record)

        if pnl < 0:
            self._daily_loss += abs(pnl)
            self._consecutive_losses += 1
        else:
            self._daily_profit += pnl
            self._consecutive_losses = 0

    def reset_daily(self):
        """Reset các thông số theo ngày."""
        self._daily_loss = 0.0
        self._daily_profit = 0.0
        self._consecutive_losses = 0
        self._session_start_balance = None
        self._trade_records.clear()

    def get_daily_stats(self) -> dict:
        """Trả về thống kê giao dịch trong ngày."""
        records = self._trade_records
        wins   = [r for r in records if r.pnl >= 0]
        losses = [r for r in records if r.pnl < 0]
        total_pnl = self._daily_profit - self._daily_loss
        win_rate  = len(wins) / len(records) * 100 if records else 0.0
        best  = max((r.pnl for r in records), default=0.0)
        worst = min((r.pnl for r in records), default=0.0)
        avg   = total_pnl / len(records) if records else 0.0
        return {
            "total_pnl":   total_pnl,
            "total_trades": len(records),
            "wins":        len(wins),
            "losses":      len(losses),
            "win_rate":    win_rate,
            "best_trade":  best,
            "worst_trade": worst,
            "avg_trade":   avg,
        }

    @property
    def daily_loss(self) -> float:
        return self._daily_loss

    @property
    def daily_pnl(self) -> float:
        return self._daily_profit - self._daily_loss

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses
