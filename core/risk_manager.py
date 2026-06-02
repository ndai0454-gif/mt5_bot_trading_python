# core/risk_manager.py
import logging
import pandas as pd
import numpy as np

class RiskManager:
    def __init__(self, config):
        # --- Thông số từ Config ---
        self.base_risk_percent = config.get("risk_per_trade", 1.0)
        self.max_drawdown_lock = config.get("max_drawdown_lock", 10.0)
        self.daily_max_loss_pct = config.get("daily_max_loss_pct", 10.0)
        self.min_free_margin_pct = config.get("min_free_margin_pct", 50.0)
        
        # --- Biến theo dõi trạng thái ---
        self.balance = 0.0
        self.peak_equity = 0.0
        self.daily_start_balance = 0.0
        self.daily_loss = 0.0
        self.daily_pnl = 0.0          # Fix lỗi AttributeError: daily_pnl
        self.consecutive_losses = 0   # Fix lỗi AttributeError: consecutive_losses
        self.daily_stopped = False
        self.last_day_checked = None
        self.circuit_breaker_tripped = False
        
        self.trade_history = []
        self.logger = logging.getLogger("RiskManager")

    def set_session_balance(self, amount):
        self.balance = amount
        self.peak_equity = amount
        self.daily_start_balance = amount
        self.last_day_checked = pd.Timestamp.now()
    def get_current_balance(self):
        return self.balance

    # --- HÀM FIX LỖI CHÍNH: check_equity_hard_stop ---
    def check_equity_hard_stop(self, equity):
        """
        Được gọi bởi main.py khi nhấn Start.
        Kiểm tra xem tài khoản có bị sụt giảm quá mức cho phép không.
        """
        if equity > self.peak_equity:
            self.peak_equity = equity

        if self.peak_equity > 0:
            drawdown = ((self.peak_equity - equity) / self.peak_equity) * 100
        else:
            drawdown = 0

        if drawdown >= self.max_drawdown_lock:
            self.circuit_breaker_tripped = True
            return False, f"HARD STOP: Drawdown {drawdown:.2f}% exceeded limit"

        return True, "Equity OK"

    def calculate_lot_size(self, balance, sl_distance, trend, tick_value, volume_step, tick_size, volume_min, volume_max=10.0, is_asia_session=False):
        risk_pct = self._calculate_adaptive_risk()
        if risk_pct == 0 or self.circuit_breaker_tripped: return 0.0

        current_risk = risk_pct
        if is_asia_session:
            current_risk *= 0.5 # Giảm 50% rủi ro phiên Á
        risk_amount_usd = balance * (current_risk / 100)
        sl_points = sl_distance / tick_size if tick_size != 0 else sl_distance
        
        try:
            raw_lot = risk_amount_usd / (sl_points * tick_value)
        except ZeroDivisionError: return 0.0
        lot = round(raw_lot / volume_step) * volume_step
        lot = float(f"{lot:.2f}")
        return max(volume_min, min(lot, volume_max)) if lot >= volume_min else 0.0

    def _calculate_adaptive_risk(self):
        if self.peak_equity <= 0: return self.base_risk_percent
        drawdown = ((self.peak_equity - self.balance) / self.peak_equity) * 100
        if drawdown <= 3.0: return self.base_risk_percent
        elif 3.0 < drawdown <= 7.0: return self.base_risk_percent * 0.5
        elif 7.0 < drawdown < self.max_drawdown_lock: return self.base_risk_percent * 0.25
        else:
            self.circuit_breaker_tripped = True
            return 0.0 

    def record_trade_result(self, pnl, direction=None, entry=None, exit_price=None, lots=None):
        self.balance += pnl
        self.daily_pnl += pnl
        self.trade_history.append(pnl)
        if self.balance > self.peak_equity:
            self.peak_equity = self.balance

        if pnl > 0: self.consecutive_losses = 0
        elif pnl < 0: self.consecutive_losses += 1

    def is_daily_limit_reached(self):
        if self.daily_start_balance <= 0: return False
        limit = self.daily_start_balance * (self.daily_max_loss_pct / 100.0)
        return self.daily_loss >= limit
    def get_daily_stats(self):
        if not self.trade_history:
            return {"total_trades": 0, "win_rate": 0.0, "total_pnl": 0.0, "best_trade": 0, "worst_trade": 0}
        wins = [p for p in self.trade_history if p > 0]
        return {
            "total_trades": len(self.trade_history),
            "win_rate": (len(wins) / len(self.trade_history)) * 100,
            "total_pnl": sum(self.trade_history),
            "best_trade": max(self.trade_history),
            "worst_trade": min(self.trade_history)
        }

