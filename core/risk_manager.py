# core/risk_manager.py
import logging

class RiskManager:
    """
    Quản lý rủi ro cho hệ thống giao dịch.
    Bao gồm: 
    1. Tính toán Lot size dựa trên % rủi ro/vốn (Fixed Fractional).
    2. Điều chỉnh rủi ro thích nghi theo mức sụt giảm vốn (Risk Degradation).
    3. Cơ chế ngắt mạch (Circuit Breaker) khi chạm mức drawdown tối đa.
    """
    def __init__(self, config):
        # Cấu hình từ config.json
        self.base_risk_percent = config.get("risk_per_trade", 1.0)
        self.max_drawdown_lock = config.get("max_drawdown_lock", 10.0)
        
        # Trạng thái vốn
        self.initial_balance = 0.0 
        self.balance = 0.0 
        self.peak_equity = 0.0
        
        # Quản lý rủi ro động
        self.current_risk_percent = self.base_risk_percent
        self.circuit_breaker_tripped = False
        
        # Lịch sử và thống kê
        self.trade_history = []
        self.logger = logging.getLogger("RiskManager")

    def set_session_balance(self, amount):
        """Thiết lập số vốn ban đầu cho phiên giao dịch/backtest"""
        self.initial_balance = amount
        self.balance = amount
        self.peak_equity = amount
        self.circuit_breaker_tripped = False
        self.trade_history = []
        self.logger.info(f"Session Balance initialized: {amount}")

    def get_current_balance(self):
        """Trả về Equity hiện tại"""
        return self.balance

    def record_trade_result(self, pnl, direction=None, entry=None, exit_price=None, lots=None):
        """
        Cập nhật số dư sau mỗi lệnh và theo dõi đỉnh vốn (Peak Equity).
        Hỗ trợ nhận thêm các thông tin chi tiết từ TradeManager để tránh lỗi TypeError.
        """
        # 1. Cập nhật số dư và lịch sử
        self.balance += pnl
        self.trade_history.append(pnl)
        
        # 2. Cập nhật đỉnh vốn (Peak Equity) để tính Drawdown cho Mô hình 2
        if self.balance > self.peak_equity:
            self.peak_equity = self.balance
        
        # 3. Log chi tiết giao dịch để theo dõi trong Terminal
        if pnl < 0:
            msg = f"Trade Loss: {pnl:.2f}"
            if direction: msg += f" | Dir: {direction}"
            if exit_price: msg += f" | Exit: {exit_price:.2f}"
            self.logger.warning(f"{msg} | Current Balance: {self.balance:.2f}")
        elif pnl > 0:
            msg = f"Trade Win: {pnl:.2f}"
            if direction: msg += f" | Dir: {direction}"
            if exit_price: msg += f" | Exit: {exit_price:.2f}"
            self.logger.info(f"{msg} | Current Balance: {self.balance:.2f}")

    def _calculate_adaptive_risk(self):
        """
        Mô hình Risk Degradation: Giảm mức rủi ro khi tài khoản sụt giảm (Drawdown).
        - DD <= 3%: Rủi ro 100% base (VD: 1%)
        - 3% < DD <= 7%: Rủi ro 50% base (VD: 0.5%)
        - 7% < DD < Max Lock: Rủi ro 25% base (VD: 0.25%)
        - DD >= Max Lock: Ngắt mạch, dừng giao dịch.
        """
        if self.peak_equity <= 0: 
            return self.base_risk_percent
        
        drawdown = ((self.peak_equity - self.balance) / self.peak_equity) * 100
        
        if drawdown <= 3.0:
            self.current_risk_percent = self.base_risk_percent
        elif 3.0 < drawdown <= 7.0:
            self.current_risk_percent = self.base_risk_percent * 0.5
        elif 7.0 < drawdown < self.max_drawdown_lock:
            self.current_risk_percent = self.base_risk_percent * 0.25
        else:
            if not self.circuit_breaker_tripped:
                self.logger.critical(f"CIRCUIT BREAKER TRIPPED! Max Drawdown {self.max_drawdown_lock}% reached.")
            self.circuit_breaker_tripped = True
            return 0.0
            
        return self.current_risk_percent

    def calculate_lot_size(self, balance, sl_distance, trend, tick_value, volume_step, tick_size, volume_min, volume_max=10.0):
        """
        Tính toán khối lượng giao dịch (Lot size) chuẩn hóa cho nhiều loại tài sản.
        
        Tham số:
        - balance: Số dư hiện tại
        - sl_distance: Khoảng cách SL (giá trị tuyệt đối, VD: 20.0 pips/points)
        - trend: Hướng giao dịch (không dùng trong tính lot nhưng giữ để khớp interface)
        - tick_value: Giá trị của 1 tick cho 1 lot (VD: Gold 0.01 lot = 0.1$)
        - volume_step: Bước nhảy lot tối thiểu (VD: 0.01)
        - tick_size: Kích thước 1 tick (VD: 0.01)
        - volume_min: Lot tối thiểu sàn cho phép (VD: 0.01)
        """
        # 1. Tính % rủi ro thích nghi
        risk_pct = self._calculate_adaptive_risk()
        if risk_pct == 0 or self.circuit_breaker_tripped:
            return 0.0

        # 2. Tính số tiền rủi ro (USD)
        risk_amount_usd = balance * (risk_pct / 100)

        # 3. Quy đổi khoảng cách SL ra số Tick/Points
        # sl_points = Khoảng cách giá / Kích thước 1 tick
        sl_points = sl_distance / tick_size if tick_size != 0 else sl_distance
        
        if sl_points <= 0:
            self.logger.error("Invalid SL distance (<= 0). Lot size set to 0.")
            return 0.0

        try:
            # Công thức: Lot = Số tiền rủi ro / (Số points SL * Giá trị 1 tick của 1 lot)
            raw_lot = risk_amount_usd / (sl_points * tick_value)
        except ZeroDivisionError:
            return 0.0

        # 4. Chuẩn hóa theo quy định của sàn (Volume Step)
        # VD: raw_lot = 0.1267, volume_step = 0.01 -> lot = 0.13
        lot = round(raw_lot / volume_step) * volume_step
        
        # Xử lý sai số floating point của Python (VD: 0.1300000000002 -> 0.13)
        lot = float(f"{lot:.2f}")

        # 5. Kiểm tra giới hạn Lot
        if lot < volume_min:
            return 0.0 

        return min(lot, volume_max)

    def reset_circuit_breaker(self):
        """Reset trạng thái ngắt mạch để cho phép giao dịch trở lại"""
        self.circuit_breaker_tripped = False
        self.logger.info("Circuit breaker has been reset.")

    def get_daily_stats(self):
        """Tính toán thống kê hiệu suất thực tế dựa trên lịch sử giao dịch"""
        if not self.trade_history:
            return {
                "total_trades": 0, 
                "win_rate": 0.0, 
                "total_pnl": 0.0, 
                "best_trade": 0.0, 
                "worst_trade": 0.0
            }

        wins = [p for p in self.trade_history if p > 0]
        total_trades = len(self.trade_history)
        
        return {
            "total_trades": total_trades,
            "win_rate": (len(wins) / total_trades) * 100,
            "total_pnl": sum(self.trade_history), # Tổng PnL thực tế
            "best_trade": max(self.trade_history),
            "worst_trade": min(self.trade_history),
            "current_drawdown": ((self.peak_equity - self.balance) / self.peak_equity) * 100 if self.peak_equity > 0 else 0
        }
