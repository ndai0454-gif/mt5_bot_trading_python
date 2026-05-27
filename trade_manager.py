import MetaTrader5 as mt5
import logging

class TradeManager:
    """
    TradeManager quản lý các lệnh đang chạy:
    - Dời SL về hòa vốn (Break-even)
    - Trailing Stop động
    - Chốt lời từng phần (Partial TP)
    """
    def __init__(self, partial_tp_percent=50.0, break_even_ratio=1.0):
        """
        :param partial_tp_percent: Phần trăm khối lượng đóng tại TP1 (ví dụ 50%)
        :param break_even_ratio: Tỷ lệ R:R để dời về hòa vốn (ví dụ 1.0 nghĩa là lãi bằng mức rủi ro ban đầu)
        """
        self.partial_tp_percent = partial_tp_percent
        self.break_even_ratio = break_even_ratio
        self.logger = logging.getLogger("TradeManager")

    def manage_position(self, position):
        """
        Hàm chính để quản lý một vị thế đang chạy.
        """
        symbol = position.symbol
        ticket = position.ticket
        entry_price = position.price_open
        current_sl = position.sl
        current_tp = position.tp
        current_price = position.price_current
        volume = position.volume
        
        # Tính khoảng cách rủi ro ban đầu (Risk)
        initial_risk = abs(entry_price - current_sl) if current_sl != 0 else 0
        
        if initial_risk == 0:
            return # Không thể quản lý nếu không có SL ban đầu

        # --- 1. Xử lý Chốt lời từng phần (Partial TP) ---
        # Kiểm tra xem lệnh đã đạt mức TP1 (giả sử TP1 là 1:1 hoặc một mức cố định)
        # Ở đây ta ví dụ: Nếu lãi đạt 1 lần Risk và volume vẫn còn nguyên -> Chốt 50%
        if self._is_partial_tp_reached(position, initial_risk):
            self._close_partial(ticket, volume)

        # --- 2. Xử lý Break-Even (Hòa vốn) ---
        # Nếu giá chạy được X lần Risk, dời SL về entry
        if self._should_move_to_be(position, initial_risk):
            if not self._is_already_at_be(current_sl, entry_price):
                self._modify_sl(ticket, entry_price)
                self.logger.info(f"Lệnh {ticket}: Đã dời SL về điểm hòa vốn (Break-even).")

        # --- 3. Trailing Stop (Dời SL theo giá) ---
        # Dời SL khi giá tạo đỉnh/đáy mới (đơn giản hóa bằng cách dời theo bước giá)
        self._apply_trailing_stop(position, initial_risk)

    def _is_partial_tp_reached(self, position, initial_risk):
        """Kiểm tra xem đã đạt mục tiêu chốt lời một phần chưa."""
        # Ví dụ: Lãi đạt 1.5 lần rủi ro thì chốt 50%
        profit_points = abs(position.price_current - position.price_open)
        return profit_points >= (initial_risk * 1.5)

    def _close_partial(self, ticket, volume):
        """Đóng một phần khối lượng của lệnh."""
        close_volume = round(volume * (self.partial_tp_percent / 100), 2)
        if close_volume <= 0: return

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5.symbol_info_tick(mt5.symbol_info(ticket).name).symbol, # Lấy symbol từ ticket
            "volume": close_volume,
            "type": mt5.ORDER_TYPE_SELL if mt5.positions_get(ticket=ticket)[0].type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": mt5.symbol_info_tick(mt5.symbol_info(ticket).name).bid if mt5.positions_get(ticket=ticket)[0].type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(mt5.symbol_info(ticket).name).ask,
            "magic": 123456,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            self.logger.info(f"Lệnh {ticket}: Đã chốt lời một phần {close_volume} lot.")
        else:
            self.logger.error(f"Lỗi chốt lời một phần: {result.comment}")

    def _should_move_to_be(self, position, initial_risk):
        """Điều kiện để dời SL về hòa vốn."""
        profit = abs(position.price_current - position.price_open)
        return profit >= (initial_risk * self.break_even_ratio)

    def _is_already_at_be(self, current_sl, entry_price):
        """Kiểm tra xem SL đã ở điểm hòa vốn chưa để tránh gửi request liên tục lên server."""
        return abs(current_sl - entry_price) < 0.01 # Sai số nhỏ cho Gold

    def _modify_sl(self, ticket, new_sl):
        """Hàm cập nhật SL lên MT5."""
        position = mt5.positions_get(ticket=ticket)[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": new_sl,
            "tp": position.tp,
        }
        result = mt5.order_send(request)
        return result

    def _apply_trailing_stop(self, position, initial_risk):
        """
        Trailing Stop: Dời SL lên khi giá tiếp tục chạy đúng hướng.
        Ví dụ: Cứ mỗi khi giá chạy thêm 1 lần Risk, dời SL lên 0.5 lần Risk.
        """
        ticket = position.ticket
        entry = position.price_open
        current_sl = position.sl
        current_price = position.price_current
        
        if position.type == mt5.ORDER_TYPE_BUY:
            # Nếu giá tăng, dời SL lên
            new_sl = current_price - (initial_risk * 0.5) 
            if new_sl > current_sl + (initial_risk * 0.1): # Chỉ dời khi có sự thay đổi đáng kể
                self._modify_sl(ticket, new_sl)
        
        elif position.type == mt5.ORDER_TYPE_SELL:
            # Nếu giá giảm, dời SL xuống
            new_sl = current_price + (initial_risk * 0.5)
            if current_sl == 0 or new_sl < current_sl - (initial_risk * 0.1):
                self._modify_sl(ticket, new_sl)