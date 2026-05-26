import logging
from datetime import datetime, timezone, time
from typing import List, Dict, Optional, Any

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _parse_time(t_str: str) -> time:
    """
    Hàm hỗ trợ chuyển đổi chuỗi 'HH:mm' sang đối tượng time.
    Ví dụ: '07:00' -> time(7, 0)
    """
    try:
        h, m = map(int, t_str.split(":"))
        return time(h, m)
    except (ValueError, IndexError, AttributeError):
        return time(0, 0)

def utc_now():
    """Trả về thời gian hiện tại theo chuẩn UTC."""
    return datetime.now(timezone.utc)

def get_active_session_name(now_utc: datetime, sessions: List[Dict]) -> Optional[str]:
    """
    Trả về tên session đang active (London/NewYork/Asian) hoặc None nếu đóng cửa.
    Hỗ trợ cả phiên giao dịch xuyên đêm (ví dụ: 22:00 hôm nay đến 04:00 hôm sau).
    """
    if not sessions:
        return None

    # Lấy giờ hiện tại, bỏ qua giây và micro giây để so sánh chính xác
    current_time = now_utc.time().replace(second=0, microsecond=0)
    
    for s in sessions:
        start = _parse_time(s.get("start_utc", "00:00"))
        end = _parse_time(s.get("end_utc", "00:00"))
        
        if start < end:
            # Phiên bình thường (không xuyên đêm). Ví dụ: 07:00 -> 12:00
            if start <= current_time < end:
                return s["name"]
        else: 
            # Phiên xuyên đêm. Ví dụ: Asian từ 22:00 hôm nay đến 04:00 hôm sau
            if current_time >= start or current_time < end:
                return s["name"]
                
    return None

def is_trading_session(now_utc: Any = None, sessions_config: Optional[List[Dict]] = None) -> bool:
    """
    Kiểm tra nhanh xem thời điểm hiện tại có nằm trong bất kỳ phiên giao dịch nào không.
    """
    try:
        current_now = now_utc if now_utc is not None else utc_now()
        if hasattr(current_now, 'to_pydatetime'):
            current_now = current_now.to_pydatetime()
            
        return get_active_session_name(current_now, sessions_config) is not None
        
    except Exception as e:
        logger.error(f"Error in is_trading_session: {e}")
        return False

# --- Block kiểm tra thử nghiệm (Khi chạy trực tiếp file này) ---
if __name__ == "__main__":
    # Giả lập cấu hình "Giờ Vàng" tối ưu (Theo phân tích PnL)
    golden_sessions = [
        {"name": "London_Golden", "start_utc": "07:00", "end_utc": "12:00"},
        {"name": "NewYork_Golden", "start_utc": "13:00", "end_utc": "16:00"},
    ]
    
    # Test 1: Kiểm tra giờ hiện tại
    now = utc_now()
    print(f"\n{'='*30}")
    print(f"--- KIỂM TRA SESSION FILTER (LIVE) ---")
    print(f"Giờ UTC hiện tại: {now.strftime('%H:%M:%S')}")
    
    is_active = is_trading_session(now, golden_sessions)
    session_name = get_active_session_name(now, golden_sessions)
    
    print(f"Có nằm trong Giờ Vàng? {'✅ Có' if is_active else '❌ Không'}")
    print(f"Phiên hiện tại: {session_name or 'NGOÀI GIỜ VÀNG'}")
    print(f"{'='*30}\n")

    # Test 2: Giả lập 08:00 UTC (Phải là London_Golden)
    test_time_london = datetime(2023, 1, 1, 8, 0, 0) 
    print(f"Giả lập 08:00 UTC -> Kết quả: {get_active_session_name(test_time_london, golden_sessions)}")

    # Test 3: Giả lập 14:00 UTC (Phải là NewYork_Golden)
    test_time_ny = datetime(2023, 1, 1, 14, 0, 0) 
    print(f"Giả lập 14:00 UTC -> Kết quả: {get_active_session_name(test_time_ny, golden_sessions)}")

    # Test 4: Giả lập 20:00 UTC (Phải là NGOÀI GIỜ VÀNG)
    test_time_off = datetime(2023, 1, 1, 20, 0, 0) 
    print(f"Giả lập 20:00 UTC -> Kết quả: {get_active_session_name(test_time_off, golden_sessions)}")

