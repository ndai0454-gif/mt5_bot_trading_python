from datetime import datetime, timezone, time
from typing import List, Dict, Optional

def _parse_time(t_str: str) -> time:
    """Hàm hỗ trợ chuyển đổi chuỗi 'HH:mm' sang đối tượng time."""
    try:
        h, m = map(int, t_str.split(":"))
        return time(h, m)
    except (ValueError, IndexError):
        return time(0, 0)

def utc_now():
    """Trả về thời gian hiện tại theo chuẩn UTC."""
    return datetime.now(timezone.utc)

def get_active_session_name(now_utc: datetime, sessions: List[Dict]) -> Optional[str]:
    """
    Trả về tên session đang active (London/NewYork/Asian) hoặc None nếu đóng cửa.
    """
    if not sessions:
        return None

    current_time = now_utc.time().replace(second=0, microsecond=0)
    
    for s in sessions:
        start = _parse_time(s.get("start_utc", "00:00"))
        end = _parse_time(s.get("end_utc", "00:00"))
        
        if start < end:
            # Phiên bình thường (không xuyên đêm)
            if start <= current_time < end:
                return s["name"]
        else: 
            # Phiên xuyên đêm (Ví dụ: Asian từ 22h hôm nay đến 4h hôm sau)
            if current_time >= start or current_time < end:
                return s["name"]
                
    return None

def is_trading_session(now_utc: datetime, sessions: List[Dict]) -> bool:
    """
    Kiểm tra xem thời điểm hiện tại có nằm trong bất kỳ phiên giao dịch nào không.
    """
    return get_active_session_name(now_utc, sessions) is not None

# --- Block kiểm tra thử nghiệm (Khi chạy file này độc lập) ---
if __name__ == "__main__":
    # Giả lập config cho test
    test_sessions = [
        {"name": "London", "start_utc": "07:00", "end_utc": "15:00"},
        {"name": "NewYork", "start_utc": "13:00", "end_utc": "21:00"},
        {"name": "Asian", "start_utc": "22:00", "end_utc": "04:00"},
    ]
    
    now = utc_now()
    print(f"--- Kiểm tra Session Filter ---")
    print(f"Giờ UTC hiện tại: {now.strftime('%H:%M:%S')}")
    print(f"Đang trong phiên trading? {'Có' if is_trading_session(now, test_sessions) else 'Không'}")
    print(f"Phiên hiện tại: {get_active_session_name(now, test_sessions) or 'CLOSED (Đóng cửa)'}")
