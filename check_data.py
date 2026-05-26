import MetaTrader5 as mt5
from datetime import datetime

SYMBOL = "XAUUSD" # Thay đổi nếu sàn bạn đặt tên khác

if not mt5.initialize():
    print("MT5 Initialize failed!")
    exit()

# 1. Kiểm tra Symbol có tồn tại không
symbol_info = mt5.symbol_info(SYMBOL)
if symbol_info is None:
    print(f"❌ Symbol {SYMBOL} không tìm thấy. Hãy kiểm tra lại tên trong Market Watch!")
else:
    print(f"✅ Symbol {SYMBOL} tồn tại.")

# 2. Thử tải 1 lượng nhỏ dữ liệu gần đây nhất
rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 100)
if rates is None:
    print("❌ Không thể lấy dữ liệu gần đây. Có thể MT5 chưa kết nối server.")
else:
    print(f"✅ Lấy được {len(rates)} nến gần nhất. Kết nối OK!")

mt5.shutdown()