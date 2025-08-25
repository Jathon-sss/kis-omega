"""
주의:
- 기본은 DRY_RUN=true 이므로 실제 주문을 보내지 않음.
- 실제 모의주문을 보내려면 .env에서 DRY_RUN=false 로 바꾸세요.
"""
from kis_omega.quotes import get_price
from kis_omega.strategy.simple_momo import decide_buy_from_quote
from kis_omega.orders import place_cash_order

TARGET = "005930"  # 삼성전자

def main():
    q = get_price(TARGET)
    ok = decide_buy_from_quote(q, min_change_ratio=0.5)  # 등락률 0.5% 이상이면 매수
    if ok:
        res = place_cash_order(TARGET, qty=1, side="buy", order_type="market", use_hashkey=False)
        print(res)
    else:
        print("신호 없음. 매수 안 함.")

if __name__ == "__main__":
    main()
