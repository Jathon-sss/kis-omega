"""
KIS TR_ID 상수 모음.
- 문서/레퍼런스에 따라 자주 쓰는 값만 우선 정의.
"""
# 시세: 주식현재가
TRID_INQUIRE_PRICE = "FHKST01010100"  # [국내주식] 기본시세 > 주식현재가 시세

# 주문: 현금매수/매도 (모의/실전 구분)
TRID_ORDER_BUY_PAPER  = "VTTC0802U"
TRID_ORDER_SELL_PAPER = "VTTC0801U"
TRID_ORDER_BUY_REAL   = "TTTC0802U"
TRID_ORDER_SELL_REAL  = "TTTC0801U"

def trid_order_buy(env: str) -> str:
    return TRID_ORDER_BUY_PAPER if env.lower() == "paper" else TRID_ORDER_BUY_REAL

def trid_order_sell(env: str) -> str:
    return TRID_ORDER_SELL_PAPER if env.lower() == "paper" else TRID_ORDER_SELL_REAL
