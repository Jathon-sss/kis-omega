"""
현금 주문 모듈: /trading/order-cash
- 모의/실전 TR_ID 다름 (paper: VTTC, real: TTTC)
- Hashkey는 옵션(문서상 비필수)이나, 여기서는 안전하게 포함 가능하도록 설계
"""
from typing import Dict, Any, Literal
from .client import KISClient
from .config import settings
from .tr_ids import trid_order_buy, trid_order_sell

OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]

def place_cash_order(isin6: str, qty: int, *, side: OrderSide, order_type: OrderType, price: float | None = None, use_hashkey: bool = False) -> Dict[str, Any]:
    """
    - side: "buy" | "sell"
    - order_type: "market" or "limit"
    - price: 지정가일 때만 필수
    - 안전장치: DRY_RUN=true 이면 주문 호출 대신 payload/헤더만 반환
    """
    client = KISClient()
    env = settings.env.lower()
    tr_id = trid_order_buy(env) if side == "buy" else trid_order_sell(env)

    # KIS 주문 바디 (필드명은 문서 스펙 기준)
    # 시장가: ORD_DVSN="01", 지정가: "00"
    ord_dvsn = "01" if order_type == "market" else "00"

    body: Dict[str, Any] = {
        "CANO": settings.cano,
        "ACNT_PRDT_CD": settings.acnt_prdt_cd,
        "PDNO": isin6,                    # 종목
        "ORD_DVSN": ord_dvsn,             # 주문구분
        "ORD_QTY": str(qty),              # 수량(문서상 문자열)
        "ORD_UNPR": "0" if order_type == "market" else (str(price) if price is not None else "0"),
    }

    headers = {
        **client._headers_auth(),
        "tr_id": tr_id,
    }

    if use_hashkey:
        headers["hashkey"] = client.issue_hashkey(body)

    if settings.dry_run:
        # 실제 API 호출 대신, 무엇을 보낼지 눈으로 확인하고 안정성 확보
        return {"dry_run": True, "headers": headers, "body": body}

    return client.post("/uapi/domestic-stock/v1/trading/order-cash", headers=headers, json_body=body)
