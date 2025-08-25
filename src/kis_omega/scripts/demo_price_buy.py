# src/kis_omega/scripts/demo_price_buy.py
import json
from kis_omega.client import KISClient
from kis_omega.config import settings

# --- 유틸: 현재가 조회 ---
def get_price(isin6: str):
    client = KISClient()
    headers = {
        **client._headers_auth(),
        "tr_id": "FHKST01010100",  # 현재가 조회 TR_ID
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",  # 주식
        "FID_INPUT_ISCD": isin6,        # 종목코드 (6자리)
    }
    data = client.get("/uapi/domestic-stock/v1/quotations/inquire-price",
                      headers=headers, params=params)
    return data

# --- 유틸: 현금 매수 주문 ---
def place_cash_order(isin6: str, qty: int = 1):
    client = KISClient()
    headers = {
        **client._headers_auth(),
        "tr_id": "VTTC0802U",  # 모의투자 현금 매수
    }
    body = {
        "CANO": settings.cano,              # 계좌번호 앞 8자리
        "ACNT_PRDT_CD": settings.acnt_prdt_cd,  # 계좌 상품 코드 (보통 01)
        "PDNO": isin6,                      # 종목코드
        "ORD_DVSN": "01",                   # 주문구분 (01=시장가)
        "ORD_QTY": str(qty),                # 수량
        "ORD_UNPR": "0",                    # 시장가=0
    }
    data = client.post("/uapi/domestic-stock/v1/trading/order-cash",
                       headers=headers, data=body)
    return data

# --- 전략: 단순 조건 ---
def decide_buy(quote: dict) -> bool:
    """
    예시 전략:
    - 현재가(stck_prpr)가 70,000 이하이면 BUY
    """
    price = int(quote["output"]["stck_prpr"])
    print(f"[전략판단] 현재가 = {price}")
    return price <= 70000

# --- 메인 ---
def main():
    isin = "005930"  # 삼성전자
    quote = get_price(isin)

    # 현재가 출력
    print(json.dumps(quote, ensure_ascii=False, indent=2))

    # 전략 판단
    if decide_buy(quote):
        print("[시그널] BUY 발생 → 모의 매수 주문 실행")
        res = place_cash_order(isin, qty=1)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        print("[시그널] 조건 미충족 → 매수 안 함")

if __name__ == "__main__":
    main()
