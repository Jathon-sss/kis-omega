"""
시세 조회 모듈: /quotations/inquire-price
- TR_ID: FHKST01010100
- 파라미터: FID_COND_MRKT_DIV_CODE (J:주식), FID_INPUT_ISCD (종목코드 6자리)
참고: 문서 및 다수 레퍼런스
"""
from typing import Dict, Any
from .client import KISClient
from .config import settings
from .tr_ids import TRID_INQUIRE_PRICE

def get_price(isin6: str) -> Dict[str, Any]:
    """
    삼성전자: "005930" 같은 6자리 코드.
    반환: 원본 JSON (필드명은 문서 스펙)
    """
    client = KISClient()
    headers = {
        **client._headers_auth(),
        "tr_id": TRID_INQUIRE_PRICE,
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",   # J: 주식
        "FID_INPUT_ISCD": isin6,
    }
    return client.get("/uapi/domestic-stock/v1/quotations/inquire-price", headers=headers, params=params)
