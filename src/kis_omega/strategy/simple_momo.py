"""
최소 동작 전략(데모):
- 현재가 조회 후, 전일대비등락값(또는 등락률)에 기반한 아주 단순한 신호.
- 실제 성능은 기대하지 말고, "주문 파이프가 정상 동작"하는지 확인용.
"""
from typing import Optional, Dict, Any

def decide_buy_from_quote(quote_json: Dict[str, Any], *, min_change_ratio: float = 1.0) -> bool:
    """
    quote_json: inquire-price 응답
    min_change_ratio: 등락률(%) 임계치
    문서/응답 형식에 따라 키가 다를 수 있어, 가능한 키를 순차조회.
    """
    # 등락률 후보 키 (환경/시점에 따라 다를 수 있음)
    candidates = ["prdy_ctrt", "PRDY_CTRT", "prdy_vrss_rt"]
    for k in candidates:
        if k in quote_json.get("output", {}):
            try:
                v = float(quote_json["output"][k])
                return v >= min_change_ratio
            except Exception:
                pass
    return False
