# src/kis_omega/scripts/demo_momentum_samsung.py
import json
import pandas as pd
from kis_omega.client import KISClient

def get_daily_chart(isin6: str, count: int = 60):
    """
    일봉 데이터 조회 (최근 N일치)
    """
    client = KISClient()
    headers = {
        **client._headers_auth(),
        "tr_id": "FHKST03010100",  # [일봉] TR_ID
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",  # J: 주식
        "FID_INPUT_ISCD": isin6,        # 종목코드
        "FID_PERIOD_DIV_CODE": "D",     # 일봉
        "FID_ORG_ADJ_PRC": "1",         # 수정주가
    }
    data = client.get(
        "/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
        headers=headers,
        params=params,
    )

    # ✅ output2 필드에 일봉 리스트가 들어있음
    rows = data.get("output2", [])
    if not rows:
        raise RuntimeError(f"일봉 데이터 없음: {json.dumps(data, ensure_ascii=False)}")

    df = pd.DataFrame(rows)
    df["stck_clpr"] = pd.to_numeric(df["stck_clpr"], errors="coerce")  # 종가
    df["stck_bsop_date"] = pd.to_datetime(df["stck_bsop_date"])        # 일자
    df = df.sort_values("stck_bsop_date").tail(count)
    return df

def momentum_strategy(df: pd.DataFrame):
    """
    단순 이동평균 모멘텀 전략
    """
    df["ma5"] = df["stck_clpr"].rolling(5).mean()
    df["ma20"] = df["stck_clpr"].rolling(20).mean()
    last = df.iloc[-1]
    if last["ma5"] > last["ma20"]:
        return "BUY"
    elif last["ma5"] < last["ma20"]:
        return "SELL"
    else:
        return "HOLD"

def main():
    isin = "005930"  # ✅ 삼성전자
    df = get_daily_chart(isin)
    signal = momentum_strategy(df)
    print("삼성전자(005930) 전략 시그널:", signal)
    print(df.tail(5)[["stck_bsop_date", "stck_clpr", "ma5", "ma20"]])

if __name__ == "__main__":
    main()
