# src/kis_omega/scripts/fetch_history_samsung.py
import pandas as pd
from pykrx import stock
import os

def fetch_samsung_history(start="2020-01-01", end="2025-01-01"):
    """
    삼성전자(005930) 일봉 데이터를 pykrx로 수집
    """
    df = stock.get_market_ohlcv_by_date(start, end, "005930")
    df = df.reset_index()  # 날짜를 컬럼으로 변환
    df.rename(columns={
        "날짜": "date",
        "시가": "open",
        "고가": "high",
        "저가": "low",
        "종가": "close",
        "거래량": "volume",
        "거래대금": "tr_amount",
        "등락률": "change"
    }, inplace=True)

    # date 컬럼을 문자열 YYYY-MM-DD로 변환
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    return df

def main():
    save_path = os.path.join("data", "samsung_history.csv")
    os.makedirs("data", exist_ok=True)

    df = fetch_samsung_history(start="2020-01-01", end="2025-08-23")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 삼성전자 일봉 데이터 저장 완료: {save_path}")
    print(df.head())
    print(df.tail())

if __name__ == "__main__":
    main()
