# src/kis_omega/features/merge_features.py
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_NEWS = ROOT / "data"
DATA_PRICES = ROOT / "data" / "prices"
OUT_DIR = ROOT / "data" / "features"

OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_news(topic: str) -> pd.DataFrame:
    """토픽별 뉴스 지수 히스토리 로드"""
    hist_path = DATA_NEWS / "news" / topic / "index_history.parquet"
    if not hist_path.exists():
        raise FileNotFoundError(hist_path)
    df = pd.read_parquet(hist_path)
    return df.rename(columns={"idx_overall": f"{topic}_sent"})

def load_price(symbol: str) -> pd.DataFrame:
    """Yahoo Finance 형식의 가격 데이터 로드"""
    path = DATA_PRICES / f"{symbol}.csv"
    df = pd.read_csv(path, parse_dates=["Date"])
    return df[["Date","Close"]].rename(columns={"Close": f"{symbol}_close"})

def merge_topic_symbol(topic: str, symbol: str):
    news = load_news(topic)
    price = load_price(symbol)

    # 뉴스 date_kst → datetime 변환
    news["Date"] = pd.to_datetime(news["date_kst"], format="%Y%m%d")
    news = news[["Date", f"{topic}_sent"]]

    df = pd.merge(news, price, on="Date", how="inner")

    # 라벨 생성: 다음날 상승(1)/하락(0)
    df[f"{symbol}_target"] = (df[f"{symbol}_close"].shift(-1) > df[f"{symbol}_close"]).astype(int)

    return df.dropna()

def main():
    merged_unh = merge_topic_symbol("unh", "UNH")
    merged_srpt = merge_topic_symbol("srpt", "SRPT")

    merged_unh.to_csv(OUT_DIR / "UNH_features.csv", index=False)
    merged_srpt.to_csv(OUT_DIR / "SRPT_features.csv", index=False)

    print("✅ Features saved:", OUT_DIR)

if __name__ == "__main__":
    main()
