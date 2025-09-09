from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf

US_TICKERS = {"NVDA","TSLA","AAPL","MSFT","AMZN","UNH"}

def load_news(path: Path, tickers: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"topic","date_kst","idx_overall","mean_conf","article_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    df["topic"] = df["topic"].astype(str)
    df = df[df["topic"].str.upper().isin([t.upper() for t in tickers])].copy()
    if df.empty:
        raise ValueError("선택한 종목에 해당하는 뉴스 데이터가 없습니다.")

    df["date_kst"] = pd.to_datetime(df["date_kst"]).dt.tz_localize("Asia/Seoul", nonexistent="NaT", ambiguous="NaT").dt.date
    df["idx_overall"] = pd.to_numeric(df["idx_overall"], errors="coerce")
    df["mean_conf"] = pd.to_numeric(df["mean_conf"], errors="coerce")
    df = df.dropna(subset=["idx_overall","mean_conf","date_kst"])

    # 미국 종목: KST 날짜 → 미국 거래일 기준 보정 (대부분 KST-1일이 같은 거래일)
    df["date_us_guess"] = pd.to_datetime(df["date_kst"]) - pd.Timedelta(days=1)
    df["date_us_guess"] = df["date_us_guess"].dt.date

    # 키 통일
    df["ticker"] = df["topic"].str.upper()
    return df

def download_prices(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    data = {}
    for t in tickers:
        dl = yf.download(t, start=start, end=end, progress=False, auto_adjust=False)
        if dl.empty:
            raise ValueError(f"{t} 가격 데이터를 내려받지 못했습니다.")
        dl = dl.reset_index()
        dl = dl.rename(columns={"Date":"date"})
        dl["date"] = pd.to_datetime(dl["date"]).dt.date
        dl = dl.sort_values("date").reset_index(drop=True)
        data[t] = dl
    return data

def compute_hit_ratio(news: pd.DataFrame, prices: dict[str, pd.DataFrame], horizon: int,
                      min_conf: float, abs_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    merged_rows = []

    for tkr, g in news.groupby("ticker"):
        g = g.copy()

        # confidence / 신호 크기 필터
        if min_conf > 0:
            g = g[g["mean_conf"] >= min_conf]
        if abs_threshold > 0:
            g = g[g["idx_overall"].abs() >= abs_threshold]
        if g.empty:
            rows.append({"ticker": tkr, "n": 0, "hit_ratio": np.nan})
            continue

        px = prices[tkr].copy()
        # 수익률: 다음(또는 T+horizon) 종가 대비
        px = px.sort_values("date").reset_index(drop=True)
        px["ret_next"] = px["Close"].pct_change(periods=horizon).shift(-horizon)

        # 뉴스(KST-1 → US date)와 동일 날짜 매칭
        m = pd.merge(g, px, left_on="date_us_guess", right_on="date", how="inner")

        if m.empty:
            rows.append({"ticker": tkr, "n": 0, "hit_ratio": np.nan})
            continue

        m["pred_sign"] = np.where(m["idx_overall"] > 0, 1, -1)
        m["real_sign"] = np.where(m["ret_next"] > 0, 1, -1)
        m["hit"] = (m["pred_sign"] == m["real_sign"]).astype(int)

        hit_ratio = m["hit"].mean() if len(m) else np.nan
        rows.append({"ticker": tkr, "n": len(m), "hit_ratio": hit_ratio})
        merged_rows.append(m.assign(ticker=tkr))

    summary = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
    details = pd.concat(merged_rows, ignore_index=True) if merged_rows else pd.DataFrame()
    return summary, details

def main():
    ap = argparse.ArgumentParser(description="뉴스 감성(idx_overall) vs 다음 거래일 수익률 hit ratio")
    ap.add_argument("--news", type=Path, default=Path("data/news_summary.csv"))
    ap.add_argument("--tickers", type=str, default="NVDA,TSLA,AAPL,MSFT,AMZN,UNH")
    ap.add_argument("--horizon", type=int, default=1, help="T+N 일 수익률 비교 (기본 1)")
    ap.add_argument("--min-conf", type=float, default=0.0, help="mean_conf 하한(기본 0=필터 없음)")
    ap.add_argument("--abs-threshold", type=float, default=0.0, help="|idx_overall| 하한(예: 0.15)")
    ap.add_argument("--start", type=str, default=None, help="가격 데이터 시작일 (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, default=None, help="가격 데이터 종료일 (YYYY-MM-DD)")
    ap.add_argument("--out-summary", type=Path, default=Path("data/metrics/news_hitratio_summary.csv"))
    ap.add_argument("--out-details", type=Path, default=Path("data/metrics/news_hitratio_details.csv"))
    args = ap.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    news = load_news(args.news, tickers)

    # 가격 범위 자동 산정(뉴스 날짜 ± 10일 버퍼)
    min_d = pd.to_datetime(news["date_us_guess"]).min() - pd.Timedelta(days=10)
    max_d = pd.to_datetime(news["date_us_guess"]).max() + pd.Timedelta(days=10)
    start = args.start or min_d.strftime("%Y-%m-%d")
    end   = args.end   or max_d.strftime("%Y-%m-%d")

    prices = download_prices(tickers, start=start, end=end)
    summary, details = compute_hit_ratio(
        news=news, prices=prices, horizon=args.horizon,
        min_conf=args.min_conf, abs_threshold=args.abs_threshold
    )

    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_details.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_summary, index=False)
    if not details.empty:
        details.to_csv(args.out_details, index=False)

    print("=== Hit Ratio (by ticker) ===")
    print(summary.to_string(index=False))
    print(f"\nSaved: {args.out_summary}")
    if not details.empty:
        print(f"Saved: {args.out_details}")

if __name__ == "__main__":
    main()
