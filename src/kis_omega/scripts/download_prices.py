# src/kis_omega/scripts/download_prices.py
import yfinance as yf
from pathlib import Path

# 프로젝트 루트: .../kis-omega
ROOT = Path(__file__).resolve().parents[2]

# 저장 위치: .../kis-omega/src/data/prices
OUT = ROOT / "data" / "prices"
OUT.mkdir(parents=True, exist_ok=True)

symbols = ["UNH", "SRPT"]

for symbol in symbols:
    df = yf.download(symbol, start="2024-01-01")
    df = df.reset_index()  # ✅ Date를 일반 컬럼으로 변환
    out_path = OUT / f"{symbol}.csv"
    df.to_csv(out_path, index=False)  # ✅ 인덱스 저장 안 함
    print(f"Saved {out_path} ({len(df)} rows)")
print("All done.")