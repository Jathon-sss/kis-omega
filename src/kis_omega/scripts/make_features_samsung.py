# src/kis_omega/scripts/make_features_samsung.py
import pandas as pd
import os

def make_features():
    # 원본 CSV 읽기
    df = pd.read_csv("data/samsung_history.csv")

    # 날짜 정렬 보장
    df = df.sort_values("date").reset_index(drop=True)

    # ==========================
    # 1) 라벨 생성
    # ==========================
    # 내일 종가 > 오늘 종가 → 1, 아니면 0
    df["label"] = (df["close"].shift(-1) > df["close"]).astype(int)

    # ==========================
    # 2) 기술적 지표 (피처)
    # ==========================
    # 이동평균선
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    # 거래량 변화율 (전일 대비 %)
    df["vol_change"] = df["volume"].pct_change() * 100

    # 일간 수익률 (전일 대비 %)
    df["return"] = df["close"].pct_change() * 100

    # RSI(14일)
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)  # 0으로 나눔 방지
    df["rsi14"] = 100 - (100 / (1 + rs))

    # ==========================
    # 3) 결측치 제거
    # ==========================
    df = df.dropna().reset_index(drop=True)

    # ==========================
    # 저장
    # ==========================
    save_path = os.path.join("data", "samsung_features.csv")
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 학습용 데이터셋 저장 완료: {save_path}")
    print(df.head())
    print(df.tail())

def main():
    make_features()

if __name__ == "__main__":
    main()
