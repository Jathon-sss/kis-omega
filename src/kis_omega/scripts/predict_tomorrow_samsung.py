import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

def main():
    ROOT = Path(__file__).resolve().parents[3]

    data_path = ROOT / "data" / "samsung_features.csv"
    model_path = ROOT / "models" / "samsung_rf.pkl"
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "predict_samsung.log"

    def log(msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ── 블록 시작
    log("----- Predict START -----")

    try:
        # 데이터 / 모델 확인
        if not data_path.exists():
            raise FileNotFoundError(f"데이터 없음: {data_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"모델 없음: {model_path}")

        df = pd.read_csv(data_path)
        feature_cols = ["ma5", "ma20", "ma60", "vol_change", "return", "rsi14"]

        X_latest = df[feature_cols].iloc[[-1]]
        latest_date = df.iloc[-1]["date"]
        latest_close = df.iloc[-1]["close"]

        model = joblib.load(model_path)
        pred = model.predict(X_latest)[0]
        prob = model.predict_proba(X_latest)[0][1]

        # 결과 메시지
        out = (
            "=== 삼성전자 내일 주가 예측 ===\n"
            f"마지막 날짜: {latest_date}\n"
            f"마지막 종가: {latest_close}\n"
            f"예측 결과: {'상승 (BUY)' if pred == 1 else '하락 (SELL)'}\n"
            f"상승 확률: {prob*100:.2f}%\n"
        )
        print(out)
        log(out.strip())  # 줄 끝에 \n 제거 후 기록

        # === 예측 결과 CSV에 누적 기록 (중복 제거) ===
        pred_path = ROOT / "data" / "predictions_samsung.csv"
        row = {
            "date": latest_date,
            "close": latest_close,
            "pred": int(pred),
            "prob_up": float(f"{prob:.6f}")
        }

        if pred_path.exists():
            prev = pd.read_csv(pred_path)
            new_df = pd.concat([prev, pd.DataFrame([row])], ignore_index=True)
            new_df = new_df.drop_duplicates(subset=["date"], keep="last")
        else:
            new_df = pd.DataFrame([row])

        new_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    except Exception as e:
        log(f"에러 발생: {e!r}")
        raise
    finally:
        # ── 블록 종료
        log("----- Predict END -------\n")


if __name__ == "__main__":
    main()
