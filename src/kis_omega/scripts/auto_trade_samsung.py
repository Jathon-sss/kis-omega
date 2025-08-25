# src/kis_omega/scripts/auto_trade_samsung.py
"""
predictions_samsung.csv의 '어제 예측'을 읽어
- 상승 확률이 임계치 이상이면 -> 모의 매수
- 그 외 -> 아무것도 안 함

안전장치:
- 한국장(09:00~15:30) 시간에만 동작
- 날짜 중복 주문 방지(1일 1회 잠금 파일)
- DRY_RUN 모드(기본 True, 실제 주문 막음)
- 예산/수량 한도
"""

from __future__ import annotations
import os
import json
from pathlib import Path
from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

from kis_omega.client import KISClient
from kis_omega.config import settings

# ===== 사용자 조정 파라미터 =====
ISIN = "005930"            # 삼성전자
THRESH = 0.55              # 매수 임계치 (상승확률 55% 이상이면 매수)
QTY = 1                    # 주문 수량(주)
BUDGET_LIMIT = 500_000     # 1일 최대 집행 예산(원). 시장가라 현재가*QTY가 이 한도 이하여야 주문
DRY_RUN = True             # True면 주문 API 호출하지 않고 콘솔만 출력(안전)


def now_kst() -> datetime:
    return datetime.now(ZoneInfo("Asia/Seoul"))


def is_market_time_kst(dt: datetime) -> bool:
    # 한국 주식 정규장 09:00:00 ~ 15:30:00 (공휴일/주말 체크는 간단히 요일로)
    if dt.weekday() >= 5:  # 토(5)/일(6)
        return False
    start = dt.replace(hour=9, minute=0, second=0, microsecond=0)
    end = dt.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= dt <= end


def read_latest_prediction(root: Path) -> dict:
    csv_path = root / "data" / "predictions_samsung.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"예측 파일이 없습니다: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError("예측 파일이 비어 있습니다.")

    last = df.iloc[-1]
    # 컬럼: date, close, pred, prob_up (이전 단계에서 생성)
    return {
        "date": str(last["date"]),
        "close": float(last["close"]),
        "pred": int(last["pred"]),
        "prob_up": float(last["prob_up"]),
    }


def get_price(isin6: str) -> dict:
    client = KISClient()
    headers = {
        **client._headers_auth(),
        "tr_id": "FHKST01010100",  # 현재가
    }
    params = {
        "FID_COND_MRKT_DIV_CODE": "J",
        "FID_INPUT_ISCD": isin6,
    }
    return client.get("/uapi/domestic-stock/v1/quotations/inquire-price",
                      headers=headers, params=params)


def place_cash_order_buy(isin6: str, qty: int) -> dict:
    client = KISClient()
    headers = {
        **client._headers_auth(),
        "tr_id": "VTTC0802U",  # 모의투자 현금 '매수'
    }
    body = {
        "CANO": settings.cano,
        "ACNT_PRDT_CD": settings.acnt_prdt_cd,
        "PDNO": isin6,
        "ORD_DVSN": "01",     # 01=시장가
        "ORD_QTY": str(qty),
        "ORD_UNPR": "0",      # 시장가=0
    }
    return client.post("/uapi/domestic-stock/v1/trading/order-cash",
                       headers=headers, data=body)


def today_lock_path(root: Path) -> Path:
    kst = now_kst()
    return root / "data" / f".trade_lock_{kst:%Y%m%d}.txt"


def main():
    ROOT = Path(__file__).resolve().parents[3]
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "auto_trade_samsung.log"

    def log(msg: str):
        ts = now_kst().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ── 블록 시작 헤더
    log("----- AutoTrade START -----")

    try:
        now = now_kst()
        if not is_market_time_kst(now):
            log("정규장 시간이 아닙니다. (09:00~15:30에서만 실행)")
            return

        lock_path = today_lock_path(ROOT)
        if lock_path.exists():
            log(f"오늘은 이미 주문 시도 기록이 있습니다: {lock_path.name}")
            return

        pred = read_latest_prediction(ROOT)
        log(f"예측 읽음: date={pred['date']}, prob_up={pred['prob_up']:.4f}, pred={pred['pred']}")

        if pred["prob_up"] < THRESH or pred["pred"] != 1:
            log(f"임계치 미충족 -> 매수 안 함 (THRESH={THRESH:.2f})")
            lock_path.write_text("no-trade\n", encoding="utf-8")
            return

        quote = get_price(ISIN)
        price_now = int(quote["output"]["stck_prpr"])
        notional = price_now * QTY
        log(f"현재가={price_now:,}원, 수량={QTY}주, 예상 체결금액≈{notional:,}원")

        if notional > BUDGET_LIMIT:
            log(f"예산 한도 초과 -> 매수 중단 (한도={BUDGET_LIMIT:,}원)")
            lock_path.write_text("blocked-budget\n", encoding="utf-8")
            return

        if DRY_RUN:
            log("[DRY_RUN] 실제 주문 대신 시뮬레이션만 수행합니다.")
            lock_path.write_text("dry-run\n", encoding="utf-8")
            return

        res = place_cash_order_buy(ISIN, QTY)
        log("주문 응답:")
        log(json.dumps(res, ensure_ascii=False))
        lock_path.write_text("ordered\n", encoding="utf-8")

    except Exception as e:
        log(f"에러: {e!r}")
        # 실패도 오늘 1회 시도로 기록(원하면 주석 처리)
        today_lock_path(ROOT).write_text("error\n", encoding="utf-8")
        raise
    finally:
        # ── 블록 종료 푸터
        log("----- AutoTrade END -------\n")



if __name__ == "__main__":
    main()
