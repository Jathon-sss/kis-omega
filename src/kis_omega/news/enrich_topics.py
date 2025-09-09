from __future__ import annotations

import time
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from transformers import pipeline

ROOT = Path(__file__).resolve().parents[3]
RAW  = ROOT / "data" / "news_raw" / "latest.parquet"
OUTD = ROOT / "data" / "news"

DEFAULTS_FILE = ROOT / "data" / "topics.yaml"
TOPICS_DIR    = ROOT / "data" / "topics"


# -------------------------
# Utility
# -------------------------
def _remap_label(label: str) -> float:
    s = (label or "")[:1]
    if s in ("1", "2"): return -1.0   # 확실한 부정
    if s in ("4", "5"): return +1.0   # 확실한 긍정
    return 0.0                        # 중립(3)

def _deadzone(x: float, dz: float = 0.15) -> float:
    return 0.0 if abs(x) < dz else x

def build_classifier(device: int = -1):
    return pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=device
    )

def label_to_score(label: str, score: float) -> float:
    s = str(label)
    if s.startswith(("4","5")): return +score
    if s.startswith(("1","2")): return -score
    return 0.0

def text_match_any(text: str, keywords: list[str]) -> bool:
    t = (text or "").lower()
    return any(k.lower() in t for k in keywords)

def log(msg: str): 
    print(msg, flush=True)

def fmt_eta(elapsed: float, done: int, total: int) -> str:
    if done == 0: return "ETA ?"
    rate = elapsed / done
    remain = rate * (total - done)
    m, s = divmod(int(remain), 60)
    return f"ETA {m}m {s}s"


# -------------------------
# Config Loader
# -------------------------
def load_defaults() -> dict:
    if not DEFAULTS_FILE.exists():
        log(f"[warn] defaults 파일이 없습니다: {DEFAULTS_FILE}")
        return {}
    with open(DEFAULTS_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("defaults", {})

def load_topics(files: list[str]) -> list[dict]:
    all_topics = []
    for file in files:
        path = Path(file)
        if not path.exists():
            path = TOPICS_DIR / file
        if not path.exists():
            log(f"[warn] 토픽 파일 없음 → skip: {file}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        topics = data.get("topics", [])
        all_topics.extend(topics)
        log(f"Loaded {len(topics)} topics from {path}")
    # 중복 제거 (id 기준)
    seen, unique = set(), []
    for t in all_topics:
        tid = t.get("id")
        if tid and tid not in seen:
            seen.add(tid)
            unique.append(t)
    return unique


# -------------------------
# Main Runner
# -------------------------
def run(topic_filter: Optional[str] = None,
        batch_size: int = 16,
        device: int = -1,
        topic_files: Optional[list[str]] = None):

    if not RAW.exists():
        raise FileNotFoundError(f"뉴스 원본 파일이 없습니다: {RAW}")

    log(f"load: {RAW}")
    df = pd.read_parquet(RAW)
    if df.empty:
        log("뉴스가 비어 있습니다. 종료합니다.")
        return

    # 날짜키 (히스토리 누적용)
    if "ts_kst" in df.columns and len(df["ts_kst"].dropna()) > 0:
        date_kst = pd.to_datetime(df["ts_kst"]).dt.tz_convert("Asia/Seoul").dt.strftime("%Y%m%d").iloc[0]
    else:
        date_kst = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d")

    # 기본값 채워넣기
    if "coverage_weight" not in df.columns: df["coverage_weight"] = 0.6
    if "is_fulltext" not in df.columns: df["is_fulltext"] = 0
    if "region" not in df.columns:
        df["region"] = df["url"].apply(lambda u: "domestic" if ".kr" in (u or "") else "global")

    # 설정 로드
    defaults = load_defaults()
    topic_files = topic_files or ["topics_stocks.yaml"]
    topics = load_topics(topic_files)

    if topic_filter:
        topics = [t for t in topics if t.get("id") == topic_filter]
        if not topics:
            log(f"지정한 토픽(id={topic_filter})을 찾을 수 없습니다. 종료.")
            return

    # 가중치 설정
    W_bias = defaults.get("weights_by_bias")
    if not W_bias:
        W_bias = defaults.get("weights", {"neutral":0.30,"positive":0.20,"negative":0.20})

    # 🔹 공통 소스 기본값 추가
    default_sources = defaults.get("sources_default", [])
    S_weight = defaults.get("sources_weight", {})  # 소스 가중치
    R_weight = defaults.get("region_weight", {"domestic": 0.9, "global": 1.0})

    # 분류기
    clf = build_classifier(device=device)
    log(f"classifier ready (device={device}, batch={batch_size})")

    latest_indices = []  # summary.csv용 모음

    for tp in topics:
        t0 = time.time()
        tid = tp["id"]
        kws = list(tp.get("keywords_ko", [])) + list(tp.get("keywords_en", []))
        use_sources = set(tp.get("sources", default_sources))


        # 소스 필터
        sub = df[df["source"].isin(use_sources)].copy()
        if sub.empty:
            log(f"[{tid}] 대상 소스 기사 없음 → skip")
            continue

        # 키워드 매칭
        text_all = (sub["title"].fillna("") + " " +
                    sub["summary"].fillna("") + " " +
                    sub["body_text"].fillna(""))
        sub = sub[text_all.apply(lambda x: text_match_any(x, kws))]
        if sub.empty:
            log(f"[{tid}] 키워드 매칭된 기사 없음 → skip")
            continue

        log(f"[{tid}] 대상 기사: {len(sub)}건 감정분석 시작")

        # ★ 감정분석 결과 → 라벨기반 재매핑 × 확신도 × 데드존
        texts = (sub["title"].fillna("") + " " + sub["summary"].fillna("") + " " + sub["body_text"].fillna("")).str[:512].tolist()

        total = len(texts)
        sents = []
        done = 0
        loop_t0 = time.time()

        for i in range(0, total, batch_size):
            batch = texts[i:i+batch_size]
            out = clf(batch)
            sents.extend(out)
            done = i + len(batch)
            elapsed = time.time() - loop_t0
            percent = (done / total) * 100
            log(f"  [{tid}] {done}/{total} ({percent:.1f}%) · {fmt_eta(elapsed, done, total)}")

        # 점수 변환
        sub = sub.reset_index(drop=True)
        labels = [x.get("label", "") for x in sents]
        confs  = [float(x.get("score", 0.0)) for x in sents]
        sub["sentiment"] = pd.Series([_deadzone(_remap_label(l) * c, 0.15) for l, c in zip(labels, confs)], index=sub.index)

        # ★ 가중치: 소스 × 리전 × 편향 × 커버리지(클립)
        sub["w_src"]  = sub["source"].map(S_weight).fillna(0.75)
        sub["w_reg"]  = sub["region"].map(R_weight).fillna(1.0)
        sub["w_bias"] = sub["bias_tag"].map(W_bias).fillna(0.25)
        sub["w_cov"]  = sub["coverage_weight"].clip(0.30, 0.90)  # 과도한 영향 방지
        sub["w"]      = sub["w_src"] * sub["w_reg"] * sub["w_bias"] * sub["w_cov"]

        # 집계
        def wavg_df(g: pd.DataFrame) -> float:
            ww = g["w"].sum()
            return float((g["sentiment"] * g["w"]).sum() / ww) if ww > 0 else float("nan")

        by_region = (
            sub.groupby("region")[["sentiment","w"]]
               .apply(wavg_df)
               .rename("idx")
        )
        overall = wavg_df(sub[["sentiment","w"]])

        # 저장 (latest & history)
        out_dir = OUTD / tid
        out_dir.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(out_dir / "latest_rows.parquet", index=False)
        idx_df = pd.DataFrame({
            "topic":[tid],
            "date_kst":[date_kst],
            "idx_domestic":[by_region.get("domestic", float("nan"))],
            "idx_global":[by_region.get("global", float("nan"))],
            "idx_overall":[overall],
            "article_count": [int(len(sub))],
            "mean_conf": [float(pd.Series(confs).mean()) if len(confs) else float("nan")],
        })
        idx_df.to_parquet(out_dir / "latest_index.parquet", index=False)

        # 히스토리 누적
        hist_path = out_dir / "index_history.parquet"
        if hist_path.exists():
            hist = pd.read_parquet(hist_path)
            merged = (pd.concat([hist, idx_df], ignore_index=True)
                        .drop_duplicates(subset=["date_kst"], keep="last")
                        .sort_values("date_kst"))
        else:
            merged = idx_df
        merged.to_parquet(hist_path, index=False)

        # summary 모음
        latest_indices.append(idx_df)

        elapsed_total = time.time() - t0
        log(f"[{tid}] 완료 · rows={len(sub)} · idx_overall={overall:.3f} · time={elapsed_total:.1f}s")

    # ---- 모든 토픽 요약 저장 ----
    if latest_indices:
        summary = pd.concat(latest_indices, ignore_index=True)

        # 최신 요약
        OUT_SUMMARY = ROOT / "data" / "news_summary.csv"
        summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")

        # 히스토리 누적
        OUT_SUMMARY_HIST = ROOT / "data" / "news_summary_history.csv"
        if OUT_SUMMARY_HIST.exists():
            old = pd.read_csv(OUT_SUMMARY_HIST, dtype={"date_kst":str})
            merged_hist = (pd.concat([old, summary], ignore_index=True)
                             .drop_duplicates(subset=["topic","date_kst"], keep="last")
                             .sort_values(["date_kst","topic"]))
        else:
            merged_hist = summary
        merged_hist.to_csv(OUT_SUMMARY_HIST, index=False, encoding="utf-8-sig")

        log(f"요약 저장 완료 → {OUT_SUMMARY}")
        log(f"요약 히스토리 갱신 → {OUT_SUMMARY_HIST}")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="뉴스 감정/지수 산출 (토픽별)")
    ap.add_argument("--topic", default=None, help="특정 토픽 id만 처리")
    ap.add_argument("--batch-size", type=int, default=16, help="감정분석 배치 크기")
    ap.add_argument("--device", type=int, default=-1, help="-1=CPU, 0=GPU")
    ap.add_argument("--files", nargs="*", help="topics/*.yaml 파일 리스트 (예: topics_stocks.yaml topics_macro.yaml)")
    args = ap.parse_args()

    run(topic_filter=args.topic,
        batch_size=args.batch_size,
        device=args.device,
        topic_files=args.files)
