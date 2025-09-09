# src/kis_omega/news/fetch.py
import datetime as dt, hashlib, time, argparse
from pathlib import Path
from urllib.parse import quote_plus
from concurrent.futures import ThreadPoolExecutor, as_completed

import bs4, feedparser, pandas as pd, requests, yaml

# KST = dt.timezone(dt.timedelta(hours=9))
# ROOT = Path(__file__).resolve().parents[3]   # e.g. C:\dev\kis-omega\src
# DATA_DIR = ROOT / "data" / "news_raw"
# DATA_DIR.mkdir(parents=True, exist_ok=True)

KST = dt.timezone(dt.timedelta(hours=9))

# ★ 프로젝트 루트(레포 루트) 기준이 되도록 parents[3]
ROOT = Path(__file__).resolve().parents[3]   # e.g. C:\dev\kis-omega
DATA_DIR = ROOT / "data" / "news_raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ★ sources.yaml: 새 위치 + 레거시(기존 위치) 폴백
CONF_SOURCES = ROOT / "data" / "sources.yaml"
CONF_SOURCES_OLD = Path(__file__).resolve().parent / "sources.yaml"  # src/kis_omega/news/sources.yaml

def now_utc(): return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
def to_kst(t): return t.astimezone(KST)
def sha16(s:str)->str: return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def fetch_rss(url: str):
    d = feedparser.parse(url)
    for e in d.entries:
        yield dict(
            title=e.get("title",""),
            url=e.get("link",""),
            published=e.get("published","") or e.get("updated",""),
            summary=e.get("summary","")
        )

def fetch_google_news(query: str, hl="ko", gl="KR"):
    rss = f"https://news.google.com/rss/search?q={quote_plus(query)}&hl={hl}&gl={gl}&ceid={gl}:{hl}"
    yield from fetch_rss(rss)

def extract_fulltext(url: str, timeout=6):
    try:
        html = requests.get(
            url, timeout=timeout,
            headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) kis-omega/0.2"}
        ).text
        soup = bs4.BeautifulSoup(html, "html.parser")
        article = soup.find("article")
        if article:
            texts = [p.get_text(" ", strip=True) for p in article.find_all("p")]
        else:
            texts = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        body = "\n".join(t for t in texts if len(t) > 30)
        return body[:20000]
    except Exception:
        return ""

# --- add (상단 import 아래 아무 곳에 추가) ---
def load_sources():
    """
    sources.yaml을 'data/'에서 우선 로드, 없으면 src/kis_omega/news/에서 레거시 폴백.
    반환: list[dict] (각 항목: id, name, kind, url, region, bias_tag)
    """
    path = None
    if CONF_SOURCES.exists():
        path = CONF_SOURCES
    elif CONF_SOURCES_OLD.exists():
        path = CONF_SOURCES_OLD
        print(f"[warn] using legacy sources file: {path}", flush=True)
    else:
        raise FileNotFoundError(
            f"뉴스 소스 파일 없음: {CONF_SOURCES} (또는 legacy {CONF_SOURCES_OLD})"
        )

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    srcs = cfg.get("sources", [])
    print(f"[ok] loaded {len(srcs)} sources from {path}", flush=True)
    return srcs

def iter_feeds():
    """
    fetch 루프에서 사용할 제너레이터.
    kind=="rss"만 필터링(필요시 atom/api 분기 추가).
    """
    for s in load_sources():
        kind = s.get("kind", "rss")
        if kind != "rss":
            continue
        yield {
            "id": s["id"],
            "name": s.get("name", s["id"]),
            "url": s["url"],
            "region": s.get("region", "global"),
            "bias_tag": s.get("bias_tag", "center"),
        }

def run(no_body: bool=False, max_per_source: int=9999, timeout: int=6, workers: int=0):
    # 소스 로드 (주의: ROOT는 src 기준이므로 "kis_omega/..."로 접근)
    sources = load_sources()

    print(f"[1/3] RSS/검색 수집 시작 (sources={len(sources)}, limit={max_per_source})")
    rows = []
    tsu = now_utc()
    for s in sources:
        collected = 0
        try:
            if s["type"] == "rss":
                for u in s["urls"]:
                    for item in fetch_rss(u):
                        rows.append((s["name"], s["bias"], item))
                        collected += 1
                        if collected >= max_per_source: break
                    if collected >= max_per_source: break
            elif s["type"] == "google_news":
                for item in fetch_google_news(s["query"]):
                    rows.append((s["name"], s["bias"], item))
                    collected += 1
                    if collected >= max_per_source: break
        except Exception as e:
            print(f"  ! {s['name']} 수집 중 오류: {e}")
        print(f"  - {s['name']}: {collected}건")
        time.sleep(0.2)

    # 메타 구성
    out = []
    total = len(rows)
    print(f"[2/3] 본문 추출 단계 시작 (no_body={no_body}, N={total})")

    def process_one(name, bias, it, idx):
        url = it.get("url","")
        if not url: return None
        body = "" if no_body or "news.google.com" in url else extract_fulltext(url, timeout=timeout)
        rid = sha16(url)
        return dict(
            id=rid,
            ts_utc=tsu.isoformat(),
            ts_kst=to_kst(tsu).isoformat(),
            source=name,
            bias_tag=bias,
            lang="ko" if ".kr" in url else "en",
            title=it.get("title",""),
            url=url,
            summary=it.get("summary",""),
            body_text=body
        )

    if workers and workers > 1:
        # 병렬 본문 추출(옵션)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(process_one, name, bias, it, idx) for idx, (name,bias,it) in enumerate(rows, start=1)]
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                if rec: out.append(rec)
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  · 본문 {done}/{total} 완료")
    else:
        # 순차 + 진행률 로그
        for idx, (name, bias, it) in enumerate(rows, start=1):
            rec = process_one(name, bias, it, idx)
            if rec: out.append(rec)
            if idx % 10 == 0 or idx == total:
                percent = (idx / total) * 100 if total else 100.0
                print(f"[{idx}/{total}] ({percent:.1f}%) 완료 - {name}")
            time.sleep(0.2)

    df = pd.DataFrame(out).drop_duplicates(subset=["url"])

    # 저장 경로/파일명
    stamp = to_kst(tsu).strftime("%Y%m%d_%H%M")
    date_kst = to_kst(tsu).strftime("%Y%m%d")
    (DATA_DIR / f"news_{stamp}.parquet").parent.mkdir(parents=True, exist_ok=True)

    # 스냅샷 + latest
    df.to_parquet(DATA_DIR / f"news_{stamp}.parquet", index=False)
    (DATA_DIR / "latest.parquet").unlink(missing_ok=True)
    df.to_parquet(DATA_DIR / "latest.parquet", index=False)

    # 일자별 파일(해당 KST 날짜의 최신본으로 덮어쓰기)
    by_day_dir = DATA_DIR / "by_day"
    by_day_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(by_day_dir / f"NEWS_{date_kst}.parquet", index=False)

    # 요약 CSV (소스/언어별 카운트)
    summary = (
        df.assign(date_kst=date_kst)
          .groupby(["date_kst","source","bias_tag","lang"], as_index=False)
          .size()
          .rename(columns={"size":"count"})
    )
    summary_path = DATA_DIR / f"summary_{date_kst}.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 요약 히스토리 누적 (중복 제거: date_kst+source)
    hist_path = DATA_DIR / "summary_history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        merged = (pd.concat([hist, summary], ignore_index=True)
                    .drop_duplicates(subset=["date_kst","source"], keep="last"))
    else:
        merged = summary
    merged.to_csv(hist_path, index=False, encoding="utf-8-sig")

    print(f"[3/3] 저장 완료: {len(df)} rows")
    print(f" - latest: {DATA_DIR / 'latest.parquet'}")
    print(f" - snapshot: {DATA_DIR / f'news_{stamp}.parquet'}")
    print(f" - by_day: {by_day_dir / f'NEWS_{date_kst}.parquet'}")
    print(f" - summary(csv): {summary_path}")
    print(f" - history(csv): {hist_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-body", action="store_true", help="본문 추출 생략(속도↑)")
    ap.add_argument("--max-per-source", type=int, default=9999, help="소스별 최대 기사 수")
    ap.add_argument("--timeout", type=int, default=6, help="요청 타임아웃(초)")
    ap.add_argument("--workers", type=int, default=0, help="본문 병렬 추출 스레드 수(0=비활성)")
    args = ap.parse_args()
    run(no_body=args.no_body, max_per_source=args.max_per_source, timeout=args.timeout, workers=args.workers)
