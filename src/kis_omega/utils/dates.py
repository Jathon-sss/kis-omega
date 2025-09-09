from __future__ import annotations
from datetime import date, datetime
import pandas as pd
import pytz

KST = pytz.timezone("Asia/Seoul")

def to_kst_date(ts) -> date:
    """UTC/naive timestamp → KST 로 바꿔 'YYYY-MM-DD' date 반환"""
    if pd.isna(ts):
        return None
    if isinstance(ts, (int, float)):
        ts = pd.to_datetime(ts, utc=True, unit="s")
    else:
        ts = pd.to_datetime(ts, utc=True)
    return ts.tz_convert(KST).date()

def parse_ymd(s: str | None) -> date | None:
    if not s:
        return None
    return datetime.strptime(s, "%Y-%m-%d").date()

def in_range(d: date, start: date | None, end: date | None) -> bool:
    if start and d < start: return False
    if end and d > end: return False
    return True
