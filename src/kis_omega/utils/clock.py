from datetime import datetime
import zoneinfo
import os

_TZ = os.getenv("TZ", "Asia/Seoul")

def now_kst() -> datetime:
    """KST(now) 반환. tz 데이터는 OS/pyzoneinfo 의존."""
    return datetime.now(zoneinfo.ZoneInfo(_TZ))
