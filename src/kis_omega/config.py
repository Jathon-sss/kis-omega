"""
환경설정 로더.
- .env 로부터 값을 읽고, 기본값/검증을 거칩니다.
"""
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv(override=True)

class Settings(BaseModel):
    env: str = Field(default=os.getenv("KIS_ENV", "paper"))           # "paper" or "real"
    app_key: str = Field(default_factory=lambda: os.getenv("KIS_APP_KEY", ""))
    app_secret: str = Field(default_factory=lambda: os.getenv("KIS_APP_SECRET", ""))
    cano: str = Field(default_factory=lambda: os.getenv("KIS_CANO", ""))
    acnt_prdt_cd: str = Field(default_factory=lambda: os.getenv("KIS_ACNT_PRDT_CD", "01"))

    dry_run: bool = Field(default=lambda: os.getenv("DRY_RUN", "true").lower() == "true")
    max_order_krw: int = Field(default=lambda: int(os.getenv("MAX_ORDER_KRW", "50000")))
    min_cash_reserve_krw: int = Field(default=lambda: int(os.getenv("MIN_CASH_RESERVE_KRW", "10000")))

    tz: str = Field(default_factory=lambda: os.getenv("TZ", "Asia/Seoul"))

    @property
    def base_url(self) -> str:
        # 문서에 명시된 모의/실전 도메인 사용
        if self.env.lower() == "paper":
            return "https://openapivts.koreainvestment.com:29443"  # 모의
        return "https://openapi.koreainvestment.com:9443"          # 실전

settings = Settings()
