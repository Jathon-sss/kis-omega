import os
import json
import time
from typing import Dict
import httpx
from rich.console import Console
from datetime import datetime
from kis_omega.config import settings

console = Console()

# 루트에 토큰 캐시 파일 저장
TOKEN_PATH = os.path.join(os.path.dirname(__file__), "..", "..", ".token.json")

class KISClient:
    def __init__(self, timeout: float = 5.0):
        self.base_url = settings.base_url
        self.app_key = settings.app_key
        self.app_secret = settings.app_secret
        self.timeout = timeout
        self._token: str | None = None
        self._token_expire_ts: float = 0.0

    # ---------- 기본 헤더 ----------
    def _headers_base(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "custtype": "P",   # 개인 계좌 전용
            "tr_cont": "",
            "tr_cont_key": "",
        }

    def _headers_auth(self) -> Dict[str, str]:
        tok = self.get_token()
        return {
            **self._headers_base(),
            "authorization": f"Bearer {tok}",
        }

    # ---------- 토큰 ----------
    def get_token(self) -> str:
        """
        접근 토큰 발급/재사용
        - 메모리 + 파일 캐싱
        - 유효하면 그대로 사용, 만료 시 새로 발급
        """
        now = time.time()

        # 1) 메모리에 있으면 사용
        if self._token and now < self._token_expire_ts - 60:
            return self._token

        # 2) 파일 캐시 확인
        if os.path.exists(TOKEN_PATH):
            try:
                with open(TOKEN_PATH, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("access_token") and now < cached.get("expires_at", 0):
                    self._token = cached["access_token"]
                    self._token_expire_ts = cached["expires_at"]
                    return self._token
            except Exception:
                pass  # 파일 깨졌으면 무시하고 새 발급

        # 3) 새 발급 요청
        url = f"{self.base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        with httpx.Client(timeout=self.timeout) as s:
            r = s.post(url, json=payload, headers={"Content-Type": "application/json"})
            if r.status_code != 200:
                raise RuntimeError(f"Token error {r.status_code}: {r.text}")
            data = r.json()

        token = data.get("access_token") or data.get("accessToken") or data.get("ACCESS_TOKEN")
        expires_in = int(data.get("expires_in", 60 * 60 * 6))  # 기본 6h, 문서상 24h
        if not token:
            raise RuntimeError(f"Token parse error: {data}")

        # 4) 메모리 + 파일 저장
        self._token = token
        self._token_expire_ts = now + expires_in
        human_time = datetime.fromtimestamp(self._token_expire_ts).strftime("%Y-%m-%d %H:%M:%S")

        try:
            with open(TOKEN_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "access_token": token,
                    "expires_at": self._token_expire_ts,
                    "expires_at_human": human_time  # ✅ 사람이 읽기 쉬운 만료 시각
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            console.log(f"[WARN] 토큰 캐시 저장 실패: {e}")

        console.log(f"[token] issued; expires_in={expires_in}s (until {human_time})")
        return token

    # ---------- GET ----------
    def get(self, path: str, headers: Dict[str, str], params: Dict[str, str]) -> Dict:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as s:
            r = s.get(url, headers=headers, params=params)
            r.raise_for_status()
            return r.json()

    # ---------- POST ----------
    def post(self, path: str, headers: Dict[str, str], data: Dict) -> Dict:
        url = f"{self.base_url}{path}"
        with httpx.Client(timeout=self.timeout) as s:
            r = s.post(url, headers=headers, json=data)
            r.raise_for_status()
            return r.json()
