# src/kis_omega/client.py
import os
import json
import time
from typing import Dict, Optional
import httpx
from rich.console import Console
from datetime import datetime
from kis_omega.config import settings
from kis_omega.utils.paths import TOKEN_PATH  # << 레포 밖 경로 사용

console = Console()

class KISClient:
    def __init__(self, timeout: float = 5.0):
        self.base_url = settings.base_url
        self.app_key = settings.app_key
        self.app_secret = settings.app_secret
        self.timeout = timeout
        self._token: Optional[str] = None
        self._token_expire_ts: float = 0.0

    # ---------- 기본 헤더 ----------
    def _headers_base(self) -> Dict[str, str]:
        return {
            "content-type": "application/json; charset=utf-8",  # 소문자여도 OK
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "custtype": "P",
            "tr_cont": "",
            "tr_cont_key": "",
        }

    def _headers_auth(self) -> Dict[str, str]:
        return {"authorization": f"Bearer {self.get_token()}"}

    def _merge_headers(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        # 기본 + 인증 + 호출자 추가 헤더 순서로 병합(호출자가 우선)
        h = {**self._headers_base(), **self._headers_auth()}
        if extra:
            h.update(extra)
        return h

    # ---------- 토큰 ----------
    def get_token(self) -> str:
        now = time.time()

        # 1) 메모리 캐시
        if self._token and now < self._token_expire_ts - 60:
            return self._token

        # 2) 파일 캐시
        if os.path.exists(TOKEN_PATH):
            try:
                with open(TOKEN_PATH, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if cached.get("access_token") and now < cached.get("expires_at", 0):
                    self._token = cached["access_token"]
                    self._token_expire_ts = cached["expires_at"]
                    return self._token
            except Exception:
                pass

        # 3) 새 발급
        url = f"{self.base_url}/oauth2/tokenP"
        payload = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
        }
        with httpx.Client(timeout=self.timeout) as s:
            r = s.post(url, json=payload, headers={"content-type": "application/json"})
            if r.status_code != 200:
                raise RuntimeError(f"Token error {r.status_code}: {r.text[:500]}")
            data = r.json()

        token = data.get("access_token") or data.get("accessToken") or data.get("ACCESS_TOKEN")
        expires_in = int(data.get("expires_in", 60 * 60 * 6))
        if not token:
            raise RuntimeError(f"Token parse error: {data}")

        # 4) 저장
        self._token = token
        self._token_expire_ts = now + expires_in
        human_time = datetime.fromtimestamp(self._token_expire_ts).strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(TOKEN_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "access_token": token,
                    "expires_at": self._token_expire_ts,
                    "expires_at_human": human_time
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            console.log(f"[WARN] 토큰 캐시 저장 실패: {e}")

        console.log(f"[token] issued; expires_in={expires_in}s (until {human_time})")
        return token

    # ---------- 공통 요청 ----------
    def get(self, path: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, str]] = None) -> Dict:
        url = f"{self.base_url}{path}"
        h = self._merge_headers(headers)
        with httpx.Client(timeout=self.timeout) as s:
            r = s.get(url, headers=h, params=params)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"GET {url} failed: {r.status_code}\n"
                    f"Resp: {r.text[:1000]}"
                ) from e
            return r.json()

    def post(self, path: str, headers: Optional[Dict[str, str]] = None, data: Optional[Dict] = None) -> Dict:
        url = f"{self.base_url}{path}"
        h = self._merge_headers(headers)
        with httpx.Client(timeout=self.timeout) as s:
            r = s.post(url, headers=h, json=data)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise RuntimeError(
                    f"POST {url} failed: {r.status_code}\n"
                    f"Resp: {r.text[:1000]}"
                ) from e
            return r.json()
