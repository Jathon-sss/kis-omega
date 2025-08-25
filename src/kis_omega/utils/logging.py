from rich.console import Console
from rich.traceback import install
import logging
import os

def setup_logging():
    """
    간단한 콘솔 로거 구성.
    LOG_LEVEL 환경변수(기본 INFO)를 읽어 설정합니다.
    """
    install(show_locals=False)
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    return Console()
