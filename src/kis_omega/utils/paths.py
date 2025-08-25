# src/kis_omega/utils/paths.py (신규)
from pathlib import Path
import os

LOCAL_APPDATA = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / ".kis-omega")))
TOKEN_DIR = LOCAL_APPDATA / "kis-omega" / "tokens"
TOKEN_DIR.mkdir(parents=True, exist_ok=True)

TOKEN_PATH = TOKEN_DIR / ".token.json"
