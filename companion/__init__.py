"""MTGA Draft Companion: web UI and a second pick advisor over vendored MTGA_Draft_17Lands.

Importing this package puts the vendored upstream on sys.path and makes the data
directory the working directory. Upstream resolves Sets/, Temp/, Logs/ and a local
config.json from os.getcwd() when src.constants is first imported, so this must run
before any `src` import. Override the location with MTGA_COMPANION_DATA.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENDOR = ROOT / "vendor" / "mtga_draft_17lands"


def _default_data_dir() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "mtga-draft-companion"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "mtga-draft-companion"


if "src.constants" in sys.modules:
    raise RuntimeError("companion must be imported before any vendored `src` module")

DATA_DIR = Path(os.environ.get("MTGA_COMPANION_DATA") or _default_data_dir()).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# A config.json in the working directory makes upstream use it instead of
# ~/.config/MTGA_Draft_Tool/config.json, keeping us isolated from their Tk app.
_config = DATA_DIR / "config.json"
if not _config.exists():
    _config.write_text("{}", encoding="utf-8")

os.chdir(DATA_DIR)
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))
