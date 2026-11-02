"""python -m companion [--browser] [--port 17173] [--log PATH] [--no-sync] [--overlay auto|on|off]"""

import argparse
import atexit
import logging
import os
import shutil
import subprocess
import sys
import threading
import webbrowser

from pathlib import Path

from companion import LAUNCH_DIR, ROOT  # bootstrap: vendor path + data dir

log = logging.getLogger("companion")


def start_overlay(mode, port):
    """The in-game overlay for this platform, owned by this process (it stops when the app stops).

    Linux under Wayland: overlay/draft.qml in Quickshell (layer-shell). Elsewhere there is no overlay yet.
    """
    if mode == "off":
        return None
    wayland = sys.platform.startswith("linux") and os.environ.get("WAYLAND_DISPLAY")
    quickshell = shutil.which("quickshell") or subprocess.run(["pgrep", "-f", "/bin/quickshell"], capture_output=True).returncode == 0
    if not (wayland and quickshell):
        if mode == "on":
            log.warning("overlay: needs Linux under Wayland with Quickshell; not started")
        return None
    env = {**os.environ, "OVERLAY_API": f"http://127.0.0.1:{port}"}
    proc = subprocess.Popen([str(ROOT / "overlay" / "run.sh")], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    atexit.register(proc.terminate)
    log.info("overlay: started (pid %s)", proc.pid)
    return proc


def main():
    p = argparse.ArgumentParser(prog="companion")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=17173)
    p.add_argument("--log", help="Player.log path (overrides discovery for this run)")
    p.add_argument("--no-sync", action="store_true", help="skip dataset sync at startup")
    p.add_argument("--browser", action="store_true", help="open the UI in a browser")
    p.add_argument("--overlay", choices=("auto", "on", "off"), default="auto", help="in-game overlay (auto: when the platform has one)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    from companion.api.app import create_app
    from companion.boot import build_session

    log_path = str((LAUNCH_DIR / Path(args.log).expanduser()).resolve()) if args.log else None
    session = build_session(log_path, sync=not args.no_sync)
    if args.browser:
        url = f"http://{args.host}:{args.port}/"
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    start_overlay(args.overlay, args.port)
    uvicorn.run(create_app(session), host=args.host, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
