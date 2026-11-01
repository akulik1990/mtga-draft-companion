"""python -m companion [--browser] [--port 17173] [--log PATH] [--no-sync]"""

import argparse
import logging
import threading
import webbrowser

import companion  # noqa: F401  (bootstrap: vendor path + data dir)


def main():
    p = argparse.ArgumentParser(prog="companion")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=17173)
    p.add_argument("--log", help="Player.log path (overrides discovery for this run)")
    p.add_argument("--no-sync", action="store_true", help="skip dataset sync at startup")
    p.add_argument("--browser", action="store_true", help="open the UI in a browser")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    from companion.api.app import create_app
    from companion.boot import build_session

    session = build_session(args.log, sync=not args.no_sync)
    if args.browser:
        url = f"http://{args.host}:{args.port}/"
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(session), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
