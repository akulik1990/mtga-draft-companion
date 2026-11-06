#!/usr/bin/env python3
"""Static copy of the /videos page for GitHub Pages: tools/video_site.py build <outdir> [--no-build]

<outdir> gets the built frontend, a redirect from / to videos/, and analysis/videos/**.json under data/videos/ (plus
<id>/engine/commits.json, the replay list the API would otherwise glob). Push the folder to a Pages repo as is.
The page finds no /api and reads those files by relative URL; the checklist is read-only there.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VIDEOS = REPO / "analysis" / "videos"


def build(a):
    out = Path(a.outdir).resolve()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} is not empty")
    if not a.no_build:
        subprocess.run(["npm", "--prefix", str(REPO / "frontend"), "run", "build"], check=True)
    shutil.copytree(REPO / "frontend" / "build", out, dirs_exist_ok=True)
    (out / "index.html").write_text('<!doctype html><meta http-equiv="refresh" content="0; url=videos/"><a href="videos/">videos</a>\n')
    (out / ".nojekyll").write_text("")  # Pages would otherwise skip the _app folder
    data = out / "data" / "videos"
    for f in VIDEOS.rglob("*.json"):
        (data / f.relative_to(VIDEOS)).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, data / f.relative_to(VIDEOS))
    for e in VIDEOS.glob("*/engine"):
        commits = sorted((p for p in e.glob("*.json") if p.stem != "latest"), key=lambda p: -p.stat().st_mtime)
        (data / e.relative_to(VIDEOS) / "commits.json").write_text("[" + ", ".join(f'"{p.stem}"' for p in commits) + "]\n")
    print(f"wrote {out}; preview: python -m http.server -d {out}, open /videos/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("outdir")
    b.add_argument("--no-build", action="store_true", help="reuse frontend/build")
    b.set_defaults(fn=build)
    a = p.parse_args()
    a.fn(a)
