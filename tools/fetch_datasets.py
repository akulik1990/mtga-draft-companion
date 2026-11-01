"""Fetch upstream's published datasets for the live sets into <data>/Sets.

Equivalent to upstream's in-app sync, but reads the gh-pages branch over git so it
works where unrealities.github.io is not reachable. Checksums are verified against
the manifest.

    uv run python tools/fetch_datasets.py [--data DIR] [--groups All Top]
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

REPO = "https://github.com/unrealities/MTGA_Draft_17Lands"


def git(*args, cwd):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data", help="companion data dir (default: ./data)")
    p.add_argument("--groups", nargs="+", default=["All"], help="user groups to fetch")
    args = p.parse_args()
    sets_dir = Path(args.data) / "Sets"
    sets_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        git("clone", "-q", "--depth", "1", "--branch", "gh-pages", "--filter=blob:none", "--no-checkout", REPO, ".", cwd=tmp)
        git("checkout", "-q", "gh-pages", "--", "manifest.json", cwd=tmp)
        manifest = json.loads((Path(tmp) / "manifest.json").read_text())
        live = set(manifest["active_sets"])
        wanted = {
            k: v for k, v in manifest["datasets"].items()
            # SET_EVENT_GROUP only; 4-part keys (ArenaDirect_Sealed) are read as custom files upstream.
            if k.split("_")[0] in live and len(k.split("_")) == 3 and k.split("_")[2] in args.groups
        }
        git("checkout", "-q", "gh-pages", "--", *(v["filename"] for v in wanted.values()), cwd=tmp)
        for key, info in sorted(wanted.items()):
            raw = (Path(tmp) / info["filename"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != info["hash"]:
                print(f"SKIP {key}: checksum mismatch")
                continue
            (sets_dir / info["filename"].removesuffix(".gz")).write_bytes(gzip.decompress(raw))
            print(f"OK   {key}  {info['start_date']} -> {info['end_date']}")
    print(f"live sets: {', '.join(sorted(live))}")


if __name__ == "__main__":
    main()
