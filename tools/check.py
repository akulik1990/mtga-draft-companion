"""One command for engine comparisons, sized to the change.

    uv run python tools/check.py quick    --note "SPLASH 0.8"   --v2 SPLASH=0.8     # ~1 min: 40 tables, one set, no harness
    uv run python tools/check.py standard --note "lands" --set TDM                    # ~3 min: 100 tables + harness, in parallel
    uv run python tools/check.py full     --note "round 2 final"                      # ~8 min: 200 tables x FRA/MID/TDM + harness FRA/TDM

quick: constants and small tweaks; it sees only large effects (lane score SE about 0.15).
standard: one engine change; the set it targets plus the harness on that set (or FRA).
full: end of a round and before a live Arena draft only; fresh seeds 1001+ so it is not
the tables a change was tuned on.

Upstream's side is cached (per seed for mock tables, per sample for the harness), so only our
engine is replayed; the first run on new seeds fills the cache. Every run is recorded in
results/runs.jsonl as usual; this prints one summary line per run.
Do not edit code under companion/ or tools/ while a check is running.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS_SETS = {"FRA", "TDM"}  # sets whose public dumps upstream's sampler can read (MID's cannot)

TIERS = {
    "quick": {"mock": [("{set}", 40, 1)], "harness": []},
    "standard": {"mock": [("{set}", 100, 1)], "harness": [("{hset}", 100, "")]},
    "full": {"mock": [("FRA", 200, 1001), ("MID", 200, 1001), ("TDM", 200, 1001)],
             "harness": [("FRA", 200, "fresh1"), ("TDM", 200, "")]},
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("tier", choices=TIERS)
    p.add_argument("--set", default="FRA", help="set for quick/standard (default FRA)")
    p.add_argument("--note", required=True, help="what is being tested; kept in the ledger")
    p.add_argument("--v2", action="append", default=[], metavar="NAME=VALUE", help="engine overrides, as in tools/mock_draft.py")
    a = p.parse_args(argv)

    hset = a.set if a.set in HARNESS_SETS else "FRA"
    overrides = [x for pair in a.v2 for x in ("--v2", pair)]
    tier = TIERS[a.tier]
    before = sum(1 for _ in open(ROOT / "results" / "runs.jsonl")) if (ROOT / "results" / "runs.jsonl").exists() else 0
    procs = []
    for set_code, seeds, start in tier["mock"]:
        s = set_code.format(set=a.set)
        cmd = [sys.executable, "tools/mock_draft.py", "compare", s, "--seeds", str(seeds), "--start", str(start),
               "--out", f"data/Temp/sim/{s}.check-{a.tier}.json", "--note", f"[{a.tier}] {a.note}", *overrides]
        procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True))

    # Harness runs write to different files per set and salt, so they run alongside the mocks.
    for set_code, drafts, salt in tier["harness"]:
        s = set_code.format(hset=hset)
        cmd = [sys.executable, "tools/replay/run.py", "compare", s, "--drafts", str(drafts),
               "--note", f"[{a.tier}] {a.note}", *overrides] + (["--salt", salt] if salt else [])
        procs.append(subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True))
    for proc in procs:
        _, err = proc.communicate()
        if proc.returncode:
            print(f"{' '.join(proc.args[1:4])} failed:\n{err[-2000:]}")

    rows = [json.loads(l) for l in open(ROOT / "results" / "runs.jsonl")][before:]
    for r in rows:
        print(summary(r))


def summary(r):
    m = r["metrics"]
    if r["tool"] == "mock":
        d, v, u = m["v2_minus_upstream"], m["v2"], m["upstream"]
        o = v["open_colour"]
        pool = v.get("pool") or {}
        return (f"mock {r['set']} x{m['tables']}: v2 deck win rate {v.get('deck_wr')}, v2 lane score {v.get('lane_score')}, "
                f"lane@P2P1 {v['lane_matches_deck']['P2P1']:.0%} vs {u['lane_matches_deck']['P2P1']:.0%}, "
                f"open after P1/P2/P3 {o['P1']['top_open']:.0%}/{o['P2']['top_open']:.0%}/{o['P3']['top_open']:.0%}, lands {pool.get('lands')}")
    bk = m.get("by_kind", {})
    lands = bk.get("kind:dual land", {})
    return (f"harness {r['set']}: top-1 {m['top1_diff']['points']:+.1f} (SE {m['top1_diff']['se']}), "
            f"strong v2 {m['head']['strong_top1']}/{m['head']['strong_top3']} vs {m['base']['strong_top1']}/{m['base']['strong_top3']}, "
            f"regret slope {m['regret']['head']} vs {m['regret']['base']}, dual lands {lands.get('head_top1')}% vs {lands.get('base_top1')}%")


if __name__ == "__main__":
    main()
