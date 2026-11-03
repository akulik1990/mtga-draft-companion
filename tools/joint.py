"""Small joint tests: a base and many variants on the same 100-draft slice, a few at a time, one table.

    uv run python tools/joint.py FRA --random 6                  # 6 random joint settings of every GRID constant
    uv run python tools/joint.py FRA --around                    # each constant moved to its other grid values, the rest kept
    uv run python tools/joint.py FRA --try "PREF_WEIGHT=6 K=600" --try "BETA=1"   # named variants (all values together)
    options: --sample FRA.test100.drafts.json.gz (default FRA.train100, made from the 600-draft samples if missing),
             --v2 NAME=VALUE (the base itself), --grid NAME=a,b,c (replace a grid row), --jobs 5, --seed 11

Every variant is applied on top of the base, all its values together, and run through tools/replay/run.py top on the
same drafts; the table shows top-1 / top-3 and their difference from the base, plus pack 1 early picks (1-5), sorted
by top-1. ~4,200 picks per 100 drafts: one pick is ~0.024 points, differences under ~0.3 are noise. Nothing is
recorded in the ledger and nothing in the engine changes; apply what holds by hand (the set's fitted.json values).
This replaces the one-constant-at-a-time tools/tune.py (2026-11-03): constants work together, and ten small runs
say more than one large one.
"""

import argparse
import gzip
import json
import random
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Three values each around the FRA setting; STEER is left out (it does nothing under the Power/Fit split).
GRID = {
    "K": [300.0, 600.0, 1200.0],
    "COMMIT_BY": [12, 18, 24],
    "ALPHA": [0.5, 1.0, 2.0],
    "ALPHA_LATE": [1.0, 1.5, 2.5],
    "BETA": [1.0, 2.0, 4.0],
    "GAMMA": [0.0, 0.5, 1.0],
    "REPLACEMENT_SD": [2.0, 3.0, 4.0],
    "SPARSE_SD": [0.5, 1.0, 1.5],
    "PREF_WEIGHT": [4.0, 6.0, 8.0],
    "FIX_WEIGHT": [0.0, 0.25, 0.5],
}


def ensure_sample(set_code, name):
    """FRA.train100 / FRA.test100: the first 100 drafts of the 600-draft top-drafter samples."""
    sys.path.insert(0, str(ROOT))
    import companion  # noqa: F401
    from tools.replay.run import CACHE, top_sample_paths

    path = CACHE / name
    if not path.exists():
        split = "test" if ".test" in name else "train"
        drafts = json.load(gzip.open(top_sample_paths(set_code, 600)[split], "rt"))
        json.dump(drafts[:100], gzip.open(path, "wt"))
    return name


def run(set_code, sample, overrides):
    cmd = [sys.executable, "tools/replay/run.py", "top", set_code, "--sample", sample, "--no-record", "--json"]
    for k, v in overrides.items():
        cmd += ["--v2", f"{k}={v}"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    line = next((l for l in out.stdout.splitlines() if l.startswith("METRICS ")), None)
    if not line:
        raise RuntimeError(f"{overrides}: {out.stderr[-1500:]}")
    return json.loads(line[len("METRICS "):])


def parse(pairs):
    out = {}
    for p in pairs:
        for item in p.split():
            k, v = item.split("=", 1)
            out[k] = v
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("--sample", default=None, help="drafts file in the replay cache (default <SET>.train100.drafts.json.gz)")
    p.add_argument("--v2", action="append", default=[], help="base overrides NAME=VALUE")
    p.add_argument("--random", type=int, default=0, help="random joint settings of every grid constant")
    p.add_argument("--around", action="store_true", help="each grid constant moved to its other values, one per variant")
    p.add_argument("--try", dest="tries", action="append", default=[], help='a named variant: "NAME=V NAME=V"')
    p.add_argument("--grid", action="append", default=[], help="replace a grid row: NAME=a,b,c")
    p.add_argument("--jobs", type=int, default=5)
    p.add_argument("--seed", type=int, default=11)
    a = p.parse_args(argv)

    grid = dict(GRID)
    for g in a.grid:
        k, vals = g.split("=", 1)
        grid[k] = [float(x) for x in vals.split(",")]
    sample = ensure_sample(a.set, a.sample or f"{a.set}.train100.drafts.json.gz")
    base = parse(a.v2)
    variants = [("base", {})]
    variants += [(t, parse([t])) for t in a.tries]
    r = random.Random(a.seed)
    variants += [(f"random {i + 1}", {k: r.choice(v) for k, v in grid.items()}) for i in range(a.random)]
    if a.around:
        variants += [(f"{k}={v}", {k: v}) for k, vals in grid.items() for v in vals]
    with ThreadPoolExecutor(a.jobs) as pool:
        results = list(pool.map(lambda nv: (nv[0], nv[1], run(a.set, sample, {**base, **nv[1]})), variants))

    b = results[0][2]
    early = lambda m: (m.get("by_stage") or {}).get("P1 early", {}).get("top1")
    print(f"{a.set} on {sample} ({b.get('picks', '?')} picks), base {' '.join(a.v2) or 'as configured'}: "
          f"top-1 {b['top1']} (SE {b['top1_se']}), top-3 {b['top3']}, P1 picks 1-5 {early(b)}")
    print(f"{'variant':44} {'top-1':>6} {'d':>6} {'top-3':>6} {'d':>6} {'P1 1-5':>7}")
    for name, values, m in sorted(results[1:], key=lambda x: -x[2]["top1"]):
        label = name if not name.startswith("random") else name + ": " + " ".join(f"{k}={v}" for k, v in values.items())
        label = re.sub(r"=/\S*/", "=", label)  # file values by their name only
        print(f"{label[:44]:44} {m['top1']:6.2f} {m['top1'] - b['top1']:+6.2f} {m['top3']:6.2f} {m['top3'] - b['top3']:+6.2f} {early(m) or 0:7.2f}")
        if name.startswith("random") and len(label) > 44:
            print(f"    {label[44:]}")


if __name__ == "__main__":
    main()
