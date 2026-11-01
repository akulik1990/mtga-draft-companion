"""Fit v2's constants to top drafters: coordinate search on the train samples, judged on test.

    uv run python tools/tune.py --sets FRA TDM --passes 2 [--jobs 8]

Objective: wins-weighted top-1 agreement with top drafters (each pick weighted by its event's
wins + 1), averaged over the sets, on the `train` samples (tools/replay/run.py sample-top).
One constant at a time, a small grid around its current value, evaluated in parallel; a move
is kept only when it beats the current objective by more than one SE. The best set found is
then run once on the `test` samples and both are written to results/runs.jsonl (tool "tune").
Nothing in the engine is changed; apply the printed constants by hand if test confirms them.
"""

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# name: candidate multipliers or absolute values around the current one
GRID = {
    "K": [150.0, 300.0, 600.0],
    "STEER": [2.0, 4.0, 6.0],
    "COMMIT_BY": [12, 18, 24],
    "ALPHA": [0.5, 1.0, 2.0],
    "ALPHA_LATE": [1.0, 1.5, 2.5],
    "BETA": [0.0, 0.5, 1.0],
    "GAMMA": [0.0, 0.5, 1.0],
    "REPLACEMENT_SD": [1.5, 2.0, 3.0],
    "SPARSE_SD": [0.0, 0.5, 1.0],
    "SPLASH": [0.7, 0.85, 1.0],
    "SPEC_SPLASH": [0.85, 1.0, 1.15],
    "PREF_WEIGHT": [2.0, 4.0, 6.0],
    "FIX_WEIGHT": [0.25, 0.5, 1.0],
}


def evaluate(set_code, split, params):
    cmd = [sys.executable, "tools/replay/run.py", "top", set_code, "--split", split, "--no-record", "--json"]
    for k, v in params.items():
        cmd += ["--v2", f"{k}={v}"]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    line = next((l for l in out.stdout.splitlines() if l.startswith("METRICS ")), None)
    if not line:
        raise RuntimeError(f"{set_code} {params}: {out.stderr[-1500:]}")
    return json.loads(line[len("METRICS "):])


def objective(results):
    """Mean wins-weighted top-1 over sets, and an SE from the sets' top-1 SEs."""
    vals = [r["weighted_top1"] for r in results]
    se = (sum(r["top1_se"] ** 2 for r in results) ** 0.5) / len(results)
    return sum(vals) / len(vals), se


def score(params, sets, split, pool):
    results = list(pool.map(lambda s: evaluate(s, split, params), sets))
    return objective(results), results


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--sets", nargs="+", default=["FRA"])
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--only", nargs="*", help="tune only these constants")
    a = p.parse_args(argv)

    sys.path.insert(0, str(ROOT))
    import companion  # noqa: F401
    from companion.advisor_v2.engine import DraftAdvisorV2

    current = {k: getattr(DraftAdvisorV2, k) for k in GRID if hasattr(DraftAdvisorV2, k)}
    names = [k for k in current if not a.only or k in a.only]
    with ThreadPoolExecutor(a.jobs) as pool:
        (best, se), _ = score(current, a.sets, "train", pool)
        start = best
        print(f"start: weighted top-1 {best:.2f} (SE {se:.2f}) with {current}", flush=True)
        for n in range(a.passes):
            moved = False
            for name in names:
                cands = [v for v in GRID[name] if v != current[name]]
                with ThreadPoolExecutor(len(cands)) as outer:
                    trials = list(outer.map(lambda v: (v, score({**current, name: v}, a.sets, "train", pool)[0]), cands))
                v, (val, _) = max(trials, key=lambda t: t[1][0])
                print(f"  pass {n + 1} {name}: " + ", ".join(f"{c}={o[0]:.2f}" for c, o in trials) + f" (current {current[name]}={best:.2f})", flush=True)
                if val > best + se:
                    current[name], best, moved = v, val, True
                    print(f"    -> {name} = {v}, weighted top-1 {best:.2f}", flush=True)
            if not moved:
                break
        (test, test_se), test_results = score(current, a.sets, "test", pool)
        (base_test, _), base_results = score({k: getattr(DraftAdvisorV2, k) for k in current}, a.sets, "test", pool)

    print(f"\ntrain: {start:.2f} -> {best:.2f}; test: {base_test:.2f} -> {test:.2f} (SE {test_se:.2f})")
    print("constants:", json.dumps(current))
    from tools import results as ledger

    md = "\n".join([f"## Tuning on top drafters ({', '.join(a.sets)})", "",
                    f"Objective: wins-weighted top-1 agreement. Train {start:.2f} -> {best:.2f}; test {base_test:.2f} -> {test:.2f} (SE {test_se:.2f}).", "",
                    "Constants found: `" + json.dumps(current) + "`", "",
                    "| set | test top-1 before | after | test outcome lift before | after |", "|---|---|---|---|---|"] +
                   [f"| {s} | {b['top1']} | {t['top1']} | {b['outcome_lift']} | {t['outcome_lift']} |" for s, b, t in zip(a.sets, base_results, test_results)])
    ledger.record("tune", "+".join(a.sets), {"passes": a.passes, "grid": {k: GRID[k] for k in names}},
                  {"train": [start, best], "test": [base_test, test], "test_se": test_se, "constants": current,
                   "test_by_set": {s: {"before": b, "after": t} for s, b, t in zip(a.sets, base_results, test_results)}}, md,
                  "coordinate search on top-drafter train samples")


if __name__ == "__main__":
    main()
