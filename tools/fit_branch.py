"""Learn when top drafters leave their colours: fit v2's branch rule on a top-drafter train sample.

    uv run python tools/fit_branch.py FRA [--winners] [--write]

For every pick by a top drafter where v2's Power and Fit orders disagree (P = strongest card if you
end up in its colours, C = best card for the colours the pool points to), record the gap
power(P) - power(C), how strong C is on its own (power(C) minus the format's mean GIH WR), the pool
size, and whether the drafter took P or C. Per pool-size stage, a two-variable logistic model
P(took P | took P or C) = sigmoid(a + b*gap + c*c_strength) says when they branch: early and when the
in-colour option is weak, a smaller gap is enough. --winners fits only on events that went 5-7 wins.
--write saves the coefficients to the branch section of companion/advisor_v2/sets/<SET>/fitted.json (branch_colour with --full), which the engine reads.
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR, VENDOR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402

sys.path.insert(0, str(VENDOR / "Tools" / "AdvisorReplay"))


def logistic(X, y, l2=1e-3, iters=50):
    """Newton's method for logistic regression with a small ridge; returns (coefficients, standard errors)."""
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(X.shape[1])
        g = X.T @ (y - p) - l2 * w
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-X @ w))
    H = X.T @ (X * (p * (1 - p))[:, None]) + l2 * np.eye(X.shape[1])
    return w, np.sqrt(np.diag(np.linalg.inv(H)))


def collect(set_code, split, n, winners_only):
    from replay import App

    from companion.advisor_v2.engine import DraftAdvisorV2, stage_of

    app = App(str(DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json"))
    path = DATA_DIR / "Temp" / "replay" / f"{set_code}.top{n}.{split}.drafts.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        drafts = json.load(f)
    rows = []
    DraftAdvisorV2.SPLIT = True
    for d in drafts:
        if winners_only and (d.get("wins") or 0) < 5:
            continue
        seen = []
        for p, k, taken, pack_names, pool_names in d["picks"]:
            pack, pool = app.cards(pack_names), app.cards(pool_names)
            seen.append((p, k, pack))
            human = app.cards([taken])
            if not human or len({c["name"] for c in pack}) < 2:
                continue
            adv = DraftAdvisorV2(app.metrics, pool, seen_packs=list(seen))
            adv.evaluate_pack(pack, k, current_pack=p)
            b = (adv.split or {}).get("branch")
            if not b:
                continue
            took = human[0]["name"]
            if took not in (b["card"], b["over"]):
                continue
            by_name = {c["name"]: c for c in pack}
            c_strength = adv.power(by_name[b["over"]], k) - adv.mean
            shared, committed = adv.branch_features(by_name[b["card"]])
            rows.append({"stage": b["stage"], "gap": b["gap"], "c_strength": c_strength, "took_power": took == b["card"],
                         "shared": shared, "committed": committed,
                         "spells": sum(1 for c in pool if "Land" not in (c.get("types") or []))})
    return rows


def fit(rows, full=False):
    by_stage = defaultdict(list)
    for r in rows:
        by_stage[r["stage"]].append(r)
    out = {}
    for stage, rs in by_stage.items():
        y = np.array([r["took_power"] for r in rs], float)
        if len(rs) < 40 or y.sum() < 10 or (1 - y).sum() < 10:
            out[stage] = {"n": len(rs), "took_power": int(y.sum()), "skipped": "too few conflicts"}
            continue
        cols = [np.ones(len(rs)), [r["gap"] for r in rs], [r["c_strength"] for r in rs]]
        if full:
            cols += [[r["shared"] for r in rs], [r["committed"] for r in rs]]
        w, se = logistic(np.column_stack(cols), y)
        out[stage] = {k: round(float(v), 4) for k, v in zip("abcde", w)}
        out[stage].update({"se": [round(float(x), 4) for x in se], "n": len(rs), "took_power": int(y.sum())})
    return out


def gap_needed(coef, c_strength):
    """The gap at which the model says 50/50, for a given in-colour strength."""
    if not coef.get("b") or coef["b"] <= 0:
        return None
    return -(coef["a"] + coef["c"] * c_strength) / coef["b"]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set"); p.add_argument("--split", default="train"); p.add_argument("--n", type=int, default=600)
    p.add_argument("--winners", action="store_true"); p.add_argument("--write", action="store_true")
    p.add_argument("--full", action="store_true", help="also fit whether the strong card keeps a colour of the top pair (d) and how settled the pool is (e)")
    p.add_argument("--out", help="write here to this file instead of the set's fitted.json (alternatives to measure)")
    p.add_argument("--v2", action="append", default=[], metavar="NAME=VALUE", help="engine overrides while collecting, as in tools/mock_draft.py")
    a = p.parse_args(argv)
    from tools.mock_draft import apply_overrides
    apply_overrides(a.v2)
    rows = collect(a.set, a.split, a.n, a.winners)
    coefs = fit(rows, a.full)
    order = ["0", "1-2", "3-5", "6-9", "10-14", "15-22", "23+"]
    print(f"{a.set} {a.split}{' winners' if a.winners else ''}: {len(rows)} picks where Power and Fit disagreed and the drafter took one of the two")
    print("stage    picks  took Power   gap needed if in-colour is weak / average / strong (GIH WR points)")
    for st in order:
        c = coefs.get(st)
        if not c:
            continue
        if "skipped" in c:
            print(f"{st:7s} {c['n']:6d}  {c['took_power']:5d}       {c['skipped']}")
            continue
        g = [gap_needed(c, s) for s in (-2.0, 0.0, 2.0)]
        print(f"{st:7s} {c['n']:6d}  {c['took_power']:5d}       " + " / ".join("-" if x is None else f"{x:+.1f}" for x in g) +
              f"   (gap coef {c['b']:+.2f} ± {c['se'][1]:.2f}, in-colour coef {c['c']:+.2f} ± {c['se'][2]:.2f}"
              + (f", keeps a colour {c['d']:+.2f} ± {c['se'][3]:.2f}, settled {c['e']:+.2f} ± {c['se'][4]:.2f}" if "d" in c else "") + ")")
    if a.write:
        doc = {"set": a.set, "fitted_on": f"top{a.n}.{a.split}" + (" winners" if a.winners else ""),
               "model": "P(take Power) = sigmoid(a + b*gap + c*c_strength" + (" + d*keeps_colour + e*settled" if a.full else "") + "), c_strength = power(C) - format mean",
               "overrides": a.v2, "stages": coefs}
        if a.out:
            out = Path(a.out)
            out.write_text(json.dumps(doc, indent=1))
        else:
            out = config.write_fitted(a.set, "branch_colour" if a.full else "branch", doc,
                                      f"tools/fit_branch.py {a.set}{' --winners' if a.winners else ''}{' --full' if a.full else ''}: when top drafters switch to the stronger card")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
