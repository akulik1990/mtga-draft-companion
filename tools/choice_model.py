"""What top drafters respond to beyond v2's score: a choice model over their picks.

    uv run python tools/choice_model.py collect FRA [--split train]   # replay once, save every pick with v2's scores
    uv run python tools/choice_model.py fit FRA [--split train]       # conditional logit per draft stage

collect: replays the top-drafter sample (tools/replay/run.py sample-top) through v2 and stores, per pick, the pack, the
pool, v2's score for each card and the card taken (<data>/Temp/replay/choices.<SET>.<split>.json.gz).

fit: at each pick the drafter chose one card from the pack. A conditional logit fits how much each feature adds to a
card's chance of being chosen, on top of v2's score (its own coefficient): synergy with the pool (the set's manual.json
themes, tagged by synergy.roles), a payoff with none of its enablers in the pool, deck needs projected to 23 spells
against top players' targets (two-drops, creatures, removal), colour (on the pool's top pair, splashable with fixing
already in the pool, a fixer) and mana value. Fitted separately for pack 1 picks 1-7, the middle picks and picks 12-14
of each pack, with standard errors from the observed information. A clearly positive coefficient is something v2 misses
at that stage.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from companion import DATA_DIR  # noqa: E402
from companion.advisor_v2 import config  # noqa: E402
from companion.advisor_v2.choice import FEATURES, STAGES, features, theme_tags, top_lane  # noqa: E402


def cache(set_code, split):
    return DATA_DIR / "Temp" / "replay" / f"choices.{set_code}.{split}.json.gz"


def cmd_collect(a):
    from companion.advisor_v2.engine import DraftAdvisorV2
    from tools.replay import run

    rows, last = [], {}
    evaluate = DraftAdvisorV2.evaluate_pack

    def capture(self, pack, pick, current_pack=1):
        recs = evaluate(self, pack, pick, current_pack)
        last["scores"] = {r.card_name: float(r.contextual_score) for r in recs}
        return recs

    DraftAdvisorV2.evaluate_pack = capture
    install = run.use_advisor

    def use_advisor(name):
        install(name)
        advise = run.replay.App.advise

        def wrapped(self, pack, pool, pack_no, pick_no):
            out = advise(self, pack, pool, pack_no, pick_no)
            taken = next((t for pk, k, t, *_ in self.current_draft["picks"] if (pk, k) == (pack_no, pick_no)), None)
            if taken and last.get("scores"):
                rows.append({"draft": self.current_draft.get("id"), "wins": self.current_draft.get("wins"), "pack": pack_no, "pick": pick_no,
                             "taken": taken, "pack_cards": [c["name"] for c in pack], "pool": [c["name"] for c in pool],
                             "scores": last["scores"]})
            return out

        run.replay.App.advise = wrapped

    run.use_advisor = use_advisor
    drafts = run.top_sample_paths(a.set, 600)[a.split]
    out = run.CACHE / f"choices.{a.set}.head.json"
    run.replay_one(a.set, drafts, "v2", out)
    out.unlink(missing_ok=True)
    with gzip.open(cache(a.set, a.split), "wt") as f:
        json.dump(rows, f)
    print(f"saved {len(rows)} picks to {cache(a.set, a.split)}")


def load_cards(set_code):
    data = json.loads((DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json").read_text())["card_ratings"]
    by = {}
    for c in data.values():
        for n in (c["name"], c["name"].split(" // ")[0]):
            by.setdefault(n, c)
    return by


def fit_clogit(X, groups, chosen):
    """Conditional logit: X rows, groups[i] = choice set id, chosen[i] = 1 for the card taken. (beta, se)."""
    order = np.argsort(groups, kind="stable")
    X, groups, chosen = X[order], groups[order], chosen[order]
    starts = np.r_[0, np.flatnonzero(np.diff(groups)) + 1]
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd

    def f(b):
        u = Z @ b
        m = np.maximum.reduceat(u, starts)
        e = np.exp(u - np.repeat(m, np.diff(np.r_[starts, len(u)])))
        s = np.add.reduceat(e, starts)
        p = e / np.repeat(s, np.diff(np.r_[starts, len(u)]))
        ll = (u[chosen == 1]).sum() - (np.log(s) + m).sum()
        g = Z[chosen == 1].sum(0) - (p[:, None] * Z).sum(0)
        return -ll, -g

    res = minimize(f, np.zeros(Z.shape[1]), jac=True, method="BFGS")
    se = np.sqrt(np.diag(res.hess_inv))
    return res.x / sd, se / sd, res.x, se  # per raw unit, and per SD


def cmd_fit(a):
    rows = json.load(gzip.open(cache(a.set, a.split), "rt"))
    by = load_cards(a.set)
    tags = theme_tags(a.set)
    themes = list(tags)
    targets = (config.fitted(a.set, "decks") or {}).get("pairs") or {}
    data = {s: ([], [], []) for s in STAGES}
    gid = 0
    for row in rows:
        stage = next(s for s, test in STAGES.items() if test(row["pack"], row["pick"]))
        pool = [by[n] for n in row["pool"] if n in by]
        pack = [by[n] for n in row["pack_cards"] if n in by and n in row["scores"]]
        if len(pack) < 2 or row["taken"] not in {c["name"] for c in pack}:
            continue
        lane = top_lane(pool)
        best = max(row["scores"][c["name"]] for c in pack)
        X, G, Y = data[stage]
        for c in pack:
            X.append(features(c, pool, lane, themes, tags, targets, row["scores"][c["name"]], best))
            G.append(gid)
            Y.append(1 if c["name"] == row["taken"] else 0)
        gid += 1
    out = {}
    for stage, (X, G, Y) in data.items():
        X, G, Y = np.array(X, float), np.array(G), np.array(Y)
        beta, se, beta_sd, se_sd = fit_clogit(X, G, Y)
        n = len(set(G.tolist()))
        print(f"\n{stage}: {n} picks")
        print(f"  {'feature':14s} {'per unit':>9s} {'SE':>7s} {'z':>6s}   {'per SD':>7s}  {'mean':>6s}")
        for i, name in enumerate(FEATURES):
            print(f"  {name:14s} {beta[i]:+9.3f} {se[i]:7.3f} {beta[i] / se[i]:+6.1f}   {beta_sd[i]:+7.3f}  {X[:, i].mean():6.2f}")
        out[stage] = {"picks": n, "features": FEATURES, "beta": [round(float(x), 4) for x in beta], "se": [round(float(x), 4) for x in se]}
    config.write_fitted(a.set, "choices", {"split": a.split, "stages": out},
                        "tools/choice_model.py fit: conditional logit of top drafters' picks on v2's score plus synergy, deck needs, colour, mana value")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("collect", cmd_collect), ("fit", cmd_fit)):
        s = sub.add_parser(name)
        s.add_argument("set")
        s.add_argument("--split", default="train")
        s.set_defaults(fn=fn)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
