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
        # For the weights-by-pick chart: each card's Power, its pool-weighted value, the signal for its colours, its wheel chance.
        extra = {}
        for c in pack:
            spell = "Land" not in (c.get("types") or [])
            colours = [x for x in (c.get("colors") or []) if x in self.signals] if spell else []
            extra[c.get("name")] = [self.power(c, pick), sum(p * self.value(c, lane) for lane, p in self.lanes.items()),
                                    sum(self.signals[x] for x in colours) / len(colours) if colours else 0.0]
        for r in recs:
            if r.card_name in extra:
                extra[r.card_name].append((r.wheel_chance or 0.0) / 100.0)
        last["extra"] = extra
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
                             "scores": last["scores"], "extra": last.get("extra", {})})
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


CHART = ["Power", "Fit", "Signal", "Wheel", "Synergy", "Needs", "Fixer", "Splash", "Mana value"]
NOTE = {"Power": "card's value in its own best pair, plus the learnt top-drafter strength (bombs land here)",
        "Fit": "card's value under the pairs the pool points to (castability, splash, land fixing value)",
        "Signal": "openness of the card's colours from what the passing neighbour sent",
        "Wheel": "chance the card comes back",
        "Synergy": "payoffs whose enablers are in the pool, and enablers for held payoffs (manual.json themes)",
        "Needs": "removal, creature and two-drop shortfall of the pool against top players' decks",
        "Fixer": "a land or card that fixes toward another colour",
        "Splash": "an off-colour single-pip card the pool already has fixing for",
        "Mana value": "the card's mana value"}


def cmd_chart(a):
    """Per pick of the draft, a conditional logit of top drafters' choices on Power, Fit, signal and the Fit parts;
    weights per SD of each input at that pick (how far one SD moves the odds of taking the card)."""
    rows = json.load(gzip.open(cache(a.set, a.split), "rt"))
    by = load_cards(a.set)
    tags = theme_tags(a.set)
    themes = list(tags)
    targets = (config.fitted(a.set, "decks") or {}).get("pairs") or {}
    data = {}
    for row in rows:
        extra = row.get("extra") or {}
        pool = [by[n] for n in row["pool"] if n in by]
        pack = [by[n] for n in row["pack_cards"] if n in by and n in extra and row["scores"].get(n, 0) > 0]
        if len(pack) < 2 or row["taken"] not in {c["name"] for c in pack}:
            continue
        lane = top_lane(pool)
        X, G, Y = data.setdefault((row["pack"], row["pick"]), ([], [], []))
        g = len(set(G))
        for c in pack:
            pw, fv, sig, wh = extra[c["name"]][:4]  # a duplicate in the pack appends its wheel twice
            x = features(c, pool, lane, themes, tags, targets, 0.0, 0.0)
            X.append([pw, fv, sig, wh, x[1] + x[2], x[4] + x[5] + x[6], x[9], x[8], x[10]])
            G.append(g)
            Y.append(1 if c["name"] == row["taken"] else 0)
    out = []
    for (pk, k), (X, G, Y) in sorted(data.items()):
        X, G, Y = np.array(X, float), np.array(G), np.array(Y)
        if len(set(G.tolist())) < 100:
            continue
        keep = X.std(0) > 0
        b = np.zeros(len(CHART)); se = np.zeros(len(CHART)); bw = np.zeros(len(CHART))
        raw, raw_se, sd_b, sd_se = fit_clogit(X[:, keep], G, Y)
        b[keep], se[keep], bw[keep] = sd_b, sd_se, raw
        out.append({"pack": pk, "pick": k, "picks": len(set(G.tolist())), "per_sd": [round(float(v), 3) for v in b],
                    "se": [round(float(v), 3) for v in se], "per_point": [round(float(v), 4) for v in bw]})
        print(f"P{pk}p{k:<2d} n={len(set(G.tolist())):4d} " + " ".join(f"{n[:6]}={v:+.2f}" for n, v in zip(CHART, b)))
    path = ROOT / "docs" / f"pick-weights-{a.set}.json"
    doc = {"set": a.set, "split": a.split, "inputs": CHART, "notes": NOTE, "picks": out}
    path.write_text(json.dumps(doc, indent=1))
    page = path.with_suffix(".html")
    page.write_text((ROOT / "tools" / "pick_weights.html").read_text().replace("__DATA__", json.dumps(doc)))
    print(f"wrote {path} and {page}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("collect", cmd_collect), ("fit", cmd_fit), ("chart", cmd_chart)):
        s = sub.add_parser(name)
        s.add_argument("set")
        s.add_argument("--split", default="train")
        s.set_defaults(fn=fn)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
