"""Learn what top drafters prefer from their picks: one strength per card, from head-to-head choices.

    uv run python tools/learn_prefs.py FRA            # writes companion/advisor_v2/prefs_FRA.json

Every pick by a top drafter (60%+ over 50+ games, the filtered dump from tools/replay/run.py
sample-top) is a choice: these cards were in the pack, this one was taken. A conditional logit
P(take i | pack) = exp(s_i) / sum_j exp(s_j) is fitted by maximum likelihood (L2-regularised,
scipy L-BFGS), with each pick weighted by its event's wins + 1, so winners' choices count more.
Two fits: "early" from pack 1 picks 1-5, where the pool says little and colour matters least,
and "all" from every pick. Strengths are z-scored over cards seen at least 200 times.
The drafts in the top-drafter test sample are left out so tests on it stay honest.
"""

import argparse
import csv
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

REPLAY = DATA_DIR / "Temp" / "replay"


def read_choices(set_code, exclude):
    """[(stage, [pack card names], taken, weight)] from the filtered top-drafter dump."""
    path = REPLAY / f"draft_data_public.{set_code}.PremierDraft.top.csv.gz"
    out = []
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        pack_cols = [(i, h[len("pack_card_"):]) for i, h in enumerate(header) if h.startswith("pack_card_")]
        for row in reader:
            if row[col["draft_id"]] in exclude:
                continue
            pack = []
            for i, name in pack_cols:
                n = int(float(row[i] or 0))
                pack += [name] * n
            taken = row[col["pick"]]
            if taken not in pack or len(set(pack)) < 2:
                continue
            p, k = int(row[col["pack_number"]]) + 1, int(row[col["pick_number"]]) + 1
            wins = float(row[col["event_match_wins"]] or 0)
            out.append(("early" if p == 1 and k <= 5 else "rest", sorted(set(pack)), taken, wins + 1.0))
    return out


def fit(choices, l2=1.0):
    names = sorted({n for _, pack, _, _ in choices for n in pack})
    index = {n: i for i, n in enumerate(names)}
    width = max(len(pack) for _, pack, _, _ in choices)
    D = len(choices)
    idx = np.full((D, width), len(names))  # padded slot points at a dummy strength fixed at -inf
    mask = np.zeros((D, width), bool)
    chosen = np.zeros(D, int)
    w = np.zeros(D)
    seen = np.zeros(len(names))
    for d, (_, pack, taken, weight) in enumerate(choices):
        ids = [index[n] for n in pack]
        idx[d, :len(ids)] = ids
        mask[d, :len(ids)] = True
        chosen[d] = ids[pack.index(taken)]
        w[d] = weight
        seen[ids] += 1
    w = w / w.mean()

    def nll(s):
        full = np.append(s, 0.0)
        S = np.where(mask, full[idx], -np.inf)
        m = S.max(axis=1, keepdims=True)
        e = np.exp(S - m)
        Z = e.sum(axis=1)
        P = e / Z[:, None]
        ll = (s[chosen] - (m[:, 0] + np.log(Z))) @ w
        g = np.zeros(len(s) + 1)
        np.add.at(g, idx, -(P * w[:, None]))
        np.add.at(g, chosen, w)
        grad = -g[:-1] + 2 * l2 * s
        return -ll + l2 * (s @ s), grad

    res = minimize(nll, np.zeros(len(names)), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    s = res.x
    ok = seen >= 200
    mu, sd = s[ok].mean(), s[ok].std() or 1.0
    return {n: round(float((s[i] - mu) / sd), 4) for n, i in index.items() if seen[i] >= 20}, {"decisions": D, "cards": int(ok.sum()), "converged": bool(res.success)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set"); p.add_argument("--n", type=int, default=600)
    a = p.parse_args(argv)
    with gzip.open(REPLAY / f"{a.set}.top{a.n}.test.drafts.json.gz", "rt", encoding="utf-8") as f:
        exclude = {d["id"] for d in json.load(f)}
    choices = read_choices(a.set, exclude)
    early = [c for c in choices if c[0] == "early"]
    out = {"set": a.set, "excluded_test_drafts": len(exclude)}
    for label, rows in (("early", early), ("all", choices)):
        strengths, info = fit(rows)
        out[label] = strengths
        out[f"{label}_info"] = info
        top = sorted(strengths.items(), key=lambda kv: -kv[1])[:8]
        print(f"{label}: {info['decisions']} choices, {info['cards']} cards, converged {info['converged']}; strongest: " + ", ".join(f"{n} {v:+.2f}" for n, v in top))
    path = Path(companion.__file__).parent / "advisor_v2" / f"prefs_{a.set}.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
