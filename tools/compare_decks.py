"""What else is different about decks that are low (or high) on one field: every other field's average, against the rest.

    uv run python tools/compare_decks.py FRA role_creatures            # 2+ under the pair's median
    uv run python tools/compare_decks.py FRA role_removal --high       # 2+ over it
    uv run python tools/compare_decks.py FRA 2 --by 1 --skill 0.0      # 2-drops, 1+ under, all players
    uv run python tools/compare_decks.py FRA wins --won 6 --lost 2      # decks with 6+ wins against decks with 2 or fewer

From table 3 (tools/deck_counts.py) only: decks by --skill players (default 60%+) in two-colour pairs with at least
100 such decks; each field taken relative to its pair's median, so pairs that differ by nature do not show up as
compensation. Fields whose average moves (in cards, with z against the rest) are what those decks did instead; fields
that stay put do not go with it. Also each group's win rate beyond the deck model (wins - predicted).
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import companion  # noqa: E402,F401
from tools import deck_counts  # noqa: E402

TEXT = {"deck", "draft", "lane", "splash", "check", "skill", "games", "wins", "pred"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("set")
    p.add_argument("field", help="a table-3 field, or 'wins' to compare decks by their record (--won / --lost)")
    p.add_argument("--won", type=int, default=6, help="field wins: decks with at least this many wins ...")
    p.add_argument("--lost", type=int, default=2, help="... against decks with at most this many")
    p.add_argument("--by", type=float, default=2.0, help="cards under (or over, --high) the pair median")
    p.add_argument("--high", action="store_true")
    p.add_argument("--skill", type=float, default=0.6)
    p.add_argument("--top", type=int, default=25)
    a = p.parse_args(argv)
    rows = [r for r in deck_counts.read(a.set) if len(r["lane"]) == 2 and r["skill"] >= a.skill - 1e-9]
    fields = [k for k in rows[0] if k not in TEXT]
    by = defaultdict(list)
    for r in rows:
        by[r["lane"]].append(r)
    X, games, wins, pred = [], [], [], []
    for lane, rs in by.items():
        if len(rs) < 100:
            continue
        m = np.array([[r[f] for f in fields] for r in rs], float)
        X.append(m - np.median(m, 0))
        games += [r["games"] for r in rs]; wins += [r["wins"] for r in rs]; pred += [r["pred"] for r in rs]
    X, games, wins, pred = np.vstack(X), np.array(games), np.array(wins), np.array(pred)
    rest = np.ones(len(X), bool)
    if a.field == "wins":
        # a deck's games in the data are its event: wins counted there (17Lands public data: one row per game)
        sel, rest, i = wins >= a.won, wins <= a.lost, None
    else:
        i = fields.index(a.field)
        sel = X[:, i] >= a.by if a.high else X[:, i] <= -a.by
        rest = ~sel
    if sel.sum() < 30:
        sys.exit(f"only {sel.sum()} decks {'over' if a.high else 'under'} by {a.by:g}")
    diff = X[sel].mean(0) - X[rest].mean(0)
    z = diff / np.sqrt(X[sel].var(0) / sel.sum() + X[rest].var(0) / rest.sum() + 1e-12)
    beyond = lambda m: 100 * (wins[m].sum() - pred[m].sum()) / games[m].sum()
    if i is None:
        print(f"{a.set}: {sel.sum()} decks with {a.won}+ wins against {rest.sum()} with {a.lost} or fewer ({a.skill:.0%}+ players, of {len(X)})")
        print(f"predicted win rate (deck model, cards + shape + player): these {100 * pred[sel].sum() / games[sel].sum():.1f}%, "
              f"the others {100 * pred[rest].sum() / games[rest].sum():.1f}%")
    else:
        print(f"{a.set}: {sel.sum()} of {len(X)} decks ({a.skill:.0%}+ players) with {a.field} {'over' if a.high else 'under'} the pair median by "
              f"{a.by:g}+ ({diff[i]:+.1f} cards on average)")
        print(f"win rate beyond the deck model: these {beyond(sel):+.2f}, the rest {beyond(rest):+.2f} points")
    print(f"\n{'field':24} {'these vs rest':>14} {'z':>6}")
    order = [j for j in np.argsort(-np.abs(z)) if j != i][:a.top]
    if i is None:
        print("(fields relative to each pair's median; + = the winning decks have more)")
    for j in order:
        print(f"{fields[j]:24} {diff[j]:+14.2f} {z[j]:+6.1f}")


if __name__ == "__main__":
    main()
