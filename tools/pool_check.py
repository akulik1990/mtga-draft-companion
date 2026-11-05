"""Pool-conditional check: "the pool had no more" against "left in the sideboard".

    uv run python tools/pool_check.py FRA extract    # one pass over the three game files, caches sideboards (about 2 min)
    uv run python tools/pool_check.py FRA            # analysis, about 3 min; writes <data>/tables/SET/13_pool.csv

Table 10 compares decks by what they play. A deck with 3 removal spells may have had no more (a pool problem) or left some in the
sideboard (a build choice, the closest the data gets to a bad build, and the only part a builder controls). Design as in
tools/criteria_fields.py (100+ games players, 2-3 base colours, all games, same controls). For each deck: castable sideboard cards in its own
colours (base + splash): removal (role_removal), finishers (cmc 5+, GIH WR 1 SD over the set mean), dual lands (two or more of the deck's
colours), and castable sideboard spells overall (pool depth).
Spec A (cells): deck count bin x (sideboard has none | has one or more), effect in points vs the reference cell, and the difference
left - forced with a cluster-robust SE. Spec B (additive): the table's deck-count bins plus sideboard-count bins held together.
Variants: "controlled" (as table 10) and "+ depth" (also holds the number of castable sideboard spells).
Associations: who leaves removal in the sideboard also chose other cards, so a gap is a cost of the choice only if those cards were not better.
"""

import gzip
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from criteria_lib import *  # noqa: E402,F403
from criteria_search import build, write  # noqa: E402

FORMATS = ("PremierDraft", "PickTwoDraft", "TradDraft")


def cache_path(set_code):
    return DATA_DIR / "Temp" / f"sideboards_{set_code}.pkl"


def extract(set_code):
    out = {}
    for fmt in FORMATS:
        t0 = time.time()
        path = DATA_DIR / "Temp" / "synergy" / f"game_data_public.{set_code}.{fmt}.csv.gz"
        with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
            header = next(csv.reader([fh.readline()]))
            assert header.index("draft_id") == 2 and header.index("build_index") == 5, header[:8]
            side = [(i, h[10:]) for i, h in enumerate(header) if h.startswith("sideboard_")]
            n = 0
            for line in fh:
                p = line.split(",", 6)
                key = (fmt, p[2], p[5])
                if key in out:
                    continue
                row = next(csv.reader([line]))
                out[key] = {name: int(float(row[i])) for i, name in side if row[i] and float(row[i]) > 0}
                n += 1
        print(f"{fmt}: {n} builds ({time.time() - t0:.0f}s)", flush=True)
    with open(cache_path(set_code), "wb") as f:
        pickle.dump(out, f)
    print(f"{len(out)} builds -> {cache_path(set_code)}")


def num(v):
    return float(v) if not isinstance(v, str) else 0.0


def side_features(set_code, D):
    """Per deck: castable sideboard removal, finishers, dual lands, spells (in the deck's own colours)."""
    import json

    import companion.advisor_v2.engine  # noqa: F401
    from companion.advisor_v2 import config
    from companion.advisor_v2.lanes import castable

    config.activate(set_code)
    SB = pickle.loads(cache_path(set_code).read_bytes())
    table = card_table(set_code)
    names = list(table)
    gw = np.array([num(table[n].get("gihwr")) for n in names])
    sp = np.array([1 - num(table[n].get("land")) for n in names])
    gw = gw[sp > 0]
    gm, gs = gw[gw > 0].mean(), gw[gw > 0].std()
    thr = gm + gs
    rec = {}
    for c in json.loads((DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json").read_text())["card_ratings"].values():
        for n in (c["name"], c["name"].split(" // ")[0]):
            rec.setdefault(n, c)
    out = {k: np.zeros(len(D["w"])) for k in ("rm", "fin", "dual", "spells")}
    miss = 0
    for i, (fg, r) in enumerate(zip(D["fg"], D["rows"])):
        side = SB.get((fg[0], fg[1], fg[2]))
        if side is None:
            miss += 1
            continue
        lane = r["combo"].split("+")[0] + (r.get("splash") or "")
        for name, k in side.items():
            t, c = table.get(name), rec.get(name)
            if t is None or c is None:
                continue
            if num(t.get("land")):
                if not num(t.get("land_basic")) and sum(num(t.get(f"land_{x}")) > 0 for x in lane) >= 2:
                    out["dual"][i] += k
            elif castable(c, lane):
                out["spells"][i] += k
                out["rm"][i] += k * (num(t.get("role_removal")) > 0)
                out["fin"][i] += k * (num(t.get("cmc")) >= 5 and num(t.get("gihwr")) >= thr)
    print(f"sideboards joined: {len(D['w']) - miss} of {len(D['w'])} decks; finisher threshold GIH WR {100 * thr:.1f}" if thr < 1 else f"joined {len(D['w']) - miss}; threshold {thr:.1f}", flush=True)
    return out


LAM = {}


def run(Xc, bg, B, D, key):
    lam_bg = LAM.setdefault(key, choose_lam(Xc, bg, D, grid=(300, 1000, 3000)))
    Xall = np.hstack([Xc, bg, B])
    lam = lams(Xc.shape[1], Xall.shape[1] - Xc.shape[1], lam_bg)
    k0 = Xc.shape[1] + bg.shape[1]
    lam[k0:] = 10.0
    b = fit(Xall, D["w"], D["n"], lam, np.ones(len(D["w"]), bool))
    V = sandwich(Xall, D["w"], D["n"], b, lam, D["draft"])
    return b[k0:], V[k0:, k0:]


def main():
    set_code = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    if len(sys.argv) > 2 and sys.argv[2] == "extract":
        return extract(set_code)
    t0 = time.time()
    D = load_data(set_code, "all")
    Xc = context(D, skill=True)
    sets, structure, quality, cols, base, combos, F = build(D)
    cols = dict(cols)
    for k, v in F.items():
        cols.setdefault(k, v)
    sf = side_features(set_code, D)
    Q = np.hstack([sets["quality"], sets["colour"]])
    S = sets["structure"]
    depth = standardise(np.minimum(sf["spells"], 12)[:, None])
    print(f"{len(D['w'])} decks, setup {time.time() - t0:.0f}s", flush=True)
    out = []
    ones = np.ones(len(D["w"]), bool)

    FIELDS = {  # label: (deck field, sideboard feature, bins lo..hi, fields left out, cell spec: {deck value: split?}, reference cell)
        "removal": ("removal", "rm", (3, 9), {"role_removal", "rm_cheap", "rm_late"}, {3: True, 4: True, 5: True, 6: False, 7: False, 8: False, 9: False}, (6, "")),
        "finishers": ("finishers", "fin", (0, 3), {"finishers"}, {0: True, 1: True, 2: True, 3: False}, (0, "forced")),
        "dual lands": ("land_multi", "dual", (0, 6), {"land_multi", "land_mono"}, None, None),
    }
    for label, (field, side_key, (lo, hi), left_out, spec, ref_cell) in FIELDS.items():
        dv = np.clip(np.round(np.asarray(cols[field], float)), lo, hi).astype(int)
        sv = sf[side_key]
        keep = [i for i, f in enumerate(structure) if f not in left_out]
        print(f"\n== {label}: sideboard holds one or more castable in the deck's colours in {100 * (sv > 0).mean():.0f}% of decks", flush=True)
        for variant, bg in (("controlled", np.hstack([Q, S[:, keep]])), ("+ depth", np.hstack([Q, S[:, keep], depth]))):
            if spec:  # spec A: cells
                cell = np.array([(d, ("left" if s > 0 else "forced") if spec[d] else "") for d, s in zip(dv, sv)], dtype=object)
                ids = sorted(set(map(tuple, cell)), key=lambda c: (c[0], c[1]))
                cnt = {c: int(sum(1 for x in cell if tuple(x) == c)) for c in ids}
                use = [c for c in ids if c != ref_cell and cnt[c] >= 30]
                B = np.column_stack([np.array([tuple(x) == c for x in cell], float) for c in use])
                b, V = run(Xc, bg, B, D, (label, variant))
                pos = {c: j for j, c in enumerate(use)}
                eff = lambda c: (PTS * b[pos[c]], PTS * np.sqrt(V[pos[c], pos[c]])) if c in pos else (0.0, 0.0)
                for c in ids:
                    e, s = eff(c)
                    out.append(dict(field=label, spec="A cells", variant=variant, deck=c[0], sideboard=c[1] or "any", decks=cnt[c], points=round(e, 2), se=round(s, 2)))
                print(f"  [{variant}] deck {label} x sideboard (points vs {ref_cell}; diff = left - forced):", flush=True)
                for d in sorted(spec):
                    if not spec[d]:
                        e, s = eff((d, ""))
                        print(f"    {d}{'+' if d == hi else ''}: {e:+.2f} (SE {s:.2f}), {cnt.get((d, ''), 0)} decks", flush=True)
                        continue
                    (ef, sf_), (el, sl) = eff((d, "forced")), eff((d, "left"))
                    dx = np.zeros(len(use))
                    if (d, "left") in pos:
                        dx[pos[(d, "left")]] += 1
                    if (d, "forced") in pos:
                        dx[pos[(d, "forced")]] -= 1
                    ds = PTS * np.sqrt(dx @ V @ dx)
                    out.append(dict(field=label, spec="A diff", variant=variant, deck=d, sideboard="left - forced", decks=cnt.get((d, "left"), 0), points=round(el - ef, 2), se=round(ds, 2)))
                    print(f"    {d}: forced {ef:+.2f} ({cnt.get((d, 'forced'), 0)} decks), left {el:+.2f} ({cnt.get((d, 'left'), 0)}); left - forced {el - ef:+.2f} (SE {ds:.2f})", flush=True)
            # spec B: additive
            cntd = np.bincount(dv - lo, minlength=hi - lo + 1)
            ref = int(np.argmax(cntd))
            sb = np.minimum(sv, 3).astype(int)
            cols_b, lab_b = [], []
            for i in range(hi - lo + 1):
                if i != ref and cntd[i] >= 30:
                    cols_b.append((dv - lo == i).astype(float)); lab_b.append(("deck", lo + i, int(cntd[i])))
            for v in (1, 2, 3):
                if (sb == v).sum() >= 30:
                    cols_b.append((sb == v).astype(float)); lab_b.append(("sideboard", v, int((sb == v).sum())))
            b, V = run(Xc, bg, np.column_stack(cols_b), D, (label, variant, "B"))
            print(f"  [{variant}] additive (deck bins vs {lo + ref}, sideboard 1 / 2 / 3+ vs none):", "  ".join(
                f"{k[:1]}{v}{'+' if v == hi and k == 'deck' else ''}:{PTS * b[j]:+.1f}" for j, (k, v, n) in enumerate(lab_b)), flush=True)
            for j, (k, v, n) in enumerate(lab_b):
                out.append(dict(field=label, spec="B additive", variant=variant, deck=v if k == "deck" else "", sideboard=v if k == "sideboard" else "", decks=n,
                                points=round(PTS * b[j], 2), se=round(PTS * np.sqrt(V[j, j]), 2)))
        # descriptive: how many decks short of the field had one in the sideboard
        short = {"removal": (3, 4, 5), "finishers": (0, 1, 2), "dual lands": ()}[label]
        for d in short:
            m = dv == d if d != lo else dv <= lo
            print(f"  deck at {d}: {int(m.sum())} decks, sideboard has more {100 * (sv[m] > 0).mean():.0f}%, mean sideboard {sv[m].mean():.2f}, raw win rate {100 * D['w'][m].sum() / D['n'][m].sum():.1f}%", flush=True)
            out.append(dict(field=label, spec="share", variant="", deck=d, sideboard="has more", decks=int(m.sum()), points=round(100 * (sv[m] > 0).mean(), 1), se=""))
    write(tables.path(set_code, "13_pool.csv"), out)
    print(f"\nwrote {tables.path(set_code, '13_pool.csv')} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
