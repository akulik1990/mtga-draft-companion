"""Re-report B1 (removal by class) and B5 (splash castability) by player skill group (CLAUDE.md evidence rule, 2026-11-06).
Reuses analysis/b1_within_removal.py and analysis/b5_splash_cast.py unchanged: their source is exec'd with only the split definitions replaced
(and the cache loader extended by user_n_games_bucket), so designs, placebos and bootstraps are identical to the originals.
  python3 analysis/skill_groups_within.py meta|b1|b5     (meta: stream user_n_games_bucket in CSV row order into NG; b1/b5 need the original caches in SCR)
Groups by user_game_win_rate_bucket: good >= 0.60, average 0.52-0.58, weak < 0.52. Primary population user_n_games_bucket >= 100; all players secondary."""
import os, re, sys
import numpy as np, pandas as pd
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCR = "/tmp/claude-1000/-home-morris-Projects-mtga-draft-companion/1d2827d1-4fa6-42b1-9fc1-592e21c156f3/scratchpad"
NG = f"{SCR}/ngames.npz"; GD = f"{R}/data/Temp/synergy/game_data_public.FRA.PremierDraft.csv.gz"
GROUPS = [("good (>=0.60)", lambda s: s >= 0.595), ("average (0.52-0.58)", lambda s: (s >= 0.515) & (s <= 0.585)), ("weak (<0.52)", lambda s: s < 0.515)]
mode = sys.argv[1]
if mode == "meta":
    M = pd.concat([c for c in pd.read_csv(GD, usecols=["draft_id", "build_index", "user_n_games_bucket", "user_game_win_rate_bucket"], chunksize=200000)], ignore_index=True)
    np.savez(NG, ng=M.user_n_games_bucket.fillna(0).values.astype(np.int16), skill=M.user_game_win_rate_bucket.values.astype(np.float32), draft=M.draft_id.values.astype(str), build=M.build_index.values.astype(np.int16))
    print("ngames cache", len(M)); sys.exit()
EXTRA = open(__file__).read().split("# ----" + " EXTRA")[1]
sys.argv = ["x", "run"]
def load(path):
    return open(f"{R}/analysis/{path}").read().replace('if sys.argv[1] == "cache":', "if False:")
g = {"__file__": f"{R}/analysis/{'b1_within_removal.py' if mode == 'b1' else 'b5_splash_cast.py'}", "__name__": "skg", "GROUPS": GROUPS, "NGFILE": NG}
if mode == "b1":
    s = load("b1_within_removal.py")
    s = s.replace('skill = Z["skill"]', 'skill = Z["skill"]; _N = np.load(NGFILE, allow_pickle=True); ng = _N["ng"]; assert (np.asarray(_N["draft"]) == draft).all() and (_N["build"] == build).all() and np.allclose(_N["skill"], skill, equal_nan=True)', 1)
    s = re.sub(r"SPLITS = \[.*?\]\n", "SPLITS = [('all decks', np.ones(len(won), bool))]\nfor _pop, _pm in (('ng>=100', ng >= 100), ('all players', np.ones(len(won), bool))):\n"
               "    for _gn, _gf in GROUPS: SPLITS.append((f'{_pop} {_gn}', _pm & _gf(skill)))\nSPLITS += [('ng>=100 good, on the play', (ng >= 100) & (skill >= 0.595) & (play == 1)), ('ng>=100 good, on the draw', (ng >= 100) & (skill >= 0.595) & (play == 0))]\n", s, count=1, flags=re.S)
    s = s.split("# Halves: adjusted")[0].split("# differences vs unconditional")[0]
    s = s.replace("\\n# Differences from unconditional", "")
    exec(compile(s + EXTRA.split("# ----" + " B5")[0], "b1", "exec"), g)
else:
    s = load("b5_splash_cast.py")
    s = s.replace('draft = np.asarray(Z["draft"], dtype=str)[sel];', '_N = np.load(NGFILE, allow_pickle=True); assert np.allclose(_N["skill"], Z["skill"], equal_nan=True) and (_N["build"] == Z["build"]).all(); SKs = Z["skill"][sel]; NGs = _N["ng"][sel]; draft = np.asarray(Z["draft"], dtype=str)[sel];', 1)
    s = s.replace("codes, _ = pd.factorize(codes)\ndcode,", "SKs, NGs = SKs[o], NGs[o]\ncodes, _ = pd.factorize(codes)\ndcode,", 1)
    s = re.sub(r"CELLS = \{.*?\}\nRES = \{\}", "CELLS = {'all': np.ones(G, bool)}\nfor _pop, _pm in (('ng>=100', NGs >= 100), ('all players', np.ones(G, bool))):\n    for _gn, _gf in GROUPS: CELLS[f'{_pop} {_gn}'] = _pm & _gf(SKs)\nRES = {}", s, count=1, flags=re.S)
    s = s.split("# pooled single/double")[0]
    exec(compile(s + EXTRA.split("# ----" + " B5")[1], "b5", "exec"), g)
sys.exit()
# ---- EXTRA
# B1 tail: good minus weak per class (adjusted points; SE = joint bootstrap of the difference, placebo SDs added in quadrature)
def gw(pop):
    P(f"\n# Good minus weak, {pop} (adjusted pts per card; 95% CI = +/-1.96 SE)\n\n| class | good adj (SE) | average adj (SE) | weak adj (SE) | good minus weak | SE | 95% CI | z |\n|---|---|---|---|---|---|---|---|")
    for c in CLASSES:
        r = [RES.get((f"{pop} {gn}", c)) for gn, _ in GROUPS]
        if not all(r): P(f"| {c} | too few |"); continue
        d = r[0]["adj"] - r[2]["adj"]; se = float(np.hypot(np.nanstd(r[0]["bs"] - r[2]["bs"], ddof=1), np.hypot(r[0]["ps"], r[2]["ps"]))); h = 1.96 * se
        P(f"| {c} | {r[0]['adj']:+.2f} ({r[0]['ase']:.2f}) | {r[1]['adj']:+.2f} ({r[1]['ase']:.2f}) | {r[2]['adj']:+.2f} ({r[2]['ase']:.2f}) | {d:+.2f} | {se:.2f} | [{d-h:+.2f}, {d+h:+.2f}] | {d/se:+.1f} |")
P("\n# SKILL GROUPS: summary"); gw("ng>=100"); gw("all players")
open(f"{OUT}/skill_groups_within_b1.txt", "w").write("\n".join(lines)); P("done")
# ---- B5
def adjd(cn, i): e, bs, pl = RES[cn]; return e[i] - np.nanmean(pl[:, i]), bs[:, i], pl[:, i]
def cell(cn, i, j=None):
    """adjusted estimate for term i, or contrast i minus j; returns (adj, SE)"""
    e, bs, pl = RES[cn]
    if j is None: a = e[i] - np.nanmean(pl[:, i]); se = np.hypot(np.nanstd(bs[:, i], ddof=1), np.nanstd(pl[:, i], ddof=1))
    else: a = (e[i] - e[j]) - np.nanmean(pl[:, i] - pl[:, j]); se = np.hypot(np.nanstd(bs[:, i] - bs[:, j], ddof=1), np.nanstd(pl[:, i] - pl[:, j], ddof=1))
    return a, se
def gwd(pop, i, j=None):
    cg, cw = f"{pop} {GROUPS[0][0]}", f"{pop} {GROUPS[2][0]}"
    (ag, sg_), (aw, sw_) = cell(cg, i, j), cell(cw, i, j)
    bsd = (RES[cg][1][:, i] - (RES[cg][1][:, j] if j is not None else 0)) - (RES[cw][1][:, i] - (RES[cw][1][:, j] if j is not None else 0))
    pd_ = np.hypot(np.nanstd(RES[cg][2][:, i] - (RES[cg][2][:, j] if j is not None else 0), ddof=1), np.nanstd(RES[cw][2][:, i] - (RES[cw][2][:, j] if j is not None else 0), ddof=1))
    se = np.hypot(np.nanstd(bsd, ddof=1), pd_); return ag - aw, se
f2 = lambda a, s: f"{a:+.2f} ({s:.2f})"
def b5table(pop):
    cns = ["all"] if pop == "all" else [f"{pop} {gn}" for gn, _ in GROUPS]
    P(f"\n## {pop}: adjusted pts per splashed card in hand (SE), by splash-colour land sources in hand\n\n| term | " + " | ".join(cns) + " | good minus weak [95% CI] |\n|---|" + "---|" * (len(cns) + 1))
    for i, nme in enumerate(NAMES):
        row = " | ".join(f2(*cell(cn, i)) for cn in cns)
        if pop != "all": d, se = gwd(pop, i); row += f" | {d:+.2f} [{d-1.96*se:+.2f}, {d+1.96*se:+.2f}]"
        else: row += " | "
        P(f"| {nme} | {row} |")
    for t, o in (("single", 0), ("double", 3)):
        for lab, i, j in (("1 source minus 0", o + 1, o), ("2+ sources minus 1", o + 2, o + 1), ("2+ sources minus 0", o + 2, o)):
            row = " | ".join(f2(*cell(cn, i, j)) for cn in cns)
            if pop != "all": d, se = gwd(pop, i, j); row += f" | {d:+.2f} [{d-1.96*se:+.2f}, {d+1.96*se:+.2f}]"
            else: row += " | "
            P(f"| {t}: {lab} | {row} |")
P("\n# SKILL GROUPS: B5 summary (adjusted = estimate minus placebo mean; games and decks per cell in the log header)")
for cn, mk in CELLS.items(): P(f"cell {cn}: games {int(mk.sum())}, decks {len(np.unique(codes[mk]))}, games with single-pip splash card {int((nSP[mk]>0).sum())}, double-pip {int((nDP[mk]>0).sum())}")
b5table("all"); b5table("ng>=100"); b5table("all players")
open(f"{OUT}/skill_groups_within_b5.txt", "w").write("\n".join(lines)); P("done")
