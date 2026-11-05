"""Pre-registered: deep-main/thin-second two-colour decks vs balanced, FRA. See docs/podcast-test-split-FRA.md.
Run: nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels])' -c python3 analysis/podcast_split.py"""
import gzip, json, re, os
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
decks = json.load(gzip.open(f"{R}/analysis/out/decks_FRA.json.gz"))
cards = json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"]
by = {}
for c in cards.values():
    by[c["name"]] = c
    by.setdefault(c["name"].split(" // ")[0], c)

REMOVAL = re.compile(r"destroy target|exile target|deals? \d+ damage to target creature|deals? x damage to target creature"
                     r"|damage to any target|deals? \d+ damage to (?:target|any)|[-−]x/[-−]x|[-−]\d+/[-−]\d+|\bfights?\b", re.I)

def cls(c, L):
    """Colours a card forces on the lane, from the front-face cost; None if uncastable. Hybrid symbol with both
    options in the lane forces nothing (flexible, counted as gold)."""
    forced, coloured = set(), False
    for sym in re.findall(r"\{([^}]*)\}", c["mana_cost"].split(" // ")[0]):
        opts = set(re.findall(r"[WUBRG]", sym.replace("/P", "")))
        if not opts or "/P" in sym:
            continue
        coloured = True
        o = opts & L
        if not o:
            return None
        if len(o) == 1 and (len(opts) == 1 or len(o) < len(opts)):
            forced |= o
    return forced, coloured

rows, rem = [], []
for d in decks:
    if d["splash"] != "" or len(d["lane"]) != 2 or not d["deck_games"]:
        continue
    spells = []
    ok = True
    for n, k in d["counts"].items():
        c = by.get(n) or by.get(n.split(" // ")[0])
        if c is None:
            continue  # basics
        if "Land" in c["types"]:
            continue
        spells += [c] * k
    if not 22 <= len(spells) <= 24:
        continue
    L = set(d["lane"])
    cl = [cls(c, L) for c in spells]
    if any(x is None for x in cl):
        continue
    mono = {x: [c for c, (f, _) in zip(spells, cl) if f == {x}] for x in L}
    a, b = sorted(L, key=lambda x: -len(mono[x]))
    tot = len(mono[a]) + len(mono[b])
    if tot == 0:
        continue
    g = [c["deck_colors"]["All Decks"]["gihwr"] for c in spells]
    if min(g) <= 0:
        continue
    g = sorted(g, reverse=True)
    rows.append(dict(draft=d["draft"], lane="".join(sorted(L)), skill=d["skill"], gb=d["games_bucket"],
                     w=d["deck_wins"], l=d["deck_games"] - d["deck_wins"], n=d["deck_games"],
                     ms=len(mono[a]) / tot, gold=sum(len(f) == 2 or (not f and col) for f, col in cl),
                     q=np.mean(g), q10=np.mean(g[:10])))
    r = lambda lst: sum(bool(REMOVAL.search(c.get("oracle_text") or "")) for c in lst)
    if rows[-1]["ms"] >= 0.65:
        rem.append((r(mono[a]), len(mono[a]), r(mono[b]), len(mono[b])))
df = pd.DataFrame(rows)
df["bucket"] = pd.cut(df.ms, [0, .55, .65, .75, 1.01], right=False, labels=["<0.55", "0.55-0.65", "0.65-0.75", ">=0.75"])
print("decks in:", len(decks), "sample:", len(df), "drafts:", df.draft.nunique(), "games:", int(df.n.sum()))

g = df.groupby("bucket", observed=True).agg(decks=("w", "size"), games=("n", "sum"), wins=("w", "sum"), skill=("skill", "mean"),
                                           q=("q", "mean"), q10=("q10", "mean"), ms=("ms", "mean"))
g["WR"] = 100 * g.wins / g.games
print(g.round(3).to_string())

y = df[["w", "l"]]
def fit(q):
    X = pd.get_dummies(df[["lane", "gb"]].astype(str), drop_first=True).astype(float)
    X["ms"], X["q"], X["q10"], X["skill"], X["gold"] = df.ms, df.q, df.q10, df.skill, df.gold
    X = sm.add_constant(X)
    return sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.draft)[0]})
m = fit(None)
b, se, p = m.params["ms"], m.bse["ms"], m.pvalues["ms"]
k = 0.1 * 0.55 * 0.45 * 100  # logit coef -> WR points per +0.1 main share at 55%
print(f"main_share coef {b:.4f} (SE {se:.4f}) p={p:.4f}; WR pts per +0.1: {b*k:.3f} [{(b-1.96*se)*k:.3f}, {(b+1.96*se)*k:.3f}]")
print("quality coef", m.params[["q", "q10"]].round(4).to_dict(), "skill", round(m.params["skill"], 3))
a = np.array(rem).sum(0)
print(f"removal share (decks ms>=0.65, n={len(rem)}): main {a[0]}/{a[1]}={a[0]/a[1]:.3f}  thin {a[2]}/{a[3]}={a[2]/a[3]:.3f}")
