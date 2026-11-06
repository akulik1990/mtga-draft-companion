"""B2 + B4 (docs/test-plan-2026-11-07.md): class share of v2's top-1, the top drafter and the pack on the harness log; conditional logit of
pick choice on rating plus class dummies. Classes fixed before the run (see CLS). Streams nothing large: log is 17 MB.
Run: systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.numpy p.scipy])' -c python3 analysis/b2b4_class_share.py
"""
import gzip, json, os, re, sys
import numpy as np
from scipy.optimize import minimize
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, R)
from companion.advisor_v2 import config
OUT = open(f"{R}/analysis/out/b2b4_class_share.txt", "w")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n"); OUT.flush()

raw = json.load(open(f"{R}/data/Sets/FRA_PremierDraft_All_Data.json"))["card_ratings"]
roles = {r["name"]: set(r["tags"]) for r in json.load(open(f"{R}/companion/advisor_v2/sets/FRA/roles.json"))}
by = {}
for v in raw.values():
    n = v["name"].split(" // ")[0]
    if n in by and not int(v.get("isprimarycard", 1)): continue
    by[n] = v
gih = {n: v["deck_colors"]["All Decks"]["gihwr"] for n, v in by.items() if v["deck_colors"]["All Decks"]["gihwr"] > 0}
isl = lambda v: "Land" in v["types"]
# ---- classes (non-board rule copied from analysis/podcast_nonboard.py: draw/life/aura/trick, removal excluded)
rc = json.load(open(f"{R}/analysis/out/removal_classes_FRA.json"))
nb = set()
for n, v in by.items():
    t, ty = set(config.tags(v, "FRA")), v["types"]
    txt = re.sub(r"\([^)]*\)", "", v.get("oracle_text") or "").lower(); ro = roles.get(n, set())
    if isl(v) or "Creature" in ty or "Planeswalker" in ty or "removal" in t or "removal" in ro: continue
    if "Aura" in v["subtypes"] or "Equipment" in v["subtypes"]: nb.add(n)
    elif "combat_trick" in t and "Instant" in ty: nb.add(n)
    elif "empower jace" in txt: continue
    elif ("card_advantage" in t or "draw" in ro) and "fixing_ramp" not in t and not re.search(r"return[^.]*to the battlefield", txt) \
            and re.search(r"\bdraws?\b|surveil|return target|cards you own", txt): nb.add(n)
    elif ("lifegain" in t or "lifegain_enabler" in ro) and "gain" in txt and "life" in txt: nb.add(n)
CLS = {
    "6+ drops": {n for n, v in by.items() if not isl(v) and v["cmc"] >= 6},
    "non-board spells": nb,
    "conditional removal": {n for n, c in rc.items() if c["cls"] == "conditional"},
    "unconditional removal": {n for n, c in rc.items() if c["cls"] == "unconditional"},
    "damage removal": {n for n, c in rc.items() if c["cls"] == "damage"},
    "counterspells (CORE)": {"Countersculpt", "Icy Reception", "Precise Redaction", "Sublime Epiphany"},
    "Ways": {n for n in by if n.startswith("Way of the ")},
    "landcyclers": {n for n, v in by.items() if not isl(v) and "landcycling" in (v.get("oracle_text") or "").lower()},
}
names = list(CLS)
for k, s in CLS.items(): P(f"class {k}: {len(s)} cards")
P("non-board list:", sorted(nb))

# ---- picks
drafts = json.load(gzip.open(f"{R}/data/Temp/replay/FRA.top600.train.drafts.json.gz"))
human = {(d["id"], p[0], p[1]): (p[2], p[3]) for d in drafts for p in d["picks"]}
log = json.load(open(f"{R}/data/Temp/replay/logs/head-78e1953.json"))
BASICS = {'Plains', 'Island', 'Swamp', 'Mountain', 'Forest'}
rows = []  # (draft, pack_no, pick_no, human, v2, pack list)
miss = 0
for e in log:
    h = human.get(tuple(e[:3]))
    if h is None or not e[3]: miss += 1; continue
    rows.append((e[0], e[1], e[2], h[0], e[3][0][0], [c for c in h[1] if c not in BASICS]))  # basics in the pack are not candidates
P(f"log picks {len(log)}, matched {len(rows)}, unmatched {miss}; drafts {len({r[0] for r in rows})}")
bad = sum(1 for r in rows if r[3] not in r[5] or r[4] not in r[5]); P("human or v2 pick not a non-basic card in the pack:", bad)
rows = [r for r in rows if r[3] in r[5] and r[4] in r[5]]

def ind(n): return np.array([n in CLS[k] for k in names], float)
IND = {}
def I(n):
    if n not in IND: IND[n] = ind(n)
    return IND[n]

def ratio(num, den, cl):
    """clustered ratio estimator sum(num)/sum(den); returns pct, se (pct points)."""
    u, g = np.unique(cl, return_inverse=True)
    N = np.bincount(g, num); D = np.bincount(g, den); r = N.sum() / D.sum()
    res = N - r * D; v = (len(u) / (len(u) - 1)) * (res ** 2).sum() / D.sum() ** 2
    return 100 * r, 100 * np.sqrt(v)

def share_table(sel, label):
    S = [r for r in rows if sel(r)]
    cl = np.array([r[0] for r in S])
    hv = np.array([I(r[3]) for r in S]); vv = np.array([I(r[4]) for r in S])
    pk = np.array([np.mean([I(c) for c in r[5]], axis=0) for r in S])
    one = np.ones(len(S))
    P(f"\n## {label}: {len(S)} picks, {len(set(cl))} drafts. Shares in %, SE clustered by draft (ratio estimator). Diff = v2 minus human, points, 95% CI.")
    P(f"{'class':24s} {'pack':>6s} {'human':>6s} {'v2':>6s}   {'v2-human':>9s} {'CI95':>16s}  {'human-pack':>10s} {'CI95':>16s}")
    out = {}
    for j, k in enumerate(names):
        pm, _ = ratio(pk[:, j], one, cl); hm, _ = ratio(hv[:, j], one, cl); vm, _ = ratio(vv[:, j], one, cl)
        d, ds = ratio(vv[:, j] - hv[:, j], one, cl); hp, hs = ratio(hv[:, j] - pk[:, j], one, cl)
        flag = "FLAG" if abs(d) >= 2 else ""
        P(f"{k:24s} {pm:6.2f} {hm:6.2f} {vm:6.2f}   {d:+9.2f} [{d-1.96*ds:+6.2f},{d+1.96*ds:+6.2f}]  {hp:+10.2f} [{hp-1.96*hs:+6.2f},{hp+1.96*hs:+6.2f}] {flag}")
        out[k] = (d, ds)
    return out
res = {}
for lab, sel in (("P1 picks 1-5", lambda r: r[1] == 1 and r[2] <= 5), ("P1 picks 6-10", lambda r: r[1] == 1 and 6 <= r[2] <= 10),
                 ("P2-3 picks 1-5 (secondary)", lambda r: r[1] >= 2 and r[2] <= 5), ("P2-3 picks 6-10 (secondary)", lambda r: r[1] >= 2 and 6 <= r[2] <= 10)):
    res[lab] = share_table(sel, lab)

# ---- B4: conditional logit, utility = b*z(GIH WR) + class dummies, softmax over the pack. Human choice and v2 top-1 choice, same packs.
vals = np.array(list(gih.values())); mu, sd = vals.mean(), vals.std()
def clogit(S, who):
    """who 3 human, 4 v2. Returns beta, per-draft score influence (H^-1 s_d), names."""
    keep = S
    L = max(len(r[5]) for r in keep); K = 2 + len(names)
    X = np.zeros((len(keep), L, K)); M = np.zeros((len(keep), L), bool); y = np.zeros(len(keep), int)
    for i, r in enumerate(keep):
        for j, c in enumerate(r[5]):
            X[i, j, 0] = (gih[c] - mu) / sd if c in gih else 0.0; X[i, j, 1] = c not in gih; X[i, j, 2:] = I(c); M[i, j] = True
        y[i] = r[5].index(r[who])
    cl = np.array([r[0] for r in keep])
    def parts(b):
        u = np.where(M, X @ b, -1e9); u -= u.max(1, keepdims=True); p = np.exp(u); p /= p.sum(1, keepdims=True)
        m = np.einsum("il,ilk->ik", p, X); return p, m
    def f(b):
        p, m = parts(b); u = np.where(M, X @ b, -1e9); lse = np.log(np.exp(u - u.max(1, keepdims=True)).sum(1)) + u.max(1)
        return -(X[np.arange(len(y)), y] @ b - lse).sum()
    def g(b):
        p, m = parts(b); return -(X[np.arange(len(y)), y] - m).sum(0)
    b = minimize(f, np.zeros(K), jac=g, method="BFGS").x
    p, m = parts(b)
    H = np.einsum("il,ilk,ilj->kj", p, X, X) - np.einsum("ik,ij->kj", m, m)
    sc = X[np.arange(len(y)), y] - m
    return b, np.linalg.inv(H), sc, cl, len(keep)
def b4(sel, label):
    S = [r for r in rows if sel(r)]
    out = {}
    for who, nm in ((3, "human"), (4, "v2")):
        out[nm] = clogit(S, who)
    (bh, Hh, sh, ch, nh), (bv, Hv, sv, cv, nv) = out["human"], out["v2"]
    # restrict to common picks for the paired difference: both use same rows when the keep sets agree
    P(f"\n## B4 {label}: conditional logit, picks human {nh} / v2 {nv}. Coefficients in log-odds; rating = GIH WR in SD units (SD {sd:.2f} pts; cards without a GIH WR (about 20 in FRA, e.g. Precise Redaction) get z 0 plus a dummy). SE clustered by draft.")
    P(f"rating term: human {bh[0]:+.3f}, v2 {bv[0]:+.3f}")
    P(f"{'class':24s} {'human':>7s} {'CI95':>16s} {'v2':>7s} {'CI95':>16s} {'human-v2':>9s} {'CI95':>16s}")
    drafts_ = np.unique(np.concatenate([ch, cv])); idx = {d: i for i, d in enumerate(drafts_)}
    def infl(H, sc, cl):
        A = np.zeros((len(drafts_), sc.shape[1]))
        np.add.at(A, [idx[c] for c in cl], sc); return A @ H  # H^-1 s per draft (H symmetric)
    Ih, Iv = infl(Hh, sh, ch), infl(Hv, sv, cv)
    n = len(drafts_)
    for j, k in enumerate(["rating", "no-rating card"] + names):
        seh = np.sqrt((Ih[:, j] ** 2).sum()); sev = np.sqrt((Iv[:, j] ** 2).sum()); sed = np.sqrt(((Ih[:, j] - Iv[:, j]) ** 2).sum())
        d = bh[j] - bv[j]
        P(f"{k:24s} {bh[j]:+7.3f} [{bh[j]-1.96*seh:+6.3f},{bh[j]+1.96*seh:+6.3f}] {bv[j]:+7.3f} [{bv[j]-1.96*sev:+6.3f},{bv[j]+1.96*sev:+6.3f}] {d:+9.3f} [{d-1.96*sed:+6.3f},{d+1.96*sed:+6.3f}]")
for lab, sel in (("P1 picks 1-5", lambda r: r[1] == 1 and r[2] <= 5), ("P1 picks 1-10", lambda r: r[1] == 1 and r[2] <= 10),
                 ("P1 picks 6-10", lambda r: r[1] == 1 and 6 <= r[2] <= 10), ("all picks", lambda r: True)):
    b4(sel, lab)

# ---- side counts
unr = lambda c: c not in gih
P("\n## Side counts")
P("human picks of cards without GIH WR:", sum(unr(r[3]) for r in rows), " v2 top-1:", sum(unr(r[4]) for r in rows), " of", len(rows))
P("Precise Redaction: human picks", sum(r[3] == "Precise Redaction" for r in rows), " v2", sum(r[4] == "Precise Redaction" for r in rows), " in pack", sum("Precise Redaction" in r[5] for r in rows))
P("top-1 agreement v2 = human, picks 1-10 of P1:", round(100 * np.mean([r[3] == r[4] for r in rows if r[1] == 1 and r[2] <= 10]), 1), "%")
for k in ("counterspells (CORE)", "landcyclers", "unconditional removal"):
    P(k, "cards:", sorted(CLS[k]))

# ---- per-card breakdown of unconditional removal, P1 picks 1-5 (post hoc, descriptive: which cards carry the flag)
S5 = [r for r in rows if r[1] == 1 and r[2] <= 5]
P("\n## Unconditional removal by card, P1 picks 1-5 (post hoc, descriptive): card, in pack, human picks, v2 picks, GIH WR")
for c in sorted(CLS["unconditional removal"], key=lambda c: -sum(r[4] == c for r in S5)):
    P(f"  {c:28s} pack {sum(c in r[5] for r in S5):4d}  human {sum(r[3] == c for r in S5):3d}  v2 {sum(r[4] == c for r in S5):3d}  GIH {gih.get(c, 0):.1f}")
