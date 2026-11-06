"""Video test 2: do mana sinks protect a deck in the late game? ("the player who runs out of things to spend mana on loses")
Sink tag (from oracle text, reminder text stripped), fixed before any outcome was looked at:
  ACT  = tools/card_table.py mana_sink regex (a mana-only activated ability "{..}: ", or "activate only as a sorcery"), minus false positives:
         FP1 the ability sacrifices / exiles the card itself or is Exhaust (once per game): Budding Insurgent, Cryotheory Adept, The Echoverse Fulcrum, Room of Refuge, Liliana the Repentant.
  X    = {X} in the mana cost.   FB = flashback printed on the card (Stingcaster Mage only grants it: excluded).
  MAN  = Murmuring Volume (named by the speaker) and Eye of Ugin ({7},{T}: tutor), added by hand because their costs include {T} and miss the regex.
  Not sinks: landcyclers (cost {2}, early), empower-Jace N cards (an expensive cast, not an outlet), Ward.
Primary count SINKS = ACT + X + FB + MAN (spells and lands); sensitivity ACT only (the repo tag). Deck GLM exactly as video_counters.py (two-colour decks, 60%+ skill,
binomial on wins, SEs clustered by draft, controls mean lane GIH WR, creatures, splash, lane, skill, games bucket); effects in WR points per sink at 55%, buckets 0 / 1-2 / 3+.
Game level: deck fixed effects (within-deck), won ~ flood (opening-hand lands >= 5) and flood x sink bucket, kept 7-card hands (0 mulligans), decks with >= 4 games, SEs clustered by draft.
Run: nix shell --impure --expr 'with import <nixpkgs> {}; python3.withPackages (p: [p.pandas p.numpy p.statsmodels p.scipy])' -c python3 analysis/video_sinks.py"""
import gzip, os, re, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video_counters as vc
from video_counters import R, OUT, K, K1, cards, load, fit, row, diff, CTRL

RX = re.compile(r"\{\d*[wubrgcx]*\}(?:\{[wubrgcx\d]\})*: |activate only as a sorcery")
FP1 = {"Budding Insurgent", "Cryotheory Adept", "The Echoverse Fulcrum", "Room of Refuge", "Liliana the Repentant"}
MAN = {"Murmuring Volume", "Eye of Ugin"}
lines = []
def P(s=""): print(s); lines.append(s)

def tag(by):
    out = {}
    for n, v in by.items():
        t = re.sub(r"\([^)]*\)", "", v.get("oracle_text") or "").lower(); mc = v.get("mana_cost") or ""
        f = set()
        if RX.search(t) and n not in FP1: f.add("ACT")
        if "{X}" in mc: f.add("X")
        if re.search(r"^flashback|\nflashback", t, re.M) and n != "Stingcaster Mage": f.add("FB")
        if n in MAN: f.add("MAN")
        if f: out[n] = f
    return out

def bucket(s): return np.where(s == 0, "b0", np.where(s <= 2, "b12", "b3p"))

def main():
    by = cards(); T = tag(by)
    P("# Sink list (derived, for review)\n\n| card | type | cost | tags | GIH WR % | rule matched |\n|---|---|---|---|---|---|")
    for n in sorted(T):
        v = by[n]; g = (v["deck_colors"].get("All Decks") or {}).get("gihwr")
        P(f"| {n} | {'/'.join(v['types'])} | {v.get('mana_cost') or '-'} | {','.join(sorted(T[n]))} | {g if g else 'n/a'} | {(RX.search((v.get('oracle_text') or '').lower()) or [''])[0] if 'ACT' in T[n] else ''} |")
    P(f"\n{len(T)} cards: " + ", ".join(f"{k} {sum(k in x for x in T.values())}" for k in ("ACT", "X", "FB", "MAN")) + f"; dropped by FP1: {sorted(FP1)}")
    def extra(c, by, d):
        return dict(sinks=sum(k for n, k in c.items() if n in T), act=sum(k for n, k in c.items() if "ACT" in T.get(n, ())),
                    big=sum(k for n, k in c.items() if n in by and "Land" not in by[n]["types"] and (by[n]["cmc"] or 0) >= 5))
    ALL = load(extra); ALL["bk"] = bucket(ALL.sinks)
    def run(df, tag_):
        P(f"\n### {tag_}: {len(df)} decks, {df.draft.nunique()} drafts, {int(df.g.sum())} games")
        P(f"sinks mean {df.sinks.mean():.2f} (sd {df.sinks.std():.2f}); buckets 0 / 1-2 / 3+: {(df.bk=='b0').sum()} / {(df.bk=='b12').sum()} / {(df.bk=='b3p').sum()} decks; 5+ cmc spells mean {df.big.mean():.2f}")
        m = fit(df, f"y ~ sinks + {CTRL}"); P(f"\nper sink (all sink types)\n\n| term | pts/sink | 95% CI | p |\n|---|---|---|---|\n{row(m, 'sinks')}")
        m = fit(df, f"y ~ act + {CTRL}"); P(row(m, "act").replace("| act |", "| ACT only |"))
        m = fit(df, f"y ~ sinks + big + {CTRL}"); P(row(m, "sinks").replace("| sinks |", "| sinks, 5+ cmc count controlled |"))
        m = fit(df, f"y ~ C(bk, Treatment('b0')) + {CTRL}")
        P("\nbuckets vs 0 sinks (pts, 24.75 x coefficient)\n\n| bucket | pts | 95% CI | p |\n|---|---|---|---|")
        for k, lab in (("b12", "1-2"), ("b3p", "3+")):
            x = f"C(bk, Treatment('b0'))[T.{k}]"; P(row(m, x, K1).replace(x, lab))
    for lbl, df in (("PRIMARY skill>=0.60", ALL[ALL.skill >= .60]), ("SENSITIVITY all skills", ALL)):
        P(f"\n## {lbl}"); run(df, "all drafts")
        if lbl.startswith("PRIMARY"):
            o = {d: i for i, d in enumerate(sorted(df.draft.unique()))}; h = df.draft.map(o) % 2
            for k in (0, 1): run(df[h == k], f"replication half {k + 1}")
    # game level
    G = pd.read_csv(f"{R}/analysis/out/hand_games_FRA.csv.gz")
    src = f"{R}/data/Temp/synergy/game_data_public.FRA.PremierDraft.csv.gz"
    head = list(pd.read_csv(src, nrows=0).columns)
    cols = [c for c in head if c.startswith("deck_") and c[5:].split(" // ")[0] in T]
    D = pd.read_csv(src, usecols=["draft_id", "build_index", "match_number", "game_number"] + cols)
    D["sinks"] = D[cols].fillna(0).to_numpy() @ np.array([1.0] * len(cols))
    G = G.merge(D[["draft_id", "build_index", "match_number", "game_number", "sinks"]], on=["draft_id", "build_index", "match_number", "game_number"], how="left")
    P(f"\n## Flood x sinks, game level\n\nsink columns matched in game data: {len(cols)} of {len(T)} tagged cards; games {len(G)}, unmatched sinks {int(G.sinks.isna().sum())}")
    G["deck"] = G.draft_id + "_" + G.build_index.astype(str)
    G = G[(G.num_mulligans == 0) & G.sinks.notna()]; G = G[G.groupby("deck").deck.transform("size") >= 4].copy()
    G["bk"] = bucket(G.sinks); G["flood"] = (G.hand_lands >= 5).astype(float); G["screw"] = (G.hand_lands <= 2).astype(float)
    G["f12"] = G.flood * (G.bk == "b12"); G["f3p"] = G.flood * (G.bk == "b3p")
    G["s12"] = G.screw * (G.bk == "b12"); G["s3p"] = G.screw * (G.bk == "b3p")
    G["fs"] = G.flood * G.sinks; G["ss"] = G.screw * G.sinks; G["sk"] = G.user_game_win_rate_bucket
    def within(df, xs, tag_):
        y = df.won.astype(float); X = df[xs]
        Xd = X - X.groupby(df.deck).transform("mean"); yd = y - y.groupby(df.deck).transform("mean")
        m = sm.OLS(yd, Xd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.draft_id)[0]})
        P(f"\n{tag_}: {len(df)} games, {df.deck.nunique()} decks, {df.draft_id.nunique()} drafts\n\n| term | pts | 95% CI | p |\n|---|---|---|---|")
        for x in xs:
            b, se = 100 * m.params[x], 100 * m.bse[x]; P(f"| {x} | {b:+.2f} | [{b-1.96*se:+.2f}, {b+1.96*se:+.2f}] | {m.pvalues[x]:.3g} |")
        return m
    for lbl, df in (("all skills", G), ("skill>=0.60", G[G.sk >= .60])):
        P(f"\n### {lbl}: share of games with flood (>=5 opening lands) by sink bucket: " + ", ".join(f"{b} {df[df.bk==b].flood.mean():.3f} (n {int((df.bk==b).sum())})" for b in ("b0", "b12", "b3p")))
        for b in ("b0", "b12", "b3p"):
            within(df[df.bk == b], ["flood", "screw", "on_play"], f"{lbl}, bucket {b}: effect of flood / screw vs 3-4 opening lands (within deck)")
        m = within(df, ["flood", "f12", "f3p", "screw", "s12", "s3p", "on_play"], f"{lbl}, pooled interaction (f12, f3p = extra flood cost vs 0 sinks)")
        m = within(df, ["flood", "fs", "screw", "ss", "on_play"], f"{lbl}, flood x sinks as a count (fs = extra flood effect per sink)")
    # exploratory: whole-game flood. drawn_* is the whole game (no turn stamp), so lands seen are compared with what the deck predicts for the cards seen.
    dr = [c for c in head if c.startswith("drawn_")]
    isl = np.array([("Land" in by[c[6:].split(" // ")[0]]["types"]) if c[6:].split(" // ")[0] in by else False for c in dr], float)
    Dr = pd.read_csv(src, usecols=["draft_id", "build_index", "match_number", "game_number", "num_turns"] + dr)
    A = Dr[dr].fillna(0).to_numpy()
    Dr["drawn"] = A.sum(1); Dr["drawn_lands"] = A @ isl
    E = G.merge(Dr[["draft_id", "build_index", "match_number", "game_number", "drawn", "drawn_lands", "num_turns"]], on=["draft_id", "build_index", "match_number", "game_number"], how="inner")
    E["seen"] = 7 + E.drawn; E["excess"] = E.hand_lands + E.drawn_lands - E.seen * E.deck_lands / 40
    E = E[E.seen >= 15].copy(); E["e12"] = E.excess * (E.bk == "b12"); E["e3p"] = E.excess * (E.bk == "b3p"); E["es"] = E.excess * E.sinks
    P(f"\n## Exploratory whole-game flood (lands seen minus deck_lands/40 x cards seen), games with >= 8 draws: {len(E)} games; excess mean {E.excess.mean():+.2f}, sd {E.excess.std():.2f}\n"
      "Caveats: drawn_* has no turn stamp; games with >= 8 draws are a post-outcome selection (length depends on the game going the long way); 'seen' is controlled but is itself a consequence of play. Descriptive only.")
    for lbl, df in (("all skills", E), ("skill>=0.60", E[E.sk >= .60])):
        within(df, ["excess", "e12", "e3p", "seen", "on_play"], f"{lbl}: per extra land seen above expectation (excess), by sink bucket (e12, e3p = difference vs 0 sinks); WR pts per land")
        within(df, ["excess", "es", "seen", "on_play"], f"{lbl}: excess x sink count (es = change in the per-land effect per extra sink)")
        for b in ("b0", "b12", "b3p"):
            within(df[df.bk == b], ["excess", "seen", "on_play"], f"{lbl}, bucket {b}")
    open(f"{OUT}/video_css_sinks.txt", "w").write("\n".join(lines))

if __name__ == "__main__":
    main()
