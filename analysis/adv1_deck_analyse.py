"""Analysis of data/Temp/adv1/deck_full.json as pre-registered in adv1_deck_run.py. Output: analysis/out/adv1_deck_results.txt"""
import json, random, statistics as S, collections as C
R = "/home/morris/Projects/mtga-draft-companion/"
rows = json.load(open(R + "data/Temp/adv1/deck_full.json")); B = 4000; rng = random.Random(1)
ok = [r for r in rows if r["v2"] and r["pl"]]
print(f"drafts {len(rows)}, with both builds {len(ok)}; v2 decks not 40 cards: {sum(sum(r['v2']['deck'].values())!=40 for r in ok)}, player-pool builds: {sum(sum(r['pl']['deck'].values())!=40 for r in ok)}; mean pool overlap v2/player {S.mean(r['pool_overlap'] for r in ok):.1f} of {S.mean(r['n_picks'] for r in ok):.0f}")
def ci(vals):
    n = len(vals); ms = sorted(S.mean(rng.choices(vals, k=n)) for _ in range(B)); return ms[int(.025*B)], ms[int(.975*B)]
def rep(label, vals):
    lo, hi = ci(vals); print(f"  {label:44s} n={len(vals):3d} mean {S.mean(vals):+.2f} [{lo:+.2f}, {hi:+.2f}]  >=0 in {sum(v >= 0 for v in vals)/len(vals):.0%}  >0 in {sum(v > 0 for v in vals)/len(vals):.0%}  sd {S.pstdev(vals):.2f}")
for grp, sel in (("ALL", ok), ("splash drafts", [r for r in ok if r["kind"] == "splash"]), ("three-colour drafts", [r for r in ok if r["kind"] == "three"])):
    print(f"\n== {grp} (yardstick WR, percentage points) ==")
    print(f"  mean WR: v2 build {S.mean(r['v2']['wr'] for r in sel):.2f}, player-pool build {S.mean(r['pl']['wr'] for r in sel):.2f}, registered deck {S.mean(r['reg_wr'] for r in sel):.2f}")
    rep("PRIMARY v2 build - player-pool build", [r["v2"]["wr"] - r["pl"]["wr"] for r in sel])
    rep("(b) v2 build - player registered deck", [r["v2"]["wr"] - r["reg_wr"] for r in sel])
    rep("player-pool build - registered deck", [r["pl"]["wr"] - r["reg_wr"] for r in sel])
print("\n== colours of the built / registered deck ==")
def tag(lane, spl): return ("three-base" if len(lane) >= 3 else "") + ("splash" if spl else "") or "two-colour"
for grp, sel in (("ALL", ok), ("splash drafts", [r for r in ok if r["kind"] == "splash"]), ("three-colour drafts", [r for r in ok if r["kind"] == "three"])):
    print(f" {grp}")
    for lab, f in (("player registered (decks file lane/splash)", lambda r: (r["reg_lane"], r["reg_splash"])),
                   ("player registered (mana_identity)", lambda r: (r["reg_main"], r["reg_ms"])),
                   ("player-pool build (builder label)", lambda r: (r["pl"]["lane"], r["pl"]["splash"] if False else ("x" if "+" in r["pl"]["label"] else ""))),
                   ("v2-pool build (builder label)", lambda r: (r["v2"]["lane"], "x" if "+" in r["v2"]["label"] else "")),
                   ("player-pool build (mana_identity)", lambda r: (r["pl"]["main"], r["pl"]["splash"])),
                   ("v2-pool build (mana_identity)", lambda r: (r["v2"]["main"], r["v2"]["splash"]))):
        c = C.Counter(tag(*f(r)) for r in sel); n = len(sel)
        print(f"   {lab:44s} " + "  ".join(f"{k} {v}/{n}" for k, v in sorted(c.items())))
    print(f"   any colour beyond the pair, mana_identity: v2 {sum(bool(r['v2']['splash']) for r in sel)}/{len(sel)}, player-pool {sum(bool(r['pl']['splash']) for r in sel)}/{len(sel)}, registered {sum(bool(r['reg_ms']) for r in sel)}/{len(sel)}")
print("\n== v2 pool composition vs the player's registered main colours ==")
print(f"  v2 build main pair equals registered pair (mana_identity): {sum(set(r['v2']['main'])==set(r['reg_main']) for r in ok)}/{len(ok)}; player-pool build: {sum(set(r['pl']['main'])==set(r['reg_main']) for r in ok)}/{len(ok)}")
# does the WR gap depend on the colour structure of v2's build
print("\n== primary difference by whether v2's build splashes (post hoc, descriptive) ==")
for lab, sel in (("v2 build has splash/third", [r for r in ok if r["v2"]["splash"] or len(r["v2"]["lane"]) >= 3]), ("v2 build plain pair", [r for r in ok if not r["v2"]["splash"] and len(r["v2"]["lane"]) < 3])):
    if sel: rep(lab, [r["v2"]["wr"] - r["pl"]["wr"] for r in sel])
