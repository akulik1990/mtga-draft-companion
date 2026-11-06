"""How often Rewrite Regrets is in splash decks vs non-splash decks, by player skill (FRA Premier decks, decks_FRA.json.gz).

    python3 analysis/rewrite_regrets_splash.py
Splash deck = the deck has a splash colour (the file's `splash` field). 'Plays black' = B is a main colour or the splash colour.
"""
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARD = "Rewrite Regrets"
decks = [d for d in json.load(gzip.open(ROOT / "analysis" / "out" / "decks_FRA.json.gz")) if d["format"] == "PremierDraft"]


def group(s):
    return "good (>=0.60)" if s >= 0.60 else "average (0.52-0.58)" if s >= 0.52 else "weak (<0.52)"


def ci(k, n):
    if not n:
        return "-"
    p = k / n
    h = 1.96 * math.sqrt(p * (1 - p) / n)
    return f"{100 * p:.1f}% [{100 * max(0, p - h):.1f}, {100 * min(1, p + h):.1f}] ({k}/{n})"


rows = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for d in decks:
    g = group(d["skill"])
    has = CARD in d["counts"]
    splash = bool(d.get("splash"))
    plays_b = "B" in d["lane"] or "B" in (d.get("splash") or "")
    b_main = "B" in d["lane"]
    b_splash = "B" in (d.get("splash") or "")
    for gg in (g, "all"):
        for key, ok in (("splash decks", splash), ("non-splash decks", not splash),
                        ("splash decks that play black", splash and plays_b), ("non-splash decks that play black", (not splash) and plays_b),
                        ("black is a main colour", b_main), ("black is only the splash", b_splash and not b_main)):
            if ok:
                rows[gg][key][1] += 1
                rows[gg][key][0] += has
out = [f"{CARD}: share of Premier decks that play it ({len(decks)} decks)", ""]
for gg in ("good (>=0.60)", "average (0.52-0.58)", "weak (<0.52)", "all"):
    out.append(f"{gg}")
    for key in ("splash decks", "non-splash decks", "splash decks that play black", "non-splash decks that play black", "black is a main colour", "black is only the splash"):
        k, n = rows[gg][key]
        out.append(f"  {key:36s} {ci(k, n)}")
    out.append("")
text = "\n".join(out)
(ROOT / "analysis" / "out" / "rewrite_regrets_splash.txt").write_text(text + "\n")
print(text)
