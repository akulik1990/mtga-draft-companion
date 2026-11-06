"""FRA cards that give a way to put or cast a card without its colours' mana (rules text families; nothing run on games).

    env MTGA_COMPANION_DATA=<repo>/data uv run python analysis/alt_cast_routes.py
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import video_draft as V  # noqa: E402

ds = "FRA_PremierDraft_All_Data.json"
metrics, cards = V.load_set(ds)
names = sorted({c["name"] for c in json.load(open(ROOT / "data" / "Sets" / ds))["card_ratings"].values()})
info = cards(names)
FAMILIES = {
    "reanimate to battlefield": r"from (?:your|a|any) graveyard to the battlefield|return target [^.]*card[^.]* from your graveyard to the battlefield|put target [^.]*card from a graveyard onto the battlefield",
    "free cast (no mana cost)": r"without paying (?:its|their) mana cost",
    "cast from graveyard or exile": r"you may cast [^.]* from (?:your graveyard|exile)|cast (?:it|that card|them) from exile|copies? of [^.]*graveyard",
    "mana of any colour": r"mana of any (?:one )?colou?r|one mana of any type|add one mana of the chosen",
    "treasure / mana token": r"create (?:a|an|two|three) treasure|heartwood token",
    "basic landcycling / land search": r"basic landcycling|search your library for (?:a|up to \w+) basic land",
    "copy a spell or permanent": r"create a token that's a copy|copy target (?:instant|sorcery|spell|creature|permanent)|you may cast the copy|discover \d",
}
out = []
for fam, pat in FAMILIES.items():
    hit = [c for c in info if re.search(pat, (c.get("oracle_text") or "").lower())]
    out.append((fam, hit))
text = []
for fam, hit in out:
    text.append(f"## {fam}: {len(hit)} cards")
    for c in sorted(hit, key=lambda c: c["name"]):
        text.append(f"- {c['name']} ({''.join(c.get('colors') or []) or 'C'}, {c.get('rarity', '?')[0]}): " + re.sub(r"\s+", " ", (c.get("oracle_text") or ""))[:150])
    text.append("")
(ROOT / "analysis" / "out" / "alt_cast_routes.txt").write_text("\n".join(text))
for fam, hit in out:
    print(f"{fam}: {len(hit)}  " + ", ".join(c["name"] for c in sorted(hit, key=lambda c: c["name"]))[:420])
