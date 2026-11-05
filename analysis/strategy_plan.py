"""What each pair's plan needs, for the overlay (Akshin, 2026-11-05: show synergy on the colour pair instead of scoring it).
Median enablers, payoffs and top payoffs in FRA trophy decks of each pair (one deck for the whole Premier run), stored as the
fitted "strategy_plan" section. Tags: companion/advisor_v2/sets/FRA/roles.json; UR's enabler (spells) is every noncreature spell.

    uv run python analysis/strategy_plan.py
"""
import gzip, json, sys
from collections import Counter
from pathlib import Path
from statistics import median

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R))
from companion.advisor_v2 import config  # noqa: E402

PLANS = {  # docs/archetypes-FRA.md (official archetype names; WU merged into Jace, WG lifegain, BG ETB value: Akshin)
    "WU": ("Jace's Mastery", "surveil and empower Jace, then attack", "jace_enabler", "jace_payoff"),
    "UG": ("Jace's Mastery", "empower Jace, draw value from the token", "jace_enabler", "jace_payoff"),
    "UB": ("Theorix Graveyard Math", "get seven or more cards in your graveyard", "mill", "gy_payoff"),
    "BR": ("Stingerquill Face Burn", "direct damage to opponents and creatures", "ping_enabler", "ping_payoff"),
    "RG": ("Konstrari Mana Ramp", "Heartwood tokens cast big top end early", "ramp", "ramp_payoff"),
    "WG": ("Vigorbloom Life Gain", "gain life, grow your board", "lifegain_enabler", "lifegain_payoff"),
    "WB": ("Liliana's Attrition", "small creatures, profit from deaths on both sides", "death_enabler", "death_payoff"),
    "UR": ("Chandra's Prowess", "cards that care about noncreature spells", "spells", "spells_payoff"),
    "BG": ("Garruk's Bestiary", "value creatures: enter-the-battlefield effects and ways to reuse them", "etb_enabler", "etb_payoff"),
    "WR": ("Ajani's Army", "+1/+1 counters on aggressive creatures", "counters_enabler", "counters_payoff"),
}
WANTS = {  # what sort of cards each side of the plan is (Akshin: describe, do not list cards); from docs/card-role-rules.md
    "WU": ("cards that empower Jace, scry or surveil", "cards that reward a Jace on the board or each surveil"),
    "UG": ("cards that empower Jace", "cards that use the Jace token's loyalty or reward a planeswalker"),
    "UB": ("cards that mill you or put your own cards in the graveyard", "cards that get better with seven or more cards in your graveyard, and flashback"),
    "BR": ("spells and creatures that deal direct damage", "cards that reward damaging the opponent or targeting their side"),
    "RG": ("Heartwood tokens, mana creatures, landcyclers and cost reducers", "strong cards costing six or more"),
    "WG": ("cards that gain life", "cards that trigger whenever you gain life"),
    "WB": ("cheap creatures, tokens and sacrifice outlets", "cards that trigger when creatures die"),
    "UR": ("instants, sorceries and other noncreature spells", "creatures that trigger on noncreature spells"),
    "BG": ("creatures with value when they enter", "cards that reuse enter effects or reward creatures entering"),
    "WR": ("cards that put +1/+1 counters on creatures", "cards that reward creatures with counters, and creatures that grow"),
}
front = lambda n: n.split(" // ")[0]
roles = {r["name"]: set(r["tags"]) for r in json.loads((R / "companion/advisor_v2/sets/FRA/roles.json").read_text())}
cards = {}
for c in json.loads((R / "data/Sets/FRA_PremierDraft_All_Data.json").read_text())["card_ratings"].values():
    cards.setdefault(front(c["name"]), c)

def tags(n):
    t = set(roles.get(front(n), ()))
    if not {"Creature", "Land"} & set(cards.get(front(n), {}).get("types") or []): t.add("spells")
    return t

raw = [d for d in json.loads(gzip.open(R / "analysis/out/decks_FRA.json.gz").read()) if d["format"] == "PremierDraft"]
per_draft = Counter(d["draft"] for d in raw)
out = {}
for lane, (name, plan, e, p) in PLANS.items():
    decks = [d for d in raw if per_draft[d["draft"]] == 1 and d["lane"] == lane and d["trophy"] and sum(d["counts"].values()) == 40]
    count = lambda d, t: sum(k for n, k in d["counts"].items() if "Land" not in (cards.get(front(n), {}).get("types") or []) and t in tags(n))
    out[lane] = {"name": name, "plan": plan, "enabler": e, "payoff": p, "wants": dict(zip(("enabler", "payoff"), WANTS[lane])), "decks": len(decks),
                 "median": {"enabler": median(count(d, e) for d in decks), "payoff": median(count(d, p) for d in decks),
                            "top_payoff": median(count(d, "top_payoff") for d in decks)}}
    print(lane, out[lane]["decks"], out[lane]["median"])
config.write_fitted("FRA", "strategy_plan", out, "analysis/strategy_plan.py: median enablers / payoffs / top payoffs in one-deck Premier "
                    "trophy decks per pair, with the pair's plan from docs/archetypes-FRA.md")
