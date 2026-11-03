from companion import deckscore


class FlatModel:
    """Every card worth its GIH WR's distance from 55, so a better card raises the deck."""
    def win_rate(self, counts, main="", splash=""):
        return 50.0 + sum(k for n, k in counts.items() if n == "Good")


def card(name, cost="{1}{U}", types=("Creature",), colours=("U",), wr=55.0):
    return {"name": name, "mana_cost": cost, "types": list(types), "colors": list(colours), "cmc": 2,
            "deck_colors": {"All Decks": {"gihwr": wr}, "UB": {"gihwr": wr}}}


def test_score_caps_copies_counts_basics_and_ranks_swaps():
    pool = [card("Good"), card("Good"), card("Meh", wr=50.0), card("Dual", cost="", types=("Land",), colours=("U", "B"))]
    out = deckscore.score("XXX", pool, {"Good": 5, "Meh": 1, "Island": 7, "Dual": 1, "Nope": 3}, FlatModel())
    assert out["total"] == 2 + 1 + 7 + 1 and out["spells"] == 3 and out["lands"] == 8
    assert {m["colour"]: m["sources"] for m in out["mana"]}["U"] == 8
    rows = {r["name"]: r for r in out["cards"]}
    assert rows["Good"]["in_deck"] == 2 and rows["Good"]["held"] == 2
    assert out["deck_wr"] == 52.0
    side = deckscore.score("XXX", pool, {"Good": 1, "Meh": 1}, FlatModel())
    meh = {r["name"]: r for r in side["cards"]}["Meh"]
    assert meh["value"] == -1.0 and meh["swap_for"] == "Good"  # the second Good in its slot would gain a point
