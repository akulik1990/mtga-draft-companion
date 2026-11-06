"""Shared setup for the builder-gap analysis (docs/builder-gap-FRA.md): his pool, the deck models, a term-by-term decomposition.
Read-only: no file under companion/ is written; builder constants are changed in memory only."""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
import companion  # noqa: E402,F401
from companion import DATA_DIR, session  # noqa: E402
from companion.advisor_v2 import config, deckbuilder  # noqa: E402
import companion.advisor_v2.engine  # noqa: E402,F401
from tools.deck_model import SHAPE, DeckModel, context_row  # noqa: E402
from tools import video_draft  # noqa: E402

VID = "mqZsj4Vg2Es"
OUT = REPO / "analysis" / "out"


def setup():
    config.activate("FRA")


def pool_cards():
    rec = json.loads((REPO / "analysis" / "videos" / VID / "picks.json").read_text(encoding="utf-8"))
    metrics, cards = video_draft.load_set(rec["dataset"])
    return cards([p["taken"] for p in rec["picks"]])


def his_deck(pool):
    d = json.loads((REPO / "analysis" / "videos" / VID / "deck.json").read_text(encoding="utf-8"))
    names = {**d["main"], **d["lands"]}
    by = {c["name"]: c for c in pool}
    deck = []
    for n, k in names.items():
        c = by.get(n) or {"name": n, "types": ["Basic", "Land"], "colors": [], "mana_cost": ""}
        deck.append({**session._trim(c), "count": k})
    return {"deck": deck, **session.mana_identity(deck)}


def counts_of(b):
    return {c["name"]: c.get("count", 1) for c in b["deck"]}


def decompose(model, counts, main, splash):
    """Logit and win-rate pieces of DeckModel.win_rate: intercept+context, each shape term (w * z), the card sum, per card."""
    from tools.deck_model import shape
    s = shape(counts, model.facts, main, splash)
    z = (np.array([s[k] for k in SHAPE]) - model.m["shape_mu"]) / model.m["shape_sd"]
    z2 = np.concatenate([z, [z[SHAPE.index("creatures")] ** 2, z[SHAPE.index("lands")] ** 2, z[SHAPE.index("cmc2")] ** 2]])
    terms = {n: float(w * v) for n, w, v in zip(model.m["shape_names"], model.m["shape"], z2)}
    per_card = {n: model.m["cards"].get(n, model.m["cards"].get(n.split(" // ")[0], 0.0)) * c for n, c in counts.items()}
    base = model.m["intercept"] + model.ctx
    logit = base + sum(terms.values()) + sum(per_card.values())
    return {"base": base, "terms": terms, "raw": s, "per_card": per_card, "cards": sum(per_card.values()), "logit": logit,
            "wr": 100 / (1 + np.exp(-logit))}
