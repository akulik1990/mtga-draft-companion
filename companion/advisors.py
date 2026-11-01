"""Runs every registered advisor on the same pack. Neither sees the other's output."""

import logging

from src.advisor.engine import DraftAdvisor

from companion.advisor_v2.engine import DraftAdvisorV2

log = logging.getLogger(__name__)

ADVISORS = {"upstream": DraftAdvisor, "v2": DraftAdvisorV2}


def rank_pack(metrics, taken_cards, pack_cards, pick, pack_number, seen_packs=()):
    """({advisor_key: [recommendation dict with rank, ...]}, v2's P(lane) over the pool, v2's colour signal in use, v2's per-neighbour signals)."""
    results, lanes, signals, sides = {}, {}, {}, {}
    for key, cls in ADVISORS.items():
        try:
            advisor = cls(metrics, taken_cards, seen_packs=list(seen_packs)) if cls is DraftAdvisorV2 else cls(metrics, taken_cards)
            recs = advisor.evaluate_pack(pack_cards, pick, current_pack=pack_number)
            results[key] = [{"rank": i, **r.model_dump()} for i, r in enumerate(recs, 1)]
            if cls is DraftAdvisorV2:
                lanes, signals = advisor.lanes, advisor.signals
                sides = {"active": advisor.signal_side, **advisor.signal_sides}
        except Exception:
            log.exception("Advisor %s failed", key)
            results[key] = []
    return results, lanes, signals, sides
