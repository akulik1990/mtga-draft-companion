"""Runs every registered advisor on the same pack. Neither sees the other's output."""

import logging

from src.advisor.engine import DraftAdvisor

from companion.advisor_v2.engine import DraftAdvisorV2

log = logging.getLogger(__name__)

ADVISORS = {"upstream": DraftAdvisor, "v2": DraftAdvisorV2}


def rank_pack(metrics, taken_cards, pack_cards, pick, pack_number, signals):
    """Return {advisor_key: [recommendation dict with rank, ...]} for the current pack."""
    results = {}
    for key, cls in ADVISORS.items():
        try:
            advisor = cls(metrics, taken_cards, signals=signals)
            recs = advisor.evaluate_pack(pack_cards, pick, current_pack=pack_number)
            results[key] = [{"rank": i, **r.model_dump()} for i, r in enumerate(recs, 1)]
        except Exception:
            log.exception("Advisor %s failed", key)
            results[key] = []
    return results
