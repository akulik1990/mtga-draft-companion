"""Advisor v2: one probabilistic score.

    score(card) = Σ_lane P(lane | pool + card, signals, format) × V(card | lane)

V(card | lane) is the lane's GIH WR for the card, shrunk toward All Decks by
sample size (w = n / (n + K)); when the card is not castable in the lane it is
the format's replacement level (mean − 2 SD). The candidate joins the pool when
P(lane) is computed, with extra weight (STEER): early the drafter follows their
picks, so a card's own lanes count; late the pool dominates and off-colour cards
fall to replacement value. Wheel chance is a tiebreaker only.

Constants are tuned on tools/replay/run.py compare; see CLAUDE.md, Validation.
"""

import statistics
from functools import lru_cache

from src import constants
from src.advisor.engine import DraftAdvisor
from src.advisor.mana_base import count_fixing
from src.advisor.schema import Recommendation

from companion.advisor_v2 import composition, synergy
from companion.advisor_v2.lanes import ALL, LANES, all_decks, castable, commitment, lane_distribution, off_pips, quality
from companion.advisor_v2.signals import colour_signals


@lru_cache(maxsize=8)
def tables(set_code):
    """Per-set typical-deck and synergy tables, if tools/synergy/build.py has produced them."""
    return composition.load(set_code, constants.TEMP_FOLDER), synergy.load(set_code, constants.TEMP_FOLDER)


class DraftAdvisorV2:
    K = 300.0  # shrinkage: w = n / (n + K)
    ALPHA = 1.0  # pool commitment per quality point
    STEER = 4.0  # the candidate's own quality counts this many times
    COMMIT_BY = 24  # spells in the pool by which STEER has decayed to 0 and ALPHA reached ALPHA_LATE; 0 = no decay
    ALPHA_LATE = 1.5  # pool commitment weight once the pool has COMMIT_BY spells
    BETA = 0.5  # per-colour signal z
    GAMMA = 0.5  # lane prior, in SDs of the lane's card mean
    REPLACEMENT_SD = 2.0  # V when not castable: mean − 2 SD
    N_SPARSE = 50  # fewer samples than this in a lane: the card is rarely played there
    SPARSE_SD = 0.5  # ...so shrink toward All Decks − SPARSE_SD × std instead of All Decks
    SPLASH = 0.0  # off: measured only together with pips (contested). 0.85 = one off-colour pip with a fixer for it in the pool: shrunk WR × SPLASH
    WHEEL_EPS = 0.3  # within this of the best score, prefer the card that wheels less

    def __init__(self, set_metrics, taken_cards, seen_packs=None):
        self.metrics = set_metrics
        self.pool = [c for c in (taken_cards or []) if c]
        self.mean, self.std = set_metrics.get_metrics(ALL, "gihwr")
        if self.mean <= 0:
            self.mean = 54.0
        if self.std <= 0:
            self.std = 4.0
        _, alsa_spread = set_metrics.get_metrics(ALL, "alsa")
        self.signals = colour_signals(seen_packs or [], self.mean, self.std, alsa_spread if alsa_spread > 0 else 2.0)
        self.prior = {}
        for lane in LANES:
            m, _ = set_metrics.get_metrics(lane, "gihwr")
            self.prior[lane] = (m - self.mean) / self.std if m > 0 else 0.0
        self.replacement = self.mean - self.REPLACEMENT_SD * self.std
        self.fixing = count_fixing(self.pool) if self.pool else {}
        spells = sum(1 for c in self.pool if "Land" not in (c.get("types") or []))
        progress = min(1.0, spells / self.COMMIT_BY) if self.COMMIT_BY > 0 else 0.0
        self.steer = self.STEER * (1.0 - progress)  # the candidate's pull fades as the pool becomes a deck
        self.alpha = self.ALPHA + (self.ALPHA_LATE - self.ALPHA) * progress
        self.commit = commitment(self.pool, self.mean, self.std)
        self.lanes = self._lanes(self.commit)  # pool only; what the Analyst shows
        set_code = next((c.get("set") for c in self.pool if c.get("set")), None)
        self.typical, self.synergy = tables(set_code) if set_code else ({}, {})
        self.composition = composition.Composition(self.typical, self.pool)

    def _lanes(self, commit):
        return lane_distribution(commit, self.signals, self.prior, self.alpha, self.BETA, self.GAMMA)

    def shrunk_wr(self, card, lane):
        """Lane GIH WR pulled toward a prior by sample size; (value, n).

        The prior is the card's All Decks GIH WR, lowered by SPARSE_SD × std when the
        card has fewer than N_SPARSE samples in the lane, or 17Lands withheld its win
        rate there: nobody playing a card in a lane is evidence it does not belong.
        """
        base = float(all_decks(card).get("gihwr") or 0.0)
        row = (card.get("deck_colors") or {}).get(lane) or {}
        n = int(row.get("samples") or 0)
        wr = float(row.get("gihwr") or 0.0)
        # 17Lands hides the win rate (0.0) when a card has too few games in a lane, even with samples > 0.
        sparse = n < self.N_SPARSE or wr <= 0
        prior = base - self.SPARSE_SD * self.std if sparse else base
        if n <= 0 or wr <= 0:
            return prior, 0
        w = n / (n + self.K)
        return w * wr + (1 - w) * prior, n

    def splash_colour(self, card, lane):
        """The colour a one-pip splash needs, when the pool has a fixer for it; else None."""
        off = off_pips(card, lane)
        if len(off) != 1:
            return None
        return next((c for c in off[0] if self.fixing.get(c, 0) > 0), None)

    def value(self, card, lane):
        if not castable(card, lane):
            if self.splash_colour(card, lane):
                return max(self.replacement, self.shrunk_wr(card, lane)[0] * self.SPLASH)
            return self.replacement
        v = self.shrunk_wr(card, lane)[0] * self.composition.multiplier(card, lane)
        if synergy.ENABLED:
            v += synergy.bonus(self.synergy, card, self.pool, lane)
        return v

    def evaluate_pack(self, pack_cards, current_pick, current_pack=1):
        if not pack_cards:
            return []
        wrs = [float(all_decks(c).get("gihwr") or 0.0) for c in pack_cards]
        wrs = [w for w in wrs if w > 0]
        pack_mean = statistics.mean(wrs) if wrs else self.mean
        pack_std = statistics.pstdev(wrs) if len(wrs) > 1 else self.std
        pack_std = pack_std if pack_std > 0 else self.std
        by_wr = sorted(pack_cards, key=lambda c: float(all_decks(c).get("gihwr") or 0.0), reverse=True)
        rank_in_pack = {c.get("name"): i for i, c in enumerate(by_wr)}
        upstream = DraftAdvisor(self.metrics, self.pool)  # only for its wheel polynomial
        if not self.typical:
            set_code = next((c.get("set") for c in pack_cards if c.get("set")), None)
            if set_code:
                self.typical, self.synergy = tables(set_code)
                self.composition = composition.Composition(self.typical, self.pool)

        recs = []
        for card in pack_cards:
            name = str(card.get("name", "")).strip()
            base = float(all_decks(card).get("gihwr") or 0.0)
            if base <= 0 or "Land" in (card.get("types") or []) and not card.get("colors"):
                recs.append(self._rec(card, name, base, 0.0, (base - pack_mean) / pack_std, 0.0, 0.0,
                                      ["No data" if base <= 0 else "Basic land"]))
                continue
            q = quality(card, self.mean, self.std)
            commit = dict(self.commit)
            for lane in LANES:
                if castable(card, lane):
                    commit[lane] += self.steer * q
            lanes = self._lanes(commit)
            score = sum(p * self.value(card, lane) for lane, p in lanes.items())
            cast_p = sum(p for lane, p in lanes.items() if castable(card, lane))
            wheel = upstream._wheel_chance(card, max(1, current_pick), rank_in_pack.get(name, 99), pack_size=len(pack_cards))
            recs.append(self._rec(card, name, base, score, (base - pack_mean) / pack_std, cast_p, wheel,
                                  self._reasons(card, lanes, cast_p)))

        recs.sort(key=lambda r: r.contextual_score, reverse=True)
        # Wheel as a tiebreaker: among cards within WHEEL_EPS of the best, take the one least likely to come back.
        if len(recs) > 1:
            top = recs[0].contextual_score
            close = [r for r in recs if top - r.contextual_score <= self.WHEEL_EPS]
            if len(close) > 1:
                pick = min(close, key=lambda r: r.wheel_chance)
                recs.remove(pick)
                recs.insert(0, pick)
        return recs

    def _rec(self, card, name, base, score, z, cast_p, wheel, reasons):
        best = max(self.lanes, key=self.lanes.get)
        return Recommendation(
            card_name=name,
            base_win_rate=round(base, 2),
            contextual_score=round(score, 2),
            z_score=round(z, 2),
            cast_probability=round(cast_p, 3),
            wheel_chance=round(wheel, 1),
            functional_cmc=float(card.get("cmc") or 0),
            reasoning=reasons,
            is_elite=z >= 1.5 and cast_p > 0.4,
            archetype_fit=best if castable(card, best) else "Off-colour",
            tags=card.get("tags") or [],
        )

    def _reasons(self, card, lanes, cast_p):
        out = []
        for lane, p in sorted(lanes.items(), key=lambda kv: kv[1], reverse=True)[:2]:
            if p < 0.1:
                break
            wr, n = self.shrunk_wr(card, lane)
            if castable(card, lane):
                out.append(f"{lane} {p:.0%} → {wr:.1f}" + (f" (n={n})" if n >= self.N_SPARSE else " (rarely played there)"))
            elif self.splash_colour(card, lane):
                c = self.splash_colour(card, lane)
                out.append(f"{lane} {p:.0%} → splash {c} (fixing {self.fixing.get(c, 0)}) {self.value(card, lane):.1f}")
            else:
                out.append(f"{lane} {p:.0%} → off-colour")
        best = max(lanes, key=lanes.get)
        m = self.composition.multiplier(card, best)
        if abs(m - 1.0) >= 0.03:
            out.append(f"curve/interaction {'+' if m > 1 else ''}{100 * (m - 1):.0f}% for {best}")
        if cast_p < 0.6:
            out.append(f"castable in {cast_p:.0%} of likely lanes")
        strong = [c for c, z in self.signals.items() if z >= 1.0 and c in (card.get("colors") or [])]
        if strong:
            out.append("open: " + "".join(strong))
        return out
