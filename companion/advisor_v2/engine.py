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
from companion.advisor_v2 import signals as signal_mod
from companion.advisor_v2.signals import colour_signals, side_for_pack, side_signals


@lru_cache(maxsize=8)
def top_ratings(set_code):
    """({card name: card} from the set's top-player dataset, shift), or ({}, 0) when there is none.

    Top players win more, so their rates run higher; shift is the mean gap on All Decks GIH WR,
    subtracted so top-player rates sit on the same scale as everything else in the engine.
    """
    import json
    from pathlib import Path

    top_path = Path(constants.SETS_FOLDER) / f"{set_code}_PremierDraft_Top_Data.json"
    all_path = Path(constants.SETS_FOLDER) / f"{set_code}_PremierDraft_All_Data.json"
    if not top_path.exists() or not all_path.exists():
        return {}, 0.0
    top = json.loads(top_path.read_text()).get("card_ratings") or {}
    every = json.loads(all_path.read_text()).get("card_ratings") or {}
    gaps, by_name = [], {}
    for key, card in top.items():
        by_name.setdefault(card.get("name"), card)
        t = float(all_decks(card).get("gihwr") or 0.0)
        a = float(all_decks(every.get(key) or {}).get("gihwr") or 0.0)
        if t > 0 and a > 0:
            gaps.append(t - a)
    return (by_name, statistics.mean(gaps)) if gaps else ({}, 0.0)


@lru_cache(maxsize=8)
def pick_order(set_code):
    """{card name: earliness z} from top players' average pick position (ATA); all players' when no top file.

    Positive means top players take the card earlier than the average card.
    """
    import json
    from pathlib import Path

    for group in ("Top", "All"):
        path = Path(constants.SETS_FOLDER) / f"{set_code}_PremierDraft_{group}_Data.json"
        if not path.exists():
            continue
        atas = {}
        for card in (json.loads(path.read_text()).get("card_ratings") or {}).values():
            ata = float(all_decks(card).get("ata") or 0.0)
            if ata > 0 and int(all_decks(card).get("pick_count") or 0) >= 50:
                atas.setdefault(card.get("name"), ata)
        if len(atas) > 20:
            mean, sd = statistics.mean(atas.values()), statistics.pstdev(atas.values()) or 1.0
            return {name: (mean - ata) / sd for name, ata in atas.items()}
    return {}


STAGES = [(0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 9, "6-9"), (10, 14, "10-14"), (15, 22, "15-22"), (23, 99, "23+")]


def stage_of(spells):
    """The pool-size bucket the branch thresholds are fitted per."""
    return next(label for lo, hi, label in STAGES if lo <= spells <= hi)


@lru_cache(maxsize=8)
def branch_thresholds(set_code):
    """{stage: gap} fitted by tools/fit_branch.py; {} when the set has none."""
    import json
    from pathlib import Path

    path = Path(__file__).with_name(f"branch_{set_code}.json")
    return json.loads(path.read_text()).get("stages", {}) if path.exists() else {}


@lru_cache(maxsize=8)
def branch_file(path):
    import json
    from pathlib import Path

    return json.loads(Path(path).read_text()).get("stages", {})


@lru_cache(maxsize=8)
def learnt_prefs(set_code, which):
    """{card name: strength z} learnt from top drafters' choices, or {} when the set has none."""
    import json
    from pathlib import Path

    path = Path(__file__).with_name(f"prefs_{set_code}.json")
    return json.loads(path.read_text()).get(which, {}) if path.exists() else {}


@lru_cache(maxsize=8)
def prefs_file(path, which):
    import json
    from pathlib import Path

    return json.loads(Path(path).read_text()).get(which, {})


@lru_cache(maxsize=8)
def tables(set_code):
    """Per-set typical-deck and synergy tables, if tools/synergy/build.py has produced them."""
    return composition.load(set_code, constants.TEMP_FOLDER), synergy.load(set_code, constants.TEMP_FOLDER)


class DraftAdvisorV2:
    K = 300.0  # shrinkage: w = n / (n + K)
    ALPHA = 1.0  # pool commitment per quality point
    STEER = 4.0  # the candidate's own quality counts this many times
    COMMIT_BY = 18  # spells in the pool by which STEER has decayed to 0 and ALPHA reached ALPHA_LATE; 0 = no decay
    ALPHA_LATE = 1.5  # pool commitment weight once the pool has COMMIT_BY spells
    BETA = 0.5  # per-colour signal z
    GAMMA = 0.5  # lane prior, in SDs of the lane's card mean
    REPLACEMENT_SD = 2.0  # V when not castable: mean − 2 SD
    N_SPARSE = 50  # fewer samples than this in a lane: the card is rarely played there
    SPARSE_SD = 0.5  # ...so shrink toward All Decks − SPARSE_SD × std instead of All Decks
    SPLASH = 0.85  # one off-colour pip with a fixer for it in the pool: shrunk WR × SPLASH (0 = off)
    SPEC_SPLASH = 1.0  # early-draft splash before any fixing exists: shrunk WR × SPEC_SPLASH × (1 − commitment progress); 0 = off
    SPEC_ALL = False  # False: only cards castable in no two-colour lane (three-colour cards) get it; True: any card one pip off
    SPLIT = True  # two orders: Power (best card if you end up in its colours) and Fit (best for the colours you are in); branch to Power above a learnt gap
    BRANCH_MODE = "learnt"  # "learnt" (fitted rule), "fit" (never branch), "power" (always take the strongest card)
    BRANCH_COLOUR = True  # switch rule that also weighs whether the strong card keeps a colour of the top pair and how settled the pool is (branch_<SET>_colour.json). FRA: test top-1 55.4 -> 56.0, deck win rate -0.86 (SE 0.34, 200 bot tables, all from tables ending in other colours); on by the user's call 2026-11-01
    BRANCH_FILE = ""  # override the fitted rule's file (measuring alternatives); "" = companion/advisor_v2/branch_<SET>.json
    BRANCH_DEFAULT = 3.0  # GIH WR points; used for stages with no fitted threshold (companion/advisor_v2/branch_<SET>.json)
    ORDER_FADE_BY = 0  # pick in the pack by which the pick-order prior has faded to 0; 0 = no fade
    PREF_WEIGHT = 4.0  # WR points per SD of top drafters' learnt preference (tools/learn_prefs.py); 0 = off
    PREF_ON_WEIGHT = 4.0  # what the app's switch sets PREF_WEIGHT to; the switch also swaps the pick-order prior out
    ORDER_OFF_WEIGHT = 2.0  # pick-order prior the app's switch restores when learnt preferences are switched off
    PREF_SCOPE = "both"  # where learnt preferences apply: "both" orders, or "power" only (Fit stays about your colours)
    PREF_FILE = ""  # override the learnt-preference file (held-out evaluations); "" = companion/advisor_v2/prefs_<SET>.json
    PREF_SET = "early"  # which learnt strengths: "early" (pack 1 picks 1-5) or "all" (every pick)
    ORDER_WEIGHT = 0.0  # replaced by learnt preferences (FRA test top-1 55.4 and best deck win rate at weight 4); 2.0 when they are off. WR points per SD of how early top players take the card (their ATA), × castability; 0 = off
    TOP_WR = False  # rejected 2026-11-01 (FRA +0.1, TDM −0.7 top-drafter top-1). Use top players' win rates (the set's _Top dataset), shrunk toward all players' by sample size
    K_TOP = 300.0  # shrinkage for top-player rates: w = n_top / (n_top + K_TOP)
    LANDS = True  # round 2: lands exempt from the sparse prior; partial-overlap lands valued by the splashes they enable
    FIX_WEIGHT = 0.5  # share of each enabled splash's gain (over replacement) credited to the land
    FIX_CAP = 3.0  # most a land's fixing can add, in WR points
    HALF_PIVOT = 0  # from this many spells in the pool, lanes sharing no colour with the pool's top pair get P = 0 (winners change one colour, not both); 0 = off
    WHEEL_EPS = 0.3  # within this of the best score, prefer the card that wheels less
    WHEEL_SPLIT = False  # apply the WHEEL_EPS tiebreak to the split's final order too (it used to apply only with SPLIT off)
    WHEEL_PENALTY = 0.0  # WR points taken from Power and Fit per 100% chance the card comes back; 0 = off

    def __init__(self, set_metrics, taken_cards, seen_packs=None):
        self.metrics = set_metrics
        self.pool = [c for c in (taken_cards or []) if c]
        self.mean, self.std = set_metrics.get_metrics(ALL, "gihwr")
        if self.mean <= 0:
            self.mean = 54.0
        if self.std <= 0:
            self.std = 4.0
        _, alsa_spread = set_metrics.get_metrics(ALL, "alsa")
        spread = alsa_spread if alsa_spread > 0 else 2.0
        seen = seen_packs or []
        self.current_pack = max((p[0] for p in seen), default=1)
        if signal_mod.DIRECTIONAL:
            # Read only the neighbour passing to us now: packs 1 and 3 come from one side, pack 2 from the other.
            self.signal_sides = side_signals(seen, self.mean, self.std, spread)
            self.signal_side = side_for_pack(self.current_pack)
            self.signals = self.signal_sides[self.signal_side]
        else:
            self.signal_sides, self.signal_side = {}, ""
            self.signals = colour_signals(seen, self.mean, self.std, spread)
        self.prior = {}
        for lane in LANES:
            m, _ = set_metrics.get_metrics(lane, "gihwr")
            self.prior[lane] = (m - self.mean) / self.std if m > 0 else 0.0
        self.replacement = self.mean - self.REPLACEMENT_SD * self.std
        self.fixing = count_fixing(self.pool) if self.pool else {}
        spells = sum(1 for c in self.pool if "Land" not in (c.get("types") or []))
        progress = min(1.0, spells / self.COMMIT_BY) if self.COMMIT_BY > 0 else 0.0
        self.progress = progress
        self.steer = self.STEER * (1.0 - progress)  # the candidate's pull fades as the pool becomes a deck
        self.alpha = self.ALPHA + (self.ALPHA_LATE - self.ALPHA) * progress
        self.commit = commitment(self.pool, self.mean, self.std)
        self.anchor = None
        free = self._lanes(self.commit)
        self.top_lane = max(free, key=free.get)
        if self.HALF_PIVOT and spells >= self.HALF_PIVOT:
            self.anchor = self.top_lane
        self.lanes = self._lanes(self.commit)  # pool only; what the Analyst shows and what Fit uses
        self.split = None
        self.order, self.prefs = {}, {}  # filled per pack in evaluate_pack
        set_code = next((c.get("set") for c in self.pool if c.get("set")), None)
        self.typical, self.synergy = tables(set_code) if set_code else ({}, {})
        self.top, self.top_shift = top_ratings(set_code) if set_code and self.TOP_WR else ({}, 0.0)
        self.composition = composition.Composition(self.typical, self.pool)

    def _lanes(self, commit):
        p = lane_distribution(commit, self.signals, self.prior, self.alpha, self.BETA, self.GAMMA)
        if self.anchor:
            # Winners keep at least one colour once they have a few cards: GU can become GR, not WR.
            p = {lane: v if set(lane) & set(self.anchor) else 0.0 for lane, v in p.items()}
            total = sum(p.values())
            p = {lane: v / total for lane, v in p.items()}
        return p

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
        if self.TOP_WR and self.top:
            base, wr = self._toward_top(card, ALL, base), self._toward_top(card, lane, wr)
        # 17Lands hides the win rate (0.0) when a card has too few games in a lane, even with samples > 0.
        # Lands are exempt: their per-lane rows are thin by nature, not because nobody plays them.
        is_land = self.LANDS and "Land" in (card.get("types") or [])
        sparse = not is_land and (n < self.N_SPARSE or wr <= 0)
        prior = base - self.SPARSE_SD * self.std if sparse else base
        if n <= 0 or wr <= 0:
            return prior, 0
        w = n / (n + self.K)
        return w * wr + (1 - w) * prior, n

    def _toward_top(self, card, lane, value):
        """value moved toward the top-player rate for this card and lane, by the top sample size."""
        top_card = self.top.get(card.get("name"))
        row = ((top_card or {}).get("deck_colors") or {}).get(lane) or {}
        t, n = float(row.get("gihwr") or 0.0), int(row.get("samples") or 0)
        if t <= 0 or n <= 0 or value <= 0:
            return value
        w = n / (n + self.K_TOP)
        return w * (t - self.top_shift) + (1 - w) * value

    def speculative(self, card, lane):
        """Early draft: is this lane a live splash for the card even with no fixing yet?

        Narrow by default: only cards no two-colour lane can cast (three-colour cards), so a
        mono-colour card keeps pulling v2 into its own colours early (round 1's main gain).
        """
        if self.SPEC_SPLASH <= 0 or self.progress >= 1 or len(off_pips(card, lane)) != 1:
            return False
        return self.SPEC_ALL or not any(castable(card, other) for other in LANES)

    def splash_colour(self, card, lane):
        """The colour a one-pip splash needs, when splashing is on and the pool has a fixer for it; else None."""
        if self.SPLASH <= 0:
            return None
        off = off_pips(card, lane)
        if len(off) != 1:
            return None
        return next((c for c in off[0] if self.fixing.get(c, 0) > 0), None)

    def fixing_gain(self, land, lane):
        """WR points a land adds in a lane by enabling one-pip splashes of pool spells (round 2, change 1).

        For each colour the land makes that is outside the lane and has no source in the pool yet,
        every pool spell needing exactly that one off-colour pip goes from replacement to a splash.
        """
        if self.SPLASH <= 0:
            return 0.0
        new = [c for c in land.get("colors") or [] if c not in lane and not self.fixing.get(c, 0)]
        gain = 0.0
        for spell in self.pool:
            if "Land" in (spell.get("types") or []):
                continue
            off = off_pips(spell, lane)
            if len(off) == 1 and any(c in off[0] for c in new):
                gain += max(0.0, self.shrunk_wr(spell, lane)[0] * self.SPLASH - self.replacement)
        return min(self.FIX_CAP, self.FIX_WEIGHT * gain)

    def value(self, card, lane):
        if self.LANDS and "Land" in (card.get("types") or []):
            colours = card.get("colors") or []
            if not colours or all(c in lane for c in colours):
                return self.shrunk_wr(card, lane)[0]  # colourless or on-lane: its own win rate
            if any(c in lane for c in colours):
                return self.replacement + self.fixing_gain(card, lane)  # taps for the lane and fixes a splash
            return self.replacement
        if not castable(card, lane):
            if self.splash_colour(card, lane):
                return max(self.replacement, self.shrunk_wr(card, lane)[0] * self.SPLASH)
            if self.speculative(card, lane):
                # Early on, a card one pip off a lane is still a live splash: top drafters take the bomb and find the fixing later.
                return max(self.replacement, self.shrunk_wr(card, lane)[0] * self.SPEC_SPLASH * (1 - self.progress))
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
        set_code = next((c.get("set") for c in (*self.pool, *pack_cards) if c.get("set")), None)
        self.order = pick_order(set_code) if self.ORDER_WEIGHT and set_code else {}
        self.prefs = (prefs_file(self.PREF_FILE, self.PREF_SET) if self.PREF_FILE else learnt_prefs(set_code, self.PREF_SET)) if self.PREF_WEIGHT and set_code else {}
        if not self.typical:
            set_code = next((c.get("set") for c in pack_cards if c.get("set")), None)
            if set_code:
                self.typical, self.synergy = tables(set_code)
                if self.TOP_WR and not self.top:
                    self.top, self.top_shift = top_ratings(set_code)
                self.composition = composition.Composition(self.typical, self.pool)

        recs = []
        for card in pack_cards:
            name = str(card.get("name", "")).strip()
            base = float(all_decks(card).get("gihwr") or 0.0)
            types = card.get("types") or []
            basic = "Basic" in types or name in constants.BASIC_LANDS or (not self.LANDS and "Land" in types and not card.get("colors"))
            if base <= 0 or basic:
                recs.append(self._rec(card, name, base, 0.0, (base - pack_mean) / pack_std, 0.0, 0.0,
                                      ["No data" if base <= 0 else "Basic land"]))
                continue
            q = quality(card, self.mean, self.std)
            commit = dict(self.commit)
            for lane in LANES:
                # The candidate pulls toward lanes where it is castable, and early on also where it is one pip off (a live splash).
                if castable(card, lane) or self.speculative(card, lane):
                    commit[lane] += self.steer * q
            lanes = self._lanes(commit)
            score = sum(p * self.value(card, lane) for lane, p in lanes.items())
            cast_p = sum(p for lane, p in lanes.items() if castable(card, lane))
            if self.ORDER_WEIGHT or self.PREF_WEIGHT:
                # What top players reach for, beyond its win rate; only as far as v2 expects to play the card.
                live = sum(p for lane, p in lanes.items() if castable(card, lane) or self.speculative(card, lane))
                score += self.order_prior(name, current_pick) * live
            wheel = upstream._wheel_chance(card, max(1, current_pick), rank_in_pack.get(name, 99), pack_size=len(pack_cards))
            recs.append(self._rec(card, name, base, score, (base - pack_mean) / pack_std, cast_p, wheel,
                                  self._reasons(card, lanes, cast_p)))

        if self.SPLIT:
            return self._split(pack_cards, recs, current_pick, set_code)
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

    def order_prior(self, name, pick, prefs=True):
        """Prior in WR points from what top players reach for: their average pick position (ORDER_WEIGHT)
        and, when switched on, the preference learnt from their choices (PREF_WEIGHT)."""
        fade = max(0.0, 1.0 - (pick - 1) / self.ORDER_FADE_BY) if self.ORDER_FADE_BY else 1.0
        out = self.ORDER_WEIGHT * self.order.get(name, 0.0) if self.ORDER_WEIGHT else 0.0
        if self.PREF_WEIGHT and prefs:
            out += self.PREF_WEIGHT * self.prefs.get(name, self.prefs.get(name.split(" // ")[0], 0.0))
        return out * fade

    def power(self, card, pick):
        """How good the card is if you end up in its colours: its best lane, ignoring the pool."""
        live = [lane for lane in LANES if castable(card, lane) or self.speculative(card, lane)]
        best = max((self.value(card, lane) for lane in live), default=self.replacement)
        return best + self.order_prior(card.get("name", ""), pick)

    def fit(self, card, pick):
        """How good the card is for the colours the pool points to: the candidate does not pull the lanes."""
        score = sum(p * self.value(card, lane) for lane, p in self.lanes.items())
        live = sum(p for lane, p in self.lanes.items() if castable(card, lane) or self.speculative(card, lane))
        return score + self.order_prior(card.get("name", ""), pick, prefs=self.PREF_SCOPE == "both") * live

    def branch_features(self, card):
        """(1 if the card keeps a colour of the pool's top pair or is colourless, else 0; that pair's probability)."""
        colours = set() if "Land" in (card.get("types") or []) else set(card.get("colors") or [])
        top = self.top_lane
        return float(not colours or bool(colours & set(top))), self.lanes.get(top, 0.0)

    def _split(self, pack_cards, recs, pick, set_code):
        """Final order from the two orders: Fit, unless the strongest card beats the best fit by more than the learnt gap."""
        by_name = {c.get("name"): c for c in pack_cards}
        playable = [r for r in recs if r.contextual_score > 0]
        if not playable:
            return recs
        wheel = {r.card_name: (r.wheel_chance or 0.0) / 100.0 for r in playable}
        power = {r.card_name: self.power(by_name[r.card_name], pick) - self.WHEEL_PENALTY * wheel[r.card_name] for r in playable}
        fit = {r.card_name: self.fit(by_name[r.card_name], pick) - self.WHEEL_PENALTY * wheel[r.card_name] for r in playable}
        p_top = max(power, key=power.get)
        c_top = max(fit, key=fit.get)
        spells = sum(1 for c in self.pool if "Land" not in (c.get("types") or []))
        stage = stage_of(spells)
        gap = power[p_top] - power[c_top]
        table = branch_file(self.BRANCH_FILE) if self.BRANCH_FILE else (branch_thresholds(set_code) if set_code else {})
        if self.BRANCH_COLOUR and not self.BRANCH_FILE and set_code:
            table = branch_thresholds(f"{set_code}_colour") or table
        coef = table.get(stage) or {}
        c_strength = power[c_top] - self.mean
        shared, committed = self.branch_features(by_name[p_top])
        if coef.get("b", 0) > 0:
            # Learnt from top drafters: branch more readily early and when the best in-colour card is weak;
            # fitted files may also weigh whether the card keeps a colour and how settled the pool is.
            threshold = -(coef["a"] + coef["c"] * c_strength + coef.get("d", 0.0) * shared + coef.get("e", 0.0) * committed) / coef["b"]
        else:
            threshold = self.BRANCH_DEFAULT
        take_power = p_top != c_top and (self.BRANCH_MODE == "power" or (self.BRANCH_MODE == "learnt" and gap > threshold))
        self.split = {
            "power": sorted(({"card": n, "score": round(v, 2)} for n, v in power.items()), key=lambda x: -x["score"])[:3],
            "fit": sorted(({"card": n, "score": round(v, 2)} for n, v in fit.items()), key=lambda x: -x["score"])[:3],
            "branch": None if p_top == c_top else {"card": p_top, "over": c_top, "gap": round(gap, 2), "threshold": round(threshold, 2),
                                                    "stage": stage, "in_colour_strength": round(c_strength, 2), "shares_colour": bool(shared),
                                                    "committed": round(committed, 2), "take": take_power},
        }
        for r in playable:
            r.contextual_score = round(fit[r.card_name], 2)
            pr = sorted(power, key=power.get, reverse=True).index(r.card_name) + 1
            fr = sorted(fit, key=fit.get, reverse=True).index(r.card_name) + 1
            r.reasoning = [f"Fit #{fr} · Power #{pr}"] + r.reasoning
        order = sorted(playable, key=lambda r: fit[r.card_name], reverse=True)
        if self.WHEEL_SPLIT and not take_power and len(order) > 1:
            # Of cards about as good as the top, take the one least likely to come back.
            close = [r for r in order if fit[order[0].card_name] - fit[r.card_name] <= self.WHEEL_EPS]
            first = min(close, key=lambda r: r.wheel_chance)
            if first is not order[0]:
                order.remove(first)
                order.insert(0, first)
                first.contextual_score = round(fit[order[1].card_name] + 0.01, 2)
                first.reasoning.insert(0, f"Wheel: {first.wheel_chance:.0f}% vs {order[1].wheel_chance:.0f}% for {order[1].card_name}, within {self.WHEEL_EPS}")
        if take_power:
            top = next(r for r in order if r.card_name == p_top)
            order.remove(top)
            order.insert(0, top)
            top.contextual_score = round(fit[c_top] + 0.1, 2)  # consumers rank by score, so the branch shows there too
            top.reasoning.insert(0, f"Branch: +{gap:.1f} over {c_top}, above the {threshold:.1f} top drafters switch at")
        return order + [r for r in recs if r.contextual_score <= 0]

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
            elif self.LANDS and "Land" in (card.get("types") or []) and self.fixing_gain(card, lane) > 0:
                out.append(f"{lane} {p:.0%} → fixes a splash (+{self.fixing_gain(card, lane):.1f})")
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
