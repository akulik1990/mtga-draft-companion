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

import math
import statistics
from functools import lru_cache

from src import constants
from src.advisor.engine import DraftAdvisor
from src.advisor.mana_base import count_fixing
from src.advisor.schema import Recommendation


class V2Recommendation(Recommendation):
    """Upstream's model plus the fields 4.28 dropped from it, which v2 still fills and the app shows."""

    base_win_rate: float = 0.0
    cast_probability: float = 1.0
    functional_cmc: float = 0.0
    archetype_fit: str = ""

from companion.advisor_v2 import composition, config, synergy
from companion.advisor_v2.lanes import ALL, LANES, all_decks, any_colour_fixer, castable, commitment, lane_distribution, off_pips, pips, pool_fixing, quality
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
def branch_thresholds(set_code, name="branch"):
    """{stage: gap} fitted by tools/fit_branch.py (sections branch, branch_colour of fitted.json); {} when the set has none."""
    return config.fitted(set_code, name).get("stages", {})


@lru_cache(maxsize=8)
def branch_file(path):
    import json
    from pathlib import Path

    return json.loads(Path(path).read_text()).get("stages", {})


@lru_cache(maxsize=8)
def learnt_prefs(set_code, which):
    """{card name: strength z} learnt from top drafters' choices (section prefs of fitted.json), or {} when the set has none."""
    return config.fitted(set_code, "prefs").get(which, {})


@lru_cache(maxsize=8)
def prefs_file(path, which):
    import json
    from pathlib import Path

    return json.loads(Path(path).read_text()).get(which, {})


def set_code_of(cards):
    return next((c.get("set") for c in cards if c.get("set")), None)


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
    COLOUR_FLOOR = 2.5  # each colour counts at least this many quality points in the lane commitment (1 = one average card); 0 = off. On by Akshin's call 2026-11-07; paired test (data/Temp/sim/floor1): top-1 -0.47 (p<0.001), top-3 -0.25 (p<0.001), mock deck WR -0.32 (p 0.18); kept by his decision
    FLOOR_BY = 0  # spells in the pool by which COLOUR_FLOOR has faded to 0; 0 = no fade (the card count does it)
    COLOUR_FLOOR_CARDS = 4  # a colour with fewer spells than this counts exactly COLOUR_FLOOR (Akshin: 1-3 cards are a splash at best, not a base)
    BETA = 0.5  # per-colour signal z
    GAMMA = 0.5  # lane prior, in SDs of the lane's card mean
    REPLACEMENT_SD = 2.0  # V when not castable: mean − 2 SD
    N_SPARSE = 50  # fewer samples than this in a lane: the card is rarely played there
    SPARSE_SD = 0.5  # ...so shrink toward All Decks − SPARSE_SD × std instead of All Decks
    SPLASH = 0.85  # one off-colour pip with a fixer for it in the pool: shrunk WR × SPLASH (0 = off)
    SPEC_SPLASH = 1.0  # early-draft splash before any fixing exists: shrunk WR × SPEC_SPLASH × (1 − commitment progress); 0 = off
    OFF_PLAY = False  # a card a lane cannot cast is worth replacement + (its value - replacement) x the chance top players play such a card outside their core pair, by its strength and off-colour pips (fitted.json "offcolour", tools/fit_offcolour.py); replaces SPLASH and SPEC_SPLASH for those cards
    SPLASH_MIN_GIH = 0.0  # Akshin's rule: off-colour cards get splash value (OFF_PLAY, SPLASH, SPEC_SPLASH / SPEC_ALL, and the fixing gain lands get from them) only at this GIH WR or more, in the lane, else All Decks; 0 = off. 56.5: lowest quarter of 60%+ players' splash cards (FRA, 6,012 cards; median 57.4)
    SPEC_ALL = False  # False: only cards castable in no two-colour lane (three-colour cards) get it; True: any card one pip off
    SPEC_ALL_PULL = True  # with SPEC_ALL: a one-pip-off card also pulls P(lane) toward the lanes it could splash into; False = it pulls only toward its own colours (and three-colour cards as before)
    SPLIT = True  # two orders: Power (best card if you end up in its colours) and Fit (best for the colours you are in); branch to Power above a learnt gap
    BRANCH_MODE = "learnt"  # "learnt" (fitted rule), "fit" (never branch), "power" (always take the strongest card)
    BRANCH_COLOUR = True  # switch rule that also weighs whether the strong card keeps a colour of the top pair and how settled the pool is (fitted.json, branch_colour). FRA: test top-1 55.4 -> 56.0, deck win rate -0.86 (SE 0.34, 200 bot tables, all from tables ending in other colours); on by the user's call 2026-11-01
    KEEP_ANY = False  # "keeps a colour" = some pair that casts the card shares a colour with the top pair (a mono-red card keeps U of a UB pool via UR); False = the card itself shares a colour with the top pair
    BRANCH_FILE = ""  # override the fitted rule's file (measuring alternatives); "" = the set's fitted.json
    CARD_PAIRS_PULL = 0.0  # extra pull of a setup bomb in the pool toward its home pairs (fitted card_pairs), x its quality; 0 = off
    COLOURLESS_EARLY = 0.0  # WR points off colourless cards (artifacts, colourless lands) in pack 1; 0 = off
    DUAL_OPEN_EDGE = 8.0  # a nonbasic dual that taps for one of the lane's colours is worth at least replacement + this x (1 - progress); 0 = off. Akshin's call 2026-11-07, untested
    POWER_EARLY = 1.0  # Fit blended toward Power by this weight at pick 1, fading to 0 by POWER_EARLY_BY spells; 0 = off. Akshin's call 2026-11-07, untested
    POWER_EARLY_BY = 4  # cards in the second-most held colour by which the Power blend has faded (Akshin: a base from 4 cards)
    SPLASH_POOL_K = 4.0  # a splashable off-colour card counts at least K x (its quality + the pool's cards of that colour) / (the pool's quality castable in the pair) of its value above replacement, over the fitted off-colour play rate; 0 = off. Akshin's call 2026-11-07, untested
    SPLASH_POOL_CAP = 0.85
    COLOUR_PACE = 12.0  # cards of one colour per 23 spells in a two-colour lane; a mono-coloured card of a colour ahead of its pace loses points (Akshin: a ceiling for wanting, not a goal; 11 or 12, 12 for now; trophy decks average 14.2 / 10.1, analysis/colour_split.py)
    COLOUR_LEAD = 1  # a colour may lead the next most held colour by this many cards before COLOUR_PACE_POINTS start
    COLOUR_LEAD_SCALE = 4.0  # cards beyond the lead for the full COLOUR_PACE_POINTS (Akshin, 2026-11-07)
    COLOUR_PACE_THREE = "7,6,5"  # the same in a three-colour lane, for the pool's most, second and third most held colour
    COLOUR_PACE_POINTS = 4.0  # up to this many WR points, in proportion to how far ahead (100% ahead = all); 0 = off. Akshin's call 2026-11-07, untested
    FIT_WEIGHTS_LANES = True  # per-pick fitted weights averaged over the likely pairs (p >= 0.05) instead of choice.top_lane's one pair; Akshin's call 2026-11-07, untested
    TOP_END_PICKS = 8  # the top-end discount applies only up to this pick of each pack (the videos: not *highly*); Akshin's call 2026-11-07
    THREE_OPEN = 0.5  # share of each likely pair's chance moved to its three-colour bases, only for a third colour with THREE_GOOD good cards (replaces the fitted THREE_LANES rule); Akshin's call 2026-11-07, untested
    THREE_GOOD = 3
    THREE_GOOD_Q = 1.5  # "good": quality >= this (1 = format mean GIH WR, 1.5 = half an SD above, about 58 in FRA)
    TOP_END_POINTS = 8.0  # WR points off a 6+ mana spell under the finisher line (mean + 1 SD), x the pool's top-end pace (5+ drops held / the top pair's typical count scaled to spells held, capped at 1); Akshin's call 2026-11-07, untested
    COLOURLESS_FADE_BY = 8  # COLOURLESS_EARLY fades linearly from pick 1 to 0 at this pick of pack 1; 0 = no fade (full all pack 1). On by Akshin's call 2026-11-07; paired test (data/Temp/sim/floor1): top-1 -0.47 (p<0.001), top-3 -0.25 (p<0.001), mock deck WR -0.32 (p 0.18); kept by his decision
    PLAY_RATE_POINTS = 0.0  # WR points off a non-land card in proportion to how far its play rate falls below PLAY_RATE_FROM (all of them at 0%); 0 = off
    PLAY_RATE_FROM = 50.0
    BOMB_WR = 61.0  # All Decks GIH WR from which the strong card counts as a bomb in the switch rule (its "f" term, tools/fit_branch.py --bomb)
    BRANCH_EARLY = 0.0  # WR points added to the switch bar at pack 1 picks 2..BRANCH_EARLY_TO (FRA train: early switches matched top drafters 30.9% vs 55.4% otherwise)
    BRANCH_EARLY_TO = 7
    BRANCH_DEFAULT = 3.0  # GIH WR points; used for stages with no fitted threshold (fitted.json, branch)
    ORDER_FADE_BY = 9  # pick in the pack by which the pick-order prior has faded to 0; 0 = no fade. 9 by Akshin's call 2026-11-07 (prefs are fitted on P1 picks 1-5; gone by the wheel), untested
    PRIOR_CAP = 3.0  # the pick-order prior is clamped to +/- this many WR points; 0 = no cap. Akshin's call 2026-11-07, untested
    PREF_WEIGHT = 4.0  # WR points per SD of top drafters' learnt preference (tools/learn_prefs.py); 0 = off
    PREF_ON_WEIGHT = 4.0  # what the app's switch sets PREF_WEIGHT to; the switch also swaps the pick-order prior out
    ORDER_OFF_WEIGHT = 2.0  # pick-order prior the app's switch restores when learnt preferences are switched off
    PREF_SCOPE = "both"  # where learnt preferences apply: "both" orders, or "power" only (Fit stays about your colours)
    PREF_FILE = ""  # override the learnt-preference file (held-out evaluations); "" = the set's fitted.json
    PREF_SET = "early"  # which learnt strengths: "early" (pack 1 picks 1-5) or "all" (every pick)
    ORDER_WEIGHT = 0.0  # replaced by learnt preferences (FRA test top-1 55.4 and best deck win rate at weight 4); 2.0 when they are off. WR points per SD of how early top players take the card (their ATA), × castability; 0 = off
    TOP_WR = False  # rejected 2026-11-01 (FRA +0.1, TDM −0.7 top-drafter top-1). Use top players' win rates (the set's _Top dataset), shrunk toward all players' by sample size
    K_TOP = 300.0  # shrinkage for top-player rates: w = n_top / (n_top + K_TOP)
    LANDS = True  # round 2: lands exempt from the sparse prior; partial-overlap lands valued by the splashes they enable
    FIX_WEIGHT = 0.5  # share of each enabled splash's gain (over replacement) credited to the land
    FIX_CAP = 3.0  # most a land's fixing can add, in WR points
    LAND_DECK = False  # on-colour and colourless lands keep their lane value in the deck term instead of "replacement plus edge over an average card" (FRA train: on-pair duals lost ~6 points there; top drafters take them at ATA 6-9)
    LAND_PREF = False  # lands get no learnt preference: it is learnt on pack 1 picks 1-5, where nobody takes lands yet
    LAND_MAIN_DUAL = False  # a nonbasic land making two or more colours of the lane is worth at least replacement + LAND_MAIN_DUAL_EDGE (top players take main-pair duals mid-pack)
    LAND_MAIN_DUAL_EDGE = 3.0
    LAND_PREF_POS = False  # with LAND_PREF, lands keep a positive learnt preference (Theorist's Sanctum) and lose only a negative one
    HALF_PIVOT = 0  # from this many spells in the pool, lanes sharing no colour with the pool's top pair get P = 0 (winners change one colour, not both); 0 = off
    # Neutral by default; for testing ideas from top players per set (sets/<SET>/manual.json, "values"):
    FIX_BONUS = 0.0  # WR points for a land that taps for the lane and makes another colour, on top of the splashes it enables
    OWN_BUILDER = True  # deck building: companion/advisor_v2/deckbuilder.py (model win rate within top players' deck shape); False = upstream's builder
    OWN_FIXING = True  # pool fixing counted by lanes.pool_fixing (nonbasic lands, any-colour lands and mana); False = upstream's count_fixing, which also counts basic landcyclers as sources of every colour
    CYCLE_FIX = False  # basic landcyclers and any-colour mana rocks count as fixing for every colour (and earn the splashes they enable)
    THREE_COLOUR_AT = 0  # the pool counts as three-colour once it has this many one-pip splash spells for its top pair; 0 = never
    DOUBLE_PIP = 0.0  # in a three-colour pool, WR points off per extra pip of one colour ({U}{U}: one extra)
    DOUBLE_PIP_ALL = False  # apply DOUBLE_PIP in any pool: {1}{U}{U} is harder to cast on curve than {2}{U} even in two colours
    DOUBLE_PIP_CMC = 99  # only cards at or below this mana value lose points (a six-drop has time to find its second pip)
    CREATURE_TARGET = 15.0  # creatures wanted in 23 spells
    CREATURE_POINTS = 0.0  # WR points for a creature when the pool, projected to 23 spells, is fully short of CREATURE_TARGET; 0 = off
    CREATURE_FROM = 8  # spells in the pool before the creature need applies
    CHOICE_ADJUST = False  # final scores move toward how top drafters choose: + sum(beta_f / beta_v2 * f) for the pick's stage from the fitted "choices" model (tools/choice_model.py; synergy with the pool, deck needs, colour, fixing, mana value)
    STRAT_SYN = 0.0  # multiplier on the fitted per-card weights of each pair's strategy tags (fitted "strategy", analysis/strategy_win_loss.py; tags from sets/<SET>/roles.json), added to Fit per likely pair; 1 = the data's weights, 0 = off
    SYNERGY_FIT = 0.0  # weight on the choice model's synergy part in Fit at every stage (payoffs with their enablers in the pool, enablers for held payoffs, payoffs with none), x the share of likely pairs that can cast the card; 0 = off
    TIE_CHEAPER = True  # among cards within TIE_EPS of the top score, castable in the pool's top pair and within TIE_WR of each other's GIH WR, the cheaper goes first (equal power, lower cost: the guides' "Juza rule")
    TIE_EPS = 0.3
    SYN_TIE = 0.0  # synergy as a tie-breaker (Akshin, 2026-11-05): among cards within this many WR points of the top, the one with the most main-pair strategy credit (strategy_points) goes first; 0 = off. Flips within ~1 point went toward top drafters' picks (analysis/synergy_flips.py)
    TIE_WR = 0.5
    CHOICE_FEATURES = ""  # per stage and feature group, e.g. "p1close:synergy;p2open:synergy,colour,splash;fit:synergy,colour,splash,fixer,needs;late:all" (stages p1open, p1close, p2open, fit, late; groups synergy, needs, colour, splash, fixer, cmc, all); overrides CHOICE_STAGES when set
    CHOICE_STAGES = "all"  # which stages CHOICE_ADJUST applies to: "all", or a comma list of p1open (P1 picks 1-7), p1close (P1 8-11), p2open (P2 1-6), fit (P2 7-11, P3 1-11), late (picks 12-14)
    FIT_WEIGHTS = False  # Fit adds the per-pick weights of top drafters (fitted "blend") on synergy (themes, deck-mates, reviewed cards), splash and the deck-slot fills, in Fit's WR points; the Power/Fit switch stays
    FIT_PASSED = True  # the per-pick weights include "Passed": quality already passed in the card's colours to the neighbour passing now (needs.passed)
    FIT_WEIGHTS_CAP = 8.0  # largest WR-point change the per-pick terms make to one card's Fit
    BLEND = False  # rank by the learnt per-pick blend of top drafters' weights on Power, Fit, signal, wheel, synergy and deck needs (fitted "blend", tools/choice_model.py blend) instead of the Power/Fit switch
    CHOICE_IN_FIT = False  # apply the choice correction inside Fit (before the Power/Fit switch) instead of to the final order
    REMOVAL_SLACK_PICK = -1.0  # removal spells below top players' count before the picker's removal need starts; -1 = the builder's REMOVAL_SLACK
    REMOVAL_POINTS = 0.0  # WR points for a removal spell when the pool, projected to 23 spells, is fully short of its top pair's removal floor (top players' count there minus deckbuilder.REMOVAL_SLACK); 0 = off
    REMOVAL_FROM = 8  # spells in the pool before the removal need applies
    REMOVAL_SKIP = ""  # ";"-separated removal cards the removal need neither counts nor rewards (FRA: conditional removal, worth about 0 in hand within deck, docs/b1-within-removal-FRA.md)
    REMOVAL_FLAT = 0.0  # WR points for every removal spell at any count (FRA games: removal pays on, no ceiling: +0.8 per spell in the deck model, drafted not built); 0 = off
    FINISHER_POINTS = 0.0  # WR points for a finisher (cmc 5+, GIH WR one SD over the format mean) in proportion to how far the pool, projected to 23 spells, is under FINISHER_UP_TO castable finishers (FRA games: 1-2 +1.4, 3+ +4.0 points; decks hold what they drafted); 0 = off
    FINISHER_UP_TO = 3  # finishers a deck wants
    COPIES_MAX = 2  # copies of one card a deck wants
    COPIES_POINTS = 0.0  # WR points off per copy at or beyond COPIES_MAX already in the pool; 0 = off
    DECK_WEIGHT = 0.0  # Fit adds this share of the card's value above replacement in the likely pairs' best 23 (top players' shape penalty included) minus its own value: a card that would not make the deck loses, one that fills a shape gap gains; 0 = off
    DECK_FROM = 12  # spells in the pool before deck-aware value applies
    DECK_MIN_P = 0.05  # pairs below this probability are not built
    LANE_FLAT = "4:1,14:0.7,45:0.8"  # P(lane) ∝ P^τ by pick: picks 1-4 unchanged, 5-14 τ 0.7, 15-45 τ 0.8 (calibrated against top drafters' final pairs); "" = off
    COLOUR_SHIFT = ""  # per-colour WR points taken off a card's value, "U:2" or "U:2,B:-1"; a card gets the mean over its colours; "" = off
    WHEEL_EPS = 0.3  # within this of the best score, prefer the card that wheels less
    WHEEL_SPLIT = False  # apply the WHEEL_EPS tiebreak to the split's final order too (it used to apply only with SPLIT off)
    THREE_LANES = 0.0  # from pack 2, each likely pair passes this share of the fitted chance that top drafters end in it plus a third colour (fitted "three_draft", tools/fit_three_draft.py) to that three-colour lane; 0 = two-colour lanes only
    THREE_MIN_P = 0.05  # pairs below this probability are not split
    SIGNAL_OTHER = 0.3  # Akshin, 2026-11-07: pack 1's side counts 0.3 in pack 2; weight of the other neighbour's signal added to the one passing now (pack 2 still learns from pack 1's side, which passes again in pack 3); 0 = off
    WHEEL_PENALTY = 0.0  # WR points taken from Power and Fit per 100% chance the card comes back; 0 = off

    land_names = frozenset()  # lands in the pack being scored (LAND_PREF)

    def __init__(self, set_metrics, taken_cards, seen_packs=None):
        self.metrics = set_metrics
        self.pool = [c for c in (taken_cards or []) if c]
        config.activate(next((c.get("set") for c in self.pool if c.get("set")), None))
        self.mean, self.std = set_metrics.get_metrics(ALL, "gihwr")
        if self.mean <= 0:
            self.mean = 54.0
        if self.std <= 0:
            self.std = 4.0
        _, alsa_spread = set_metrics.get_metrics(ALL, "alsa")
        spread = alsa_spread if alsa_spread > 0 else 2.0
        seen = seen_packs or []
        self.seen_packs = seen
        self.current_pack = max((p[0] for p in seen), default=1)
        if signal_mod.DIRECTIONAL:
            # Read only the neighbour passing to us now: packs 1 and 3 come from one side, pack 2 from the other.
            self.signal_sides = side_signals(seen, self.mean, self.std, spread)
            self.signal_side = side_for_pack(self.current_pack)
            self.signals = self.signal_sides[self.signal_side]
            if self.SIGNAL_OTHER:
                other = next((v for k, v in self.signal_sides.items() if k != self.signal_side), {})
                self.signals = {c: v + self.SIGNAL_OTHER * other.get(c, 0.0) for c, v in self.signals.items()}
        else:
            self.signal_sides, self.signal_side = {}, ""
            self.signals = colour_signals(seen, self.mean, self.std, spread)
        self.prior = {}
        for lane in LANES:
            m, _ = set_metrics.get_metrics(lane, "gihwr")
            self.prior[lane] = (m - self.mean) / self.std if m > 0 else 0.0
            self.prior[lane] += config.ACTIVE["tables"]["pairs"].get(lane, 0.0) / self.std  # manual pair strength, in WR points
        self.replacement = self.mean - self.REPLACEMENT_SD * self.std
        self.fixing = (pool_fixing(self.pool) if self.OWN_FIXING else count_fixing(self.pool)) if self.pool else {}
        if self.CYCLE_FIX:
            for c in self.pool:
                if any_colour_fixer(c):
                    for colour in "WUBRG":
                        self.fixing[colour] = self.fixing.get(colour, 0) + 1
        self.names = {}
        for c in self.pool:
            self.names[c.get("name")] = self.names.get(c.get("name"), 0) + 1
        spells = sum(1 for c in self.pool if "Land" not in (c.get("types") or []))
        self.spells = spells  # set early: value() reads it (TOP_END_POINTS) while the lanes are built
        self.colour_count = {x: sum(1 for c in self.pool if "Land" not in (c.get("types") or []) and x in (c.get("colors") or []))
                             for x in "WUBRG"}
        progress = min(1.0, spells / self.COMMIT_BY) if self.COMMIT_BY > 0 else 0.0
        self.progress = progress
        self.steer = self.STEER * (1.0 - progress)  # the candidate's pull fades as the pool becomes a deck
        self.alpha = self.ALPHA + (self.ALPHA_LATE - self.ALPHA) * progress
        self.commit = commitment(self.pool, self.mean, self.std)
        if self.CARD_PAIRS_PULL:
            # Setup bombs pull toward the pairs they shine in (fitted card_pairs, tools/card_pairs.py), not just their colours.
            homes = config.fitted(set_code_of(self.pool), "card_pairs") or {}
            for c in self.pool:
                for lane in homes.get(c.get("name"), homes.get((c.get("name") or "").split(" // ")[0], [])):
                    if lane in self.commit:
                        self.commit[lane] += self.CARD_PAIRS_PULL * quality(c, self.mean, self.std)
        if self.COLOUR_FLOOR:
            # Akshin, 2026-11-07: one card means nothing. A colour with fewer than COLOUR_FLOOR_CARDS spells counts exactly
            # COLOUR_FLOOR quality points (none and one card alike); with more it counts its own quality, at least the floor.
            # FLOOR_BY > 0 fades the floor by that many spells. ponytail: pool only; a candidate's STEER pull adds on top.
            floor = self.COLOUR_FLOOR * (max(0.0, 1.0 - spells / self.FLOOR_BY) if self.FLOOR_BY > 0 else 1.0)
            have, count = dict.fromkeys("WUBRG", 0.0), dict.fromkeys("WUBRG", 0)
            for c in self.pool:
                if "Land" in (c.get("types") or []):
                    continue
                q = quality(c, self.mean, self.std)
                for col in c.get("colors") or ():
                    if col in have:
                        have[col] += q
                        count[col] += 1
            target = {col: floor if count[col] < self.COLOUR_FLOOR_CARDS else max(have[col], floor) for col in have}
            for lane in self.commit:
                self.commit[lane] += sum(target[col] - have[col] for col in lane if col in have)
        self.anchor = None
        free = self._lanes(self.commit)
        self.top_lane = max(free, key=free.get)
        # Top end on pace: 5+ drops held vs the typical deck of the top pair, scaled to the spells held (1 = on pace or ahead).
        target = ((((config.fitted(set_code_of(self.pool), "decks") or {}).get("pairs") or {}).get(self.top_lane) or {}).get("five_plus")
                  if set_code_of(self.pool) else None) or 3.5
        due = target * spells / 23.0
        have = sum(1 for c in self.pool if "Land" not in (c.get("types") or []) and (c.get("cmc") or 0) >= 5)
        self.top_end_pace = min(1.0, have / due) if due > 0 else 1.0
        splashes = sum(1 for c in self.pool if "Land" not in (c.get("types") or []) and len(off_pips(c, self.top_lane)) == 1)
        self.three_colour = bool(self.THREE_COLOUR_AT) and splashes >= self.THREE_COLOUR_AT
        creatures = sum(1 for c in self.pool if "Creature" in (c.get("types") or []))
        self.creature_need = (max(0.0, self.CREATURE_TARGET - creatures * 23.0 / spells) / self.CREATURE_TARGET
                              if self.CREATURE_POINTS and spells >= self.CREATURE_FROM else 0.0)
        self.removal_need = 0.0
        if self.REMOVAL_POINTS and spells >= self.REMOVAL_FROM and set_code_of(self.pool):
            # FRA games: decks under 6 removal lose ~0.6 WR points beyond their cards; the builder keeps a floor, the picker leans toward it.
            from companion.advisor_v2 import deckbuilder  # deckbuilder imports this module
            target = ((config.fitted(set_code_of(self.pool), "decks") or {}).get("pairs") or {}).get(self.top_lane) or {}
            slack = deckbuilder.REMOVAL_SLACK if self.REMOVAL_SLACK_PICK < 0 else self.REMOVAL_SLACK_PICK
            floor = max(0.0, target.get("removal", 6.0 + slack) - slack - (deckbuilder.skip_mean(self.top_lane) if self.REMOVAL_SKIP else 0.0))
            removal = sum(1 for c in self.pool if "Land" not in (c.get("types") or []) and self.counts_as_removal(c))
            self.removal_need = max(0.0, floor - removal * 23.0 / spells) / floor if floor else 0.0
        self.finisher_need = 0.0
        if self.FINISHER_POINTS and spells >= self.REMOVAL_FROM:
            have = sum(1 for c in self.pool if self.is_finisher(c) and castable(c, self.top_lane))
            self.finisher_need = max(0.0, self.FINISHER_UP_TO - have * 23.0 / spells) / self.FINISHER_UP_TO
        if self.HALF_PIVOT and spells >= self.HALF_PIVOT:
            self.anchor = self.top_lane
        self.lanes = self._lanes(self.commit)  # pool only; what the Analyst shows and what Fit uses
        # Per-colour ceiling for COLOUR_PACE: COLOUR_PACE in a pair; COLOUR_PACE_THREE by the pool's count order in a triple;
        # averaged over the lanes holding the colour, by their chance.
        three = [float(v) for v in self.COLOUR_PACE_THREE.split(",")]
        self.colour_pace = {}
        for x in "WUBRG":
            num = den = 0.0
            for lane, p in self.lanes.items():
                if x not in lane or p <= 0:
                    continue
                if len(lane) == 3:
                    order = sorted(lane, key=lambda y: (-self.colour_count[y], "WUBRG".index(y)))
                    num += p * three[order.index(x)]
                else:
                    num += p * self.COLOUR_PACE
                den += p
            self.colour_pace[x] = num / den if den else self.COLOUR_PACE
        self.spells = spells
        self.decks = {}  # lane -> [(value, card)] of the pool's castable spells, best first; built on demand
        self.split = None
        self.set_code_now, self.pack_now = None, 1  # set by evaluate_pack
        self.order, self.prefs = {}, {}  # filled per pack in evaluate_pack
        set_code = next((c.get("set") for c in self.pool if c.get("set")), None)
        self.set_code = set_code
        self.offcolour = config.fitted(set_code, "offcolour") if set_code and self.OFF_PLAY else None
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
        power = self.lane_flat_power()
        if power != 1.0:
            p = {lane: v ** power for lane, v in p.items()}
            total = sum(p.values())
            p = {lane: v / total for lane, v in p.items()}
        return self._three(p)

    def _three(self, p):
        """Move each likely pair's chance of ending three-colour (fitted from top drafters, pack 2 on) to that lane."""
        set_code = set_code_of(self.pool)
        if self.THREE_OPEN:
            # Akshin, 2026-11-07: draft two colours + a splash; a third base colour only with THREE_GOOD good cards in it
            # (quality >= THREE_GOOD_Q). Each likely pair passes THREE_OPEN of its chance, evenly, to its qualifying triples.
            good = {x: sum(1 for c in self.pool if "Land" not in (c.get("types") or []) and x in (c.get("colors") or [])
                           and quality(c, self.mean, self.std) >= self.THREE_GOOD_Q) for x in "WUBRG"}
            out = dict(p)
            for lane, v in p.items():
                if len(lane) != 2 or v < self.THREE_MIN_P:
                    continue
                thirds = [x for x in "WUBRG" if x not in lane and good[x] >= self.THREE_GOOD]
                if not thirds:
                    continue
                out[lane] -= v * self.THREE_OPEN
                for x in thirds:
                    three = "".join(sorted(lane + x, key="WUBRG".index))
                    out[three] = out.get(three, 0.0) + v * self.THREE_OPEN / len(thirds)
            return out
        rule = config.fitted(set_code, "three_draft") if self.THREE_LANES and set_code and len(self.pool) >= 14 else None
        if not rule:
            return p
        from companion.advisor_v2 import deckbuilder  # deckbuilder imports this module
        out = dict(p)
        for lane, v in p.items():
            if v < self.THREE_MIN_P:
                continue
            f3 = deckbuilder.three_features(self.pool, lane)
            if not f3[3]:
                continue
            named = {**dict(zip(deckbuilder.THREE_INPUTS, f3)), "pick": len(self.pool) + 1}
            share = v * min(1.0, self.THREE_LANES * deckbuilder.rule_chance(rule, [named[k] for k in rule["inputs"]]))
            three = "".join(sorted(lane + f3[3], key="WUBRG".index))
            out[lane] -= share
            out[three] = out.get(three, 0.0) + share
        return out

    def lane_flat_power(self):
        """The LANE_FLAT exponent for the pick being made (pool size + 1)."""
        pick = len(self.pool) + 1
        for part in filter(None, self.LANE_FLAT.split(",")):
            last, power = part.split(":")
            if pick <= int(last):
                return float(power)
        return 1.0

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

    def off_play(self, card, lane):
        """Chance top players play this card in a deck whose core pair is the lane (the fitted "offcolour" rule)."""
        r = self.offcolour
        wr = float(all_decks(card).get("gihwr") or 0.0)
        z = r["w"][0] + r["w"][1] * (wr - r["mean"]) / r["std"] + r["w"][2] * (len(off_pips(card, lane)) >= 2)
        return 1.0 / (1.0 + math.exp(-z))

    def speculative(self, card, lane, pull=False):
        """Early draft: is this lane a live splash for the card even with no fixing yet?

        Narrow by default: only cards no two-colour lane can cast (three-colour cards), so a
        mono-colour card keeps pulling v2 into its own colours early (round 1's main gain).
        """
        if self.SPEC_SPLASH <= 0 or self.progress >= 1 or len(off_pips(card, lane)) != 1 or not self.splash_worthy(card, lane):
            return False
        wide = self.SPEC_ALL and (self.SPEC_ALL_PULL or not pull)
        return wide or not any(castable(card, other) for other in LANES)

    def splash_worthy(self, card, lane):
        """SPLASH_MIN_GIH: the card's GIH WR in the lane (All Decks where the lane has none) reaches the floor; True when off."""
        if self.SPLASH_MIN_GIH <= 0:
            return True
        from companion.advisor_v2 import deckbuilder  # deckbuilder imports this module

        return deckbuilder._wr(card, lane[:2]) >= self.SPLASH_MIN_GIH

    def splash_colour(self, card, lane):
        """The colour a one-pip splash needs, when splashing is on and the pool has a fixer for it; else None."""
        if self.SPLASH <= 0 or not self.splash_worthy(card, lane):
            return None
        off = off_pips(card, lane)
        if len(off) != 1:
            return None
        return next((c for c in off[0] if self.fixing.get(c, 0) > 0), None)

    def cycle_fix_gain(self, card, lane):
        """A landcycler or any-colour rock in the lane enables every off-colour one-pip splash with no source yet."""
        new = [c for c in "WUBRG" if c not in lane and not self.fixing.get(c, 0)]
        gain = 0.0
        for spell in self.pool:
            if "Land" in (spell.get("types") or []):
                continue
            off = off_pips(spell, lane)
            if len(off) == 1 and any(c in off[0] for c in new) and self.splash_worthy(spell, lane):
                gain += max(0.0, self.shrunk_wr(spell, lane)[0] * self.SPLASH - self.replacement)
        return min(self.FIX_CAP, self.FIX_WEIGHT * gain)

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
            if len(off) == 1 and any(c in off[0] for c in new) and self.splash_worthy(spell, lane):
                gain += max(0.0, self.shrunk_wr(spell, lane)[0] * self.SPLASH - self.replacement)
        return min(self.FIX_CAP, self.FIX_WEIGHT * gain)

    def colour_shift(self, card):
        if not self.COLOUR_SHIFT or "Land" in (card.get("types") or []):
            return 0.0
        shift = {k: float(v) for k, v in (x.split(":") for x in self.COLOUR_SHIFT.split(","))}
        cols = card.get("colors") or []
        return sum(shift.get(c, 0.0) for c in cols) / len(cols) if cols else 0.0

    def counts_as_removal(self, card):
        return "removal" in config.tags(card) and card.get("name") not in self.REMOVAL_SKIP.split(";")

    def is_finisher(self, card):
        """A spell of mana value 5+ whose GIH WR is a standard deviation over the format mean (the finisher field of the deck tables)."""
        if "Land" in (card.get("types") or []) or (card.get("cmc") or 0) < 5:
            return False
        wr = float(all_decks(card).get("gihwr") or 0.0)
        return wr > 0 and wr >= self.mean + self.std

    def splash_share(self, card, lane):
        """Akshin, 2026-11-07: a splash is worth what its colour brings: the candidate's quality plus every pool card of the
        splash colour, over the total quality of the pool castable in the pair; x SPLASH_POOL_K, capped at SPLASH_POOL_CAP."""
        off = off_pips(card, lane)
        if len(off) != 1:
            return 0.0
        x = off[0][0] if isinstance(off[0], (list, tuple)) else off[0]
        splash, deck = quality(card, self.mean, self.std), 0.0
        for c in self.pool:
            if "Land" in (c.get("types") or []):
                continue
            q = quality(c, self.mean, self.std)
            if castable(c, lane):
                deck += q
            elif x in (c.get("colors") or []):
                splash += q
        return min(self.SPLASH_POOL_CAP, self.SPLASH_POOL_K * splash / deck) if deck > 0 else self.SPLASH_POOL_CAP

    def adjust(self, card, lane):
        """Manual per-set adjustments and the neutral-by-default deck-need settings, in WR points."""
        t = config.ACTIVE["tables"]
        name, types = card.get("name"), card.get("types") or []
        kind = "land" if "Land" in types else "creature" if "Creature" in types else "other"
        a = t["pairs"].get(lane, 0.0) + t["types"].get(kind, 0.0) + t["cards"].get(name, 0.0)
        for combo in t["combos"]:
            cards = combo.get("cards") or []
            if name in cards and any(other in self.names for other in cards if other != name):
                a += combo.get("points", 0.0)
        if self.creature_need and kind == "creature":
            a += self.CREATURE_POINTS * self.creature_need
        if self.COLOURLESS_EARLY and not (card.get("colors") or []) and self.pack_now == 1:
            # Colourless cards fit every lane, so spread early lanes overrate them; drafters leave them for later (ATA 7-8).
            fade = max(0.0, 1.0 - (getattr(self, "pick_now", 1) - 1) / self.COLOURLESS_FADE_BY) if self.COLOURLESS_FADE_BY else 1.0
            a -= self.COLOURLESS_EARLY * fade
        if self.PLAY_RATE_POINTS and kind != "land":
            # Rarely played when picked (17Lands play rate): niche or unplayable cards, even when their win rate looks fine.
            rate = all_decks(card).get("play_rate")
            if rate is not None:
                a -= self.PLAY_RATE_POINTS * max(0.0, self.PLAY_RATE_FROM - float(rate)) / self.PLAY_RATE_FROM
        if self.removal_need and self.counts_as_removal(card) and kind != "land":
            a += self.REMOVAL_POINTS * self.removal_need
        if self.REMOVAL_FLAT and "removal" in config.tags(card) and kind != "land":
            a += self.REMOVAL_FLAT
        if (self.TOP_END_POINTS and kind != "land" and (card.get("cmc") or 0) >= 6 and not self.is_finisher(card)
                and getattr(self, "pick_now", 1) <= self.TOP_END_PICKS):
            # Videos (8 of 21): do not take 6-7 drops highly; lock up one or two finishers, then curve. Full while the pool's
            # top end is on pace, less as it falls short (Akshin: by the pool, not the pick number).
            a -= self.TOP_END_POINTS * getattr(self, "top_end_pace", 1.0)
        cols = [c for c in (card.get("colors") or []) if c in "WUBRG"]
        if self.COLOUR_PACE_POINTS and kind != "land" and len(cols) == 1:
            # Akshin, 2026-11-07: two main colours, not one; a colour ahead of COLOUR_PACE cards per 23 spells wants less.
            due = getattr(self, "colour_pace", {}).get(cols[0], self.COLOUR_PACE) * getattr(self, "spells", 0) / 23.0
            counts = getattr(self, "colour_count", {})
            have = counts.get(cols[0], 0)
            absolute = min(1.0, (have - due) / due) if due > 0 and have > due else 0.0
            # Akshin: relative to the next colour too; the 8th blue card with a 2-card second colour is penalised.
            other = max((v for k, v in counts.items() if k != cols[0]), default=0)
            relative = min(1.0, max(0.0, have - other - self.COLOUR_LEAD) / self.COLOUR_LEAD_SCALE)
            a -= self.COLOUR_PACE_POINTS * max(absolute, relative)
        if self.finisher_need and self.is_finisher(card):
            a += self.FINISHER_POINTS * self.finisher_need
        if self.COPIES_POINTS and self.names.get(name, 0) >= self.COPIES_MAX:
            a -= self.COPIES_POINTS * (self.names[name] - self.COPIES_MAX + 1)
        if self.DOUBLE_PIP and (self.three_colour or self.DOUBLE_PIP_ALL) and (card.get("cmc") or 0) <= self.DOUBLE_PIP_CMC:
            singles = [o[0] for o in pips(card) if len(o) == 1]
            a -= self.DOUBLE_PIP * max(0, max((singles.count(c) for c in set(singles)), default=0) - 1)
        return a

    def value(self, card, lane):
        v = self._value(card, lane)
        if v == self.replacement and not castable(card, lane):
            return v  # an uncastable card stays at replacement level
        return v + self.adjust(card, lane)

    def _value(self, card, lane):
        if self.LANDS and "Land" in (card.get("types") or []):
            colours = card.get("colors") or []
            if not colours or all(c in lane for c in colours):
                v = self.shrunk_wr(card, lane)[0]  # colourless or on-lane: its own win rate
                if self.LAND_MAIN_DUAL and len(colours) >= 2 and "Basic" not in (card.get("types") or []):
                    v = max(v, self.replacement + self.LAND_MAIN_DUAL_EDGE)
                return v
            if any(c in lane for c in colours):
                v = self.replacement + self.fixing_gain(card, lane) + self.FIX_BONUS  # taps for the lane and fixes a splash
                if self.DUAL_OPEN_EDGE and len(colours) >= 2 and "Basic" not in (card.get("types") or []):
                    # Akshin, 2026-11-07: early on a dual keeps its other colour open (pair or splash); fades with the pool.
                    v = max(v, self.replacement + self.DUAL_OPEN_EDGE * (1.0 - self.progress))
                return v
            return self.replacement
        if not castable(card, lane):
            if not self.splash_worthy(card, lane):
                return self.replacement
            if self.OFF_PLAY and self.offcolour:
                play = self.off_play(card, lane)
                if self.SPLASH_POOL_K:
                    play = max(play, self.splash_share(card, lane))
                return self.replacement + max(0.0, self.shrunk_wr(card, lane)[0] - self.replacement) * play
            if self.splash_colour(card, lane):
                return max(self.replacement, self.shrunk_wr(card, lane)[0] * self.SPLASH)
            if self.speculative(card, lane):
                # Early on, a card one pip off a lane is still a live splash: top drafters take the bomb and find the fixing later.
                return max(self.replacement, self.shrunk_wr(card, lane)[0] * self.SPEC_SPLASH * (1 - self.progress))
            return self.replacement
        v = (self.shrunk_wr(card, lane)[0] - self.colour_shift(card)) * self.composition.multiplier(card, lane)
        if self.CYCLE_FIX and any_colour_fixer(card):
            v += self.cycle_fix_gain(card, lane)
        if synergy.ENABLED:
            v += synergy.bonus(self.synergy, card, self.pool, lane)
        return v

    def evaluate_pack(self, pack_cards, current_pick, current_pack=1):
        if not pack_cards:
            return []
        self.land_names = {c.get("name") for c in pack_cards if "Land" in (c.get("types") or [])}
        wrs = [float(all_decks(c).get("gihwr") or 0.0) for c in pack_cards]
        wrs = [w for w in wrs if w > 0]
        pack_mean = statistics.mean(wrs) if wrs else self.mean
        pack_std = statistics.pstdev(wrs) if len(wrs) > 1 else self.std
        pack_std = pack_std if pack_std > 0 else self.std
        by_wr = sorted(pack_cards, key=lambda c: float(all_decks(c).get("gihwr") or 0.0), reverse=True)
        rank_in_pack = {c.get("name"): i for i, c in enumerate(by_wr)}
        upstream = DraftAdvisor(self.metrics, self.pool)  # only for its wheel polynomial
        set_code = next((c.get("set") for c in (*self.pool, *pack_cards) if c.get("set")), None)
        config.activate(set_code)
        self.set_code_now, self.pack_now, self.pick_now = set_code, current_pack, current_pick
        if self.OFF_PLAY and not self.offcolour and set_code:
            self.offcolour = config.fitted(set_code, "offcolour")
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
                if castable(card, lane) or self.speculative(card, lane, pull=True):
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

        if self.SPLIT and self.BLEND and set_code and config.fitted(set_code, "blend"):
            return self._tie_synergy(self._tie_cheaper(self._blend(self._split(pack_cards, recs, current_pick, set_code), pack_cards, current_pick, current_pack, set_code), pack_cards), pack_cards)
        if self.SPLIT:
            return self._tie_synergy(self._tie_cheaper(self._choice_adjust(self._split(pack_cards, recs, current_pick, set_code), pack_cards, current_pick, current_pack, set_code), pack_cards), pack_cards)
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

    def _tie_cheaper(self, recs, pack_cards):
        """Equal power, lower cost: among cards tied at the top (TIE_EPS), castable in the pool's top pair and of
        near-equal raw rating (TIE_WR), put the cheapest first. Never overrides a real difference."""
        if not self.TIE_CHEAPER or len(recs) < 2 or recs[0].contextual_score <= 0:
            return recs
        by_name = {c.get("name"): c for c in pack_cards}
        top = recs[0]
        wr = lambda r: float(all_decks(by_name[r.card_name]).get("gihwr") or 0.0)
        tied = [r for r in recs if top.contextual_score - r.contextual_score <= self.TIE_EPS and r.contextual_score > 0
                and castable(by_name[r.card_name], self.top_lane) and abs(wr(r) - wr(top)) <= self.TIE_WR]
        if len(tied) < 2 or not castable(by_name[top.card_name], self.top_lane):
            return recs
        cmc = lambda r: by_name[r.card_name].get("cmc") or 0
        first = min(tied, key=lambda r: (cmc(r), -r.contextual_score))
        if first is top or cmc(first) >= cmc(top):
            return recs
        recs.remove(first)
        recs.insert(0, first)
        first.contextual_score = round(top.contextual_score + 0.01, 2)
        first.reasoning.insert(0, f"Tie on power with {top.card_name}: cheaper ({cmc(first):g} vs {cmc(top):g} mana)")
        return recs

    def _tie_synergy(self, recs, pack_cards):
        """Among cards within SYN_TIE of the top score, put first the one with the most main-pair strategy credit, if it has
        more than the top card. Outside the window card values decide alone."""
        if not self.SYN_TIE or len(recs) < 2 or recs[0].contextual_score <= 0 or not self.set_code_now:
            return recs
        by_name = {c.get("name"): c for c in pack_cards}
        top = recs[0]
        close = [r for r in recs if r.contextual_score > 0 and top.contextual_score - r.contextual_score <= self.SYN_TIE]
        syn = {r.card_name: self.strategy_points(by_name[r.card_name]) for r in close}
        first = max(close, key=lambda r: (syn[r.card_name], r.contextual_score))
        if first is top or syn[first.card_name] <= syn[top.card_name]:
            return recs
        recs.remove(first)
        recs.insert(0, first)
        first.reasoning.insert(0, f"Synergy breaks a near-tie with {top.card_name} ({top.contextual_score - first.contextual_score:.1f} points apart)")
        first.contextual_score = round(top.contextual_score + 0.01, 2)
        return recs

    def blend_inputs(self, card, pick, wheel, set_code):
        """The blend's inputs for a card (choice_model.CHART order): Power, Fit (pool-weighted value), signal for its
        colours, wheel chance, then Fit's parts (blend_parts)."""
        spell = "Land" not in (card.get("types") or [])
        colours = [x for x in (card.get("colors") or []) if x in self.signals] if spell else []
        return ([self.power(card, pick), sum(p * self.value(card, l) for l, p in self.lanes.items()),
                 sum(self.signals[c] for c in colours) / len(colours) if colours else 0.0, wheel]
                + self.blend_parts(card, set_code))

    def _blend(self, recs, pack_cards, pick, pack_no, set_code):
        """Order by the learnt blend for this pick; scores in WR points (the utility over Power's and Fit's weights)."""
        table = config.fitted(set_code, "blend")
        picks = table["picks"]
        key = f"{pack_no}-{pick}"
        if key not in picks:
            same = [k for k in picks if k.startswith(f"{pack_no}-")]
            if not same:
                return recs
            key = min(same, key=lambda k: abs(int(k.split("-")[1]) - pick))
        w, names = picks[key], table["inputs"]
        scale = w[0] + w[1] if w[0] + w[1] > 0 else 1.0
        by_name = {c.get("name"): c for c in pack_cards}
        playable = [r for r in recs if r.contextual_score > 0]
        for r in playable:
            x = self.blend_inputs(by_name[r.card_name], pick, (r.wheel_chance or 0.0) / 100.0, set_code)
            parts = [wi * xi for wi, xi in zip(w, x)]
            r.contextual_score = round(sum(parts) / scale, 2)
            top = sorted((p / scale, n) for p, n in zip(parts[2:], names[2:]) if abs(p / scale) >= 0.3)
            if top:
                r.reasoning.append("Blend: " + ", ".join(f"{n} {v:+.1f}" for v, n in top[::-1]))
        playable.sort(key=lambda r: r.contextual_score, reverse=True)
        return playable + [r for r in recs if r.contextual_score <= 0]

    def _choice_adjust(self, recs, pack_cards, pick, pack_no, set_code):
        """Re-rank the final order by the fitted choice model of top drafters (CHOICE_ADJUST); scores stay in WR points."""
        if not self.CHOICE_ADJUST or self.CHOICE_IN_FIT or not set_code:
            return recs
        from companion.advisor_v2 import choice

        by_name = {c.get("name"): c for c in pack_cards}
        lane = choice.top_lane(self.pool)
        playable = [r for r in recs if r.contextual_score > 0]
        for r in playable:
            adj = choice.correction(by_name[r.card_name], self.pool, set_code, pack_no, pick, lane, self.CHOICE_STAGES,
                                    skip=choice.SYNERGY if self.SYNERGY_FIT else (), spec=self.CHOICE_FEATURES)
            if abs(adj) >= 0.05:
                r.contextual_score = round(r.contextual_score + adj, 2)
                r.reasoning.append(f"Top drafters' choices: {adj:+.1f}")
        playable.sort(key=lambda r: r.contextual_score, reverse=True)
        return playable + [r for r in recs if r.contextual_score <= 0]

    def order_prior(self, name, pick, prefs=True):
        """Prior in WR points from what top players reach for: their average pick position (ORDER_WEIGHT)
        and, when switched on, the preference learnt from their choices (PREF_WEIGHT)."""
        fade = max(0.0, 1.0 - (pick - 1) / self.ORDER_FADE_BY) if self.ORDER_FADE_BY else 1.0
        out = self.ORDER_WEIGHT * self.order.get(name, 0.0) if self.ORDER_WEIGHT else 0.0
        if self.PREF_WEIGHT and prefs:
            pref = self.prefs.get(name, self.prefs.get(name.split(" // ")[0], 0.0))
            if self.LAND_PREF and name in self.land_names:
                pref = max(0.0, pref) if self.LAND_PREF_POS else 0.0
            out += self.PREF_WEIGHT * pref
        if self.PRIOR_CAP:
            out = max(-self.PRIOR_CAP, min(self.PRIOR_CAP, out))  # breaks ties, cannot override a real win-rate gap
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
        synergy = 0.0
        if self.SYNERGY_FIT and self.set_code_now:
            from companion.advisor_v2 import choice

            synergy = self.SYNERGY_FIT * choice.synergy_points(card, self.pool, self.set_code_now, self.pack_now, pick) * live
        if self.CHOICE_ADJUST and self.CHOICE_IN_FIT and self.set_code_now:
            from companion.advisor_v2 import choice

            synergy += choice.correction(card, self.pool, self.set_code_now, self.pack_now, pick, choice.top_lane(self.pool), self.CHOICE_STAGES,
                                         skip=choice.SYNERGY if self.SYNERGY_FIT else (), spec=self.CHOICE_FEATURES)
        if self.FIT_WEIGHTS and self.set_code_now:
            synergy += self.fit_weights(card, pick)
        if self.STRAT_SYN and self.set_code_now:
            synergy += self.STRAT_SYN * self.strategy_points(card)
        return score + self.order_prior(card.get("name", ""), pick, prefs=self.PREF_SCOPE == "both") * live + self.deck_term(card) + synergy

    def strategy_points(self, card):
        """WR points for the card's tags in the pool's main pair's strategy, weighted by that pair's chance. Credit from other
        pairs moved picks away from top drafters' (analysis/synergy_flips.py: 37% helped off-plan vs 55% on-plan)."""
        lane = self.top_lane
        if not (castable(card, lane) or self.speculative(card, lane)):
            return 0.0
        weights = config.fitted(self.set_code_now, "strategy").get(lane, {})
        tags = set(config.roles(self.set_code_now).get(card.get("name", "").split(" // ")[0], ()))
        if "Creature" not in (card.get("types") or []) and "Land" not in (card.get("types") or []):
            tags.add("spells")  # UR's enabler is any noncreature spell
        return self.lanes.get(lane, 0.0) * sum(w for t, w in weights.items() if t in tags)

    FIT_PARTS_FROM = 4  # blend inputs from here on are Fit's parts (before: Power, Fit, signal, wheel)

    def fit_weights(self, card, pick):
        """Top drafters' per-pick weights on Fit's parts, over their weight on Fit itself: WR points added to Fit."""
        table = config.fitted(self.set_code_now, "blend")
        if not table:
            return 0.0
        picks = table["picks"]
        key = f"{self.pack_now}-{pick}"
        if key not in picks:
            same = [k for k in picks if k.startswith(f"{self.pack_now}-")]
            if not same:
                return 0.0
            key = min(same, key=lambda k: abs(int(k.split("-")[1]) - pick))
        w = picks[key]
        if w[1] <= 0.02:
            return 0.0
        # Akshin, 2026-11-07: over the engine's own likely pairs, not choice.top_lane's single pair (which kept UR while the
        # lanes had moved on, so a W/B card got -4 and a blue one +4).
        lanes = [(lane, p) for lane, p in self.lanes.items() if len(lane) == 2 and p >= 0.05] if self.FIT_WEIGHTS_LANES else [(None, 1.0)]
        total = sum(p for _, p in lanes) or 1.0
        out = sum(p * sum(wi * xi for wi, xi in zip(w[self.FIT_PARTS_FROM:], self.blend_parts(card, self.set_code_now, pick, lane)))
                  for lane, p in lanes) / total / w[1]
        return max(-self.FIT_WEIGHTS_CAP, min(self.FIT_WEIGHTS_CAP, out))

    def blend_parts(self, card, set_code, pick=0, lane=None):
        """Fit's parts among the blend inputs (choice_model.CHART from Themes on), for lane (default: choice.top_lane)."""
        from companion.advisor_v2 import choice, needs

        tags = choice.theme_tags(set_code)
        targets = (config.fitted(set_code, "decks") or {}).get("pairs") or {}
        lane = lane or choice.top_lane(self.pool)
        x = choice.features(card, self.pool, lane, list(tags), tags, targets, 0.0, 0.0)
        wr = needs.set_wr(set_code)
        passed = needs.passed(card, self.pool, self.seen_packs, self.pack_now, pick, *wr) if self.FIT_PASSED else 0.0
        return ([x[1] + x[2], needs.mate_score(card, self.pool, lane, set_code), needs.card_synergy(card, self.pool, set_code), x[8]]
                + needs.fills(card, self.pool, lane, (targets.get(lane) or {}).get("dims"), *wr) + [passed])

    def deck_term(self, card):
        """Value above replacement: in each likely pair, the card is worth replacement level plus how much the best 23
        improves with it (empty slots count at replacement, shape penalty included); the term is that less its own
        value. Zero while the deck has room and the shape is fine; down to replacement level once the card would only
        push out something nearly as good or not make the deck; positive when it fills a gap in the shape."""
        targets = (config.fitted(self.set_code, "decks") or {}).get("pairs") if self.set_code else None
        if not self.DECK_WEIGHT or not targets or self.spells < self.DECK_FROM:
            return 0.0
        from companion.advisor_v2 import deckbuilder  # deckbuilder imports this module
        out = 0.0
        if "Land" in (card.get("types") or []):
            if self.LAND_DECK:
                return 0.0
            # A land takes a basic's slot, not a spell's: replacement plus its edge over a basic (an average card).
            # Fixing lands are already valued from replacement up (fixing_gain), so only colourless and on-lane ones.
            colours = card.get("colors") or []
            for lane, p in self.lanes.items():
                if p >= self.DECK_MIN_P and all(c in lane for c in colours):
                    v = self.value(card, lane)
                    out += p * (self.replacement + max(0.0, v - self.mean) - v)
            return self.DECK_WEIGHT * out
        for lane, p in self.lanes.items():
            if p < self.DECK_MIN_P or lane not in targets:
                continue
            if lane not in self.decks:
                # Every pool spell the lane can use, splashes included: they all compete for the 23 slots.
                spells = [(self.value(c, lane), c) for c in self.pool if "Land" not in (c.get("types") or [])]
                self.decks[lane] = sorted((x for x in spells if x[0] > self.replacement), key=lambda x: -x[0])
            now, v = self.decks[lane], self.value(card, lane)
            if v <= self.replacement:
                continue  # not playable here: worth replacement either way
            scale = min(1.0, (len(now) + 1) / deckbuilder.SPELLS)
            after = sorted(now + [(v, card)], key=lambda x: -x[0])
            # Never below zero: a card that would spoil the shape is left in the sideboard.
            gain = max(0.0, self._deck_score(after, lane, targets[lane], scale) - self._deck_score(now, lane, targets[lane], scale))
            out += p * (self.replacement + gain - v)  # worth replacement plus what it adds; Fit already counts v
        return self.DECK_WEIGHT * out

    def _deck_score(self, ranked, lane, target, scale):
        """The best 23's value, empty slots filled at replacement level, less the shape penalty."""
        from companion.advisor_v2 import deckbuilder
        top = ranked[:deckbuilder.SPELLS]
        filler = (deckbuilder.SPELLS - len(top)) * self.replacement
        return sum(v for v, _ in top) + filler - deckbuilder.shape_penalty([c for _, c in top], target, lane, scale=scale)

    def best_card_features(self, card):
        """(the pool's best card's All Decks GIH WR over the format mean, 0 for an empty pool; 1 if the card shares a colour
        with that best card or is colourless). Drafters holding a strong card switch less, and more when the switch keeps it."""
        spells = [c for c in self.pool if "Land" not in (c.get("types") or [])]
        if not spells:
            return 0.0, 0.0
        best = max(spells, key=lambda c: float(all_decks(c).get("gihwr") or 0.0))
        colours = set(card.get("colors") or []) if "Land" not in (card.get("types") or []) else set()
        return float(all_decks(best).get("gihwr") or 0.0) - self.mean, float(not colours or bool(colours & set(best.get("colors") or [])))

    def branch_features(self, card):
        """(1 if the card keeps a colour of the pool's top pair or is colourless, else 0; that pair's probability)."""
        colours = set() if "Land" in (card.get("types") or []) else set(card.get("colors") or [])
        top = self.top_lane
        if self.KEEP_ANY:
            # Taking the card moves the deck to a pair that casts it; it keeps a colour if such a pair overlaps the top pair.
            keeps = not colours or any(castable(card, lane) and set(lane) & set(top) for lane in LANES)
        else:
            keeps = not colours or bool(colours & set(top))
        return float(keeps), self.lanes.get(top, 0.0)

    def _split(self, pack_cards, recs, pick, set_code):
        """Final order from the two orders: Fit, unless the strongest card beats the best fit by more than the learnt gap."""
        by_name = {c.get("name"): c for c in pack_cards}
        playable = [r for r in recs if r.contextual_score > 0]
        if not playable:
            return recs
        wheel = {r.card_name: (r.wheel_chance or 0.0) / 100.0 for r in playable}
        power = {r.card_name: self.power(by_name[r.card_name], pick) - self.WHEEL_PENALTY * wheel[r.card_name] for r in playable}
        fit = {r.card_name: self.fit(by_name[r.card_name], pick) - self.WHEEL_PENALTY * wheel[r.card_name] for r in playable}
        spells = sum(1 for c in self.pool if "Land" not in (c.get("types") or []))
        if self.POWER_EARLY and self.POWER_EARLY_BY > 0:
            # Akshin, 2026-11-07: raw strength counts while the second colour is unformed (Fit dilutes every non-main card
            # across the open pairs and favours main-colour filler); fades as the second colour reaches POWER_EARLY_BY cards.
            second = sorted(self.colour_count.values(), reverse=True)[1]
            w = self.POWER_EARLY * max(0.0, 1.0 - second / self.POWER_EARLY_BY)
            fit = {n: (1.0 - w) * fit[n] + w * power[n] for n in fit}
        p_top = max(power, key=power.get)
        c_top = max(fit, key=fit.get)
        stage = stage_of(spells)
        gap = power[p_top] - power[c_top]
        table = branch_file(self.BRANCH_FILE) if self.BRANCH_FILE else (branch_thresholds(set_code) if set_code else {})
        if self.BRANCH_COLOUR and not self.BRANCH_FILE and set_code:
            table = branch_thresholds(set_code, "branch_colour") or table
        coef = table.get(stage) or {}
        c_strength = power[c_top] - self.mean
        shared, committed = self.branch_features(by_name[p_top])
        if coef.get("b", 0) > 0:
            # Learnt from top drafters: branch more readily early and when the best in-colour card is weak;
            # fitted files may also weigh whether the card keeps a colour and how settled the pool is.
            bomb = float(float((all_decks(by_name[p_top])).get("gihwr") or 0.0) >= self.BOMB_WR)
            best_wr, keeps_best = self.best_card_features(by_name[p_top])
            threshold = -(coef["a"] + coef["c"] * c_strength + coef.get("d", 0.0) * shared + coef.get("e", 0.0) * committed
                          + coef.get("f", 0.0) * bomb + coef.get("g", 0.0) * best_wr + coef.get("h", 0.0) * keeps_best) / coef["b"]
        else:
            threshold = self.BRANCH_DEFAULT
        if self.pack_now == 1 and 2 <= pick <= self.BRANCH_EARLY_TO:
            threshold += self.BRANCH_EARLY  # top drafters stay with their first picks' colours more than the fitted rule says
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
        return V2Recommendation(
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
            tags=config.tags(card),
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


config.BASELINE.update(config.snapshot(DraftAdvisorV2))  # the code's values are the baseline layer
