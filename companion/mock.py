"""Mock drafts the app can run itself: a seeded table of bots, or a replayed 17Lands draft.

Both write the same Player.log lines Arena writes, into a log the scanner
watches, so nothing downstream knows it is a mock. Seat 0 is the player (or a
tool following an engine); seven bots take the best GIH WR card and stick to
two colours once they have them. The bots' pools are ground truth for
open-colour detection in tests.
"""

import gzip
import json
import random
import uuid
from collections import Counter
from pathlib import Path


SEATS, PACK_SIZE, PACKS = 8, 14, 3
COLOURS = "WUBRG"
PASSER = {1: 7, 2: 1, 3: 7}  # which seat hands seat 0 its packs

JOIN = '[UnityCrossThreadLogger]==> Event_Join {"id":"%s","request":"{\\"EventName\\":\\"PremierDraft_%s_20260101\\"}"}\n'
NOTIFY = '[UnityCrossThreadLogger]Draft.Notify {"draftId":"%s","SelfPick":%d,"SelfPack":%d,"PackCards":"%s"}\n'
PICK = (
    '[UnityCrossThreadLogger]==> Event_PlayerDraftMakePick {"id":"%s","request":"{\\"DraftId\\":\\"%s\\",'
    '\\"GrpIds\\":[%s],\\"Pack\\":%d,\\"Pick\\":%d}"}\n'
)


def gihwr(card):
    return float((card.get("deck_colors") or {}).get("All Decks", {}).get("gihwr") or 0.0)


class Cards:
    """A set's dataset as the mock needs it: ids, names, rarity, colours, GIH WR."""

    def __init__(self, dataset_path):
        data = json.loads(Path(dataset_path).read_text())
        self.set_code = Path(dataset_path).name.split("_")[0]
        self.by_id = {i: c for i, c in data["card_ratings"].items() if gihwr(c) > 0}
        self.id_of = {}
        for i, c in data["card_ratings"].items():
            self.id_of.setdefault(c["name"], i)
            self.id_of.setdefault(c["name"].split(" // ")[0], i)
        self.by_rarity = {"common": [], "uncommon": [], "rare": [], "mythic": []}
        for i, c in self.by_id.items():
            if c.get("rarity") in self.by_rarity:
                self.by_rarity[c["rarity"]].append(i)

    def name(self, grp_id):
        return self.by_id[grp_id]["name"]

    def colours(self, grp_id):
        c = self.by_id[grp_id]
        return [] if "Land" in (c.get("types") or []) else list(c.get("colors") or [])


class Table:
    """Eight seats from a seed. pack() is seat 0's current pack; take() advances everyone."""

    def __init__(self, cards, seed):
        self.cards, self.seed = cards, seed
        self.rng = random.Random(seed)
        self.pools = [[] for _ in range(SEATS)]
        self.pack_no, self.pick = 1, 1
        self._deal()

    def _deal(self):
        r = self.cards.by_rarity
        self.packs = []
        for _ in range(SEATS):
            rare = r["mythic"] if r["mythic"] and self.rng.random() < 1 / 8 else r["rare"]
            self.packs.append(self.rng.sample(r["common"], 10) + self.rng.sample(r["uncommon"], 3) + [self.rng.choice(rare)])

    def _holding(self):
        direction = -1 if self.pack_no == 2 else 1
        return [self.packs[(s - direction * (self.pick - 1)) % SEATS] for s in range(SEATS)]

    @property
    def done(self):
        return self.pack_no > PACKS

    def pack(self):
        return list(self._holding()[0])

    def bot_pick(self, pack, pool):
        colours = Counter(c for i in pool for c in self.cards.colours(i))
        main = {c for c, _ in colours.most_common(2)} if len(pool) >= 3 else set()
        return max(pack, key=lambda i: gihwr(self.cards.by_id[i]) + (3.0 if main and set(self.cards.colours(i)) <= main else 0.0))

    def take(self, grp_id):
        holding = self._holding()
        holding[0].remove(grp_id)
        self.pools[0].append(grp_id)
        for s in range(1, SEATS):
            c = self.bot_pick(holding[s], self.pools[s])
            holding[s].remove(c)
            self.pools[s].append(c)
        self.pick += 1
        if self.pick > PACK_SIZE:
            self.pack_no, self.pick = self.pack_no + 1, 1
            if not self.done:
                self._deal()

    def openness(self, pack_no):
        """Truth for the seat that passed to us in that pack: open = colours it did not take."""
        counts = Counter(c for i in self.pools[PASSER[pack_no]] for c in self.cards.colours(i))
        return {c: -counts.get(c, 0) for c in COLOURS}


class RealDraft:
    """A sampled 17Lands draft (tools/replay/run.py sample) replayed pack by pack.

    Packs are what that human saw, so after a different pick the wheel can show
    a card already taken. Fine for a mock; not for the harness.
    """

    def __init__(self, cards, draft):
        self.cards, self.id = cards, draft["id"]
        self.picks = [(p, k, taken, [cards.id_of[n] for n in names if n in cards.id_of]) for p, k, taken, names, _ in draft["picks"]]
        self.i = 0

    @property
    def done(self):
        return self.i >= len(self.picks)

    @property
    def pack_no(self):
        return self.picks[self.i][0]

    @property
    def pick(self):
        return self.picks[self.i][1]

    def pack(self):
        return list(self.picks[self.i][3])

    def human(self):
        return self.cards.id_of.get(self.picks[self.i][2])

    def take(self, grp_id):
        self.i += 1


def sampled_drafts(data_dir, set_code):
    """Every sampled 17Lands draft for the set under <data>/Temp/replay/."""
    out = []
    for path in sorted(Path(data_dir, "Temp", "replay").glob(f"{set_code}.*.drafts.json.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            out.extend(json.load(f))
    return out


class MockDraft:
    """Writes one mock draft into `log_path` as Arena would. start() shows the first pack; pick() the next."""

    def __init__(self, cards, source, log_path):
        self.cards, self.source, self.log = cards, source, Path(log_path)
        self.draft_id = str(uuid.uuid4())
        self.taken = []

    @classmethod
    def table(cls, cards, log_path, seed=None):
        seed = seed if seed is not None else random.randrange(1, 10**6)
        return cls(cards, Table(cards, seed), log_path)

    @classmethod
    def real(cls, cards, log_path, draft):
        return cls(cards, RealDraft(cards, draft), log_path)

    def _write(self, line):
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(line)

    def _notify(self):
        if not self.source.done:
            self._write(NOTIFY % (self.draft_id, self.source.pick, self.source.pack_no, ",".join(self.source.pack())))

    def start(self):
        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.log.write_text("MTGA mock log\n", encoding="utf-8")
        self._write(JOIN % (uuid.uuid4(), self.cards.set_code))
        self._notify()

    def pick(self, name):
        """Take a card by name from the current pack. Returns False when the draft is over."""
        if self.source.done:
            return False
        pack = self.source.pack()
        grp = next((i for i in pack if self.cards.name(i) == name), None)
        if grp is None:
            raise ValueError(f"{name} is not in the current pack")
        pack_no, pick = self.source.pack_no, self.source.pick
        self._write(PICK % (uuid.uuid4(), self.draft_id, grp, pack_no, pick))
        self.taken.append(grp)
        self.source.take(grp)
        self._notify()
        return not self.source.done

    def describe(self):
        src = self.source
        return {
            "kind": "table" if isinstance(src, Table) else "17lands",
            "id": src.seed if isinstance(src, Table) else src.id,
            "set": self.cards.set_code,
            "done": src.done,
            "pack_number": None if src.done else src.pack_no,
            "pick_number": None if src.done else src.pick,
            "picks": len(self.taken),
            "human": None if src.done or isinstance(src, Table) else (self.cards.name(src.human()) if src.human() in self.cards.by_id else None),
        }
