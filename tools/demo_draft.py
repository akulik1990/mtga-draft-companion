"""Write a synthetic Premier Draft Player.log from a real dataset, for UI work without Arena.

Simulates an 8-seat table with packs drawn by rarity (10C/3U/1R-or-M). Bots take the
highest All Decks GIH WR card; our seat (0) takes the highest too unless --random-picks.
Packs pass left in packs 1 and 3, right in pack 2, so signals are realistic.

    uv run python tools/demo_draft.py data/Sets/FRA_PremierDraft_All_Data.json \\
        --out data/demo/Player.log --until 2:5
    MTGA_COMPANION_DATA=$PWD/data uv run python -m companion --no-sync --log data/demo/Player.log --browser

--until P:K stops after the pack for pack P pick K is shown (that pick not yet made).
--with-p1p1 emits a Draft.Notify for P1P1, which real Premier logs do not.
"""

import argparse
import json
import random
import uuid
from pathlib import Path

SEATS, PACK_SIZE, PACKS = 8, 14, 3


def load_pool(path):
    data = json.loads(Path(path).read_text())
    by_rarity = {"common": [], "uncommon": [], "rare": [], "mythic": []}
    gih = {}
    for grp_id, card in data["card_ratings"].items():
        stats = (card.get("deck_colors") or {}).get("All Decks") or {}
        if not stats.get("game_count"):
            continue  # no 17Lands games: basics (all printings) and cards not in this format's boosters
        if card.get("rarity") in by_rarity:
            by_rarity[card["rarity"]].append(grp_id)
            gih[grp_id] = stats.get("gihwr", 0.0)
    return by_rarity, gih


def booster(rng, by_rarity):
    rare = by_rarity["mythic"] if by_rarity["mythic"] and rng.random() < 1 / 8 else by_rarity["rare"]
    return rng.sample(by_rarity["common"], 10) + rng.sample(by_rarity["uncommon"], 3) + [rng.choice(rare)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--out", default="data/demo/Player.log")
    p.add_argument("--until", default="3:14", help="P:K, stop with this pack shown")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--random-picks", action="store_true", help="our seat picks at random")
    p.add_argument("--with-p1p1", action="store_true")
    args = p.parse_args()

    set_code = Path(args.dataset).name.split("_")[0]
    stop_pack, stop_pick = map(int, args.until.split(":"))
    rng = random.Random(args.seed)
    by_rarity, gih = load_pool(args.dataset)
    draft_id = str(uuid.UUID(int=rng.getrandbits(128)))

    lines = [
        "MTGA demo log\n",
        '[UnityCrossThreadLogger]==> Event_Join {"id":"%s","request":"{\\"EventName\\":\\"PremierDraft_%s_20260101\\"}"}\n'
        % (uuid.uuid4(), set_code),
    ]
    for pack_no in range(1, PACKS + 1):
        packs = [booster(rng, by_rarity) for _ in range(SEATS)]
        direction = -1 if pack_no == 2 else 1
        for pick in range(1, PACK_SIZE + 1):
            # Seat s holds the pack that started at seat (s - direction*(pick-1)).
            holding = [packs[(s - direction * (pick - 1)) % SEATS] for s in range(SEATS)]
            mine = holding[0]
            if (pack_no, pick) != (1, 1) or args.with_p1p1:
                lines.append(
                    '[UnityCrossThreadLogger]Draft.Notify {"draftId":"%s","SelfPick":%d,"SelfPack":%d,"PackCards":"%s"}\n'
                    % (draft_id, pick, pack_no, ",".join(mine))
                )
            if (pack_no, pick) == (stop_pack, stop_pick):
                break
            for s, pk in enumerate(holding):
                choice = rng.choice(pk) if (s == 0 and args.random_picks) else max(pk, key=lambda c: gih.get(c, 0))
                pk.remove(choice)
                if s == 0:
                    lines.append(
                        '[UnityCrossThreadLogger]==> Event_PlayerDraftMakePick {"id":"%s","request":"{\\"DraftId\\":\\"%s\\",\\"GrpIds\\":[%s],\\"Pack\\":%d,\\"Pick\\":%d}"}\n'
                        % (uuid.uuid4(), draft_id, choice, pack_no, pick)
                    )
        else:
            continue
        break

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines), stopped at P{stop_pack}P{stop_pick}")


if __name__ == "__main__":
    main()
