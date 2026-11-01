"""Turn a real 17Lands draft (from the replay sampler) into a Premier Player.log.

    uv run python tools/replay/run.py sample FRA --drafts 50          # writes <data>/Temp/replay/FRA.50.drafts.json.gz
    uv run python tools/draft_to_log.py <data>/Temp/replay/FRA.50.drafts.json.gz \\
        --dataset data/Sets/FRA_PremierDraft_All_Data.json --index 0 --out data/demo/Player.log --until 2:5

Unlike tools/demo_draft.py, the packs and picks are a human's, so signals and
advisor output are realistic. Stream the result with tools/stream_log.py to watch
the UI follow it pick by pick. Stdlib only.
"""

import argparse
import gzip
import json
import uuid
from pathlib import Path

JOIN = '[UnityCrossThreadLogger]==> Event_Join {"id":"%s","request":"{\\"EventName\\":\\"PremierDraft_%s_20260101\\"}"}\n'
NOTIFY = '[UnityCrossThreadLogger]Draft.Notify {"draftId":"%s","SelfPick":%d,"SelfPack":%d,"PackCards":"%s"}\n'
PICK = (
    '[UnityCrossThreadLogger]==> Event_PlayerDraftMakePick {"id":"%s","request":"{\\"DraftId\\":\\"%s\\",'
    '\\"GrpIds\\":[%s],\\"Pack\\":%d,\\"Pick\\":%d}"}\n'
)


def name_to_id(dataset_path):
    """Dump names are front faces only; prefer the first printing of each name."""
    data = json.loads(Path(dataset_path).read_text())
    ids = {}
    for grp_id, card in data["card_ratings"].items():
        name = card.get("name", "")
        ids.setdefault(name, grp_id)
        ids.setdefault(name.split(" // ")[0], grp_id)
    return ids


def to_log_lines(draft, ids, set_code, until=None, with_p1p1=False):
    """Log lines for one sampled draft; stops with pack `until` shown and unpicked."""
    draft_id = str(uuid.uuid5(uuid.NAMESPACE_URL, draft["id"]))
    lines = ["MTGA log from 17Lands draft %s\n" % draft["id"], JOIN % (uuid.uuid4(), set_code)]
    for pack_no, pick_no, taken, pack_names, _pool in draft["picks"]:
        pack_ids = [ids[n] for n in pack_names if n in ids]
        if (pack_no, pick_no) != (1, 1) or with_p1p1:
            lines.append(NOTIFY % (draft_id, pick_no, pack_no, ",".join(pack_ids)))
        if until and (pack_no, pick_no) == until:
            break
        if taken in ids:
            lines.append(PICK % (uuid.uuid4(), draft_id, ids[taken], pack_no, pick_no))
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("drafts", help="drafts.json.gz from the replay sampler")
    p.add_argument("--dataset", required=True, help="the set's *_All_Data.json, for card IDs")
    p.add_argument("--index", type=int, default=0, help="which sampled draft")
    p.add_argument("--id", help="draft_id instead of --index")
    p.add_argument("--set", help="set code for Event_Join (default: from the dataset file name)")
    p.add_argument("--out", default="data/demo/Player.log")
    p.add_argument("--until", help="P:K, stop with this pack shown")
    p.add_argument("--with-p1p1", action="store_true")
    args = p.parse_args()

    with gzip.open(args.drafts, "rt", encoding="utf-8") as f:
        drafts = json.load(f)
    draft = next(d for d in drafts if d["id"] == args.id) if args.id else drafts[args.index]
    set_code = args.set or Path(args.dataset).name.split("_")[0]
    until = tuple(map(int, args.until.split(":"))) if args.until else None
    lines = to_log_lines(draft, name_to_id(args.dataset), set_code, until, args.with_p1p1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(lines)} lines) from draft {draft['id']} ({draft['wins']}-{draft['losses']}, skill {draft['skill']})")


if __name__ == "__main__":
    main()
