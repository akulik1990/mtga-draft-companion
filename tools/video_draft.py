"""Strong players' recorded drafts (docs/video-drafts.md): frames, transcript, replay through v2.

    uv run python tools/video_draft.py frames <id> tiles --start 320 --end 1000 --crop 300:56:850:100 [--every 2]
    uv run python tools/video_draft.py frames <id> grab 76 134 158 ...       # full 1920x1080 frames at those seconds
    uv run python tools/video_draft.py transcript <id> [--start 0 --end 1000]
    uv run python tools/video_draft.py replay <id> | --all

Raw video, subtitles and frames live in data/videos/raw/ (gitignored); the record is analysis/videos/<id>/picks.json and deck.json.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "videos" / "raw"
OUT = REPO / "analysis" / "videos"


def ffmpeg(*args):
    subprocess.run(["nice", "-n", "19", "nix", "shell", "nixpkgs#ffmpeg", "-c", "ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def cmd_frames(a):
    video, work = RAW / f"{a.id}.mp4", RAW / a.id
    if a.what == "tiles":
        # Banner crops with the video time burnt in, 5 x 20 per sheet: where "Pack N | Pick M" changes is where a pick ends.
        (work / "tiles").mkdir(parents=True, exist_ok=True)
        ffmpeg("-ss", str(a.start), "-to", str(a.end), "-i", str(video), "-vf",
               f"fps=1/{a.every},crop={a.crop},drawtext=text='%{{pts\\:hms}}':x=2:y=2:fontsize=16:fontcolor=yellow:box=1:boxcolor=black,tile=5x20",
               str(work / "tiles" / f"{a.prefix}_%02d.jpg"))
        print(f"tiles -> {work / 'tiles'} (pts is offset by --start {a.start})")
    else:
        # Full frames, or with --crop W:H:X:Y only the pack (layout b puts it in the top half; native pixels keep small names readable).
        (work / "frames").mkdir(parents=True, exist_ok=True)
        tag = "c" if a.grab_crop else "t"
        for t in a.times:
            vf = ["-vf", f"crop={a.grab_crop}"] if a.grab_crop else []
            ffmpeg("-ss", t, "-i", str(video), "-frames:v", "1", *vf, "-q:v", "2", str(work / "frames" / f"{tag}{float(t):07.1f}.jpg"))
        print(f"{len(a.times)} frames -> {work / 'frames'}")


def seconds(ts):
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def transcript(vid):
    """[(start seconds, text)] from YouTube auto captions: each cue repeats the previous line, so keep only the line with word timings."""
    lines, start = [], None
    for raw in (RAW / f"{vid}.en.vtt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"(\d+:\d+:\d+\.\d+) -->", raw)
        if m:
            start = seconds(m.group(1))
        elif "<c>" in raw and start is not None:
            text = re.sub(r"<[^>]+>", "", raw).strip()
            if text and (not lines or lines[-1][1] != text):
                lines.append((start, text))
    return lines


def cmd_transcript(a):
    for t, text in transcript(a.id):
        if a.start <= t <= a.end:
            print(f"{int(t // 60)}:{t % 60:04.1f} {text}")


# ── replay ───────────────────────────────────────────────────────────────────

def load_set(dataset_name):
    sys.path.insert(0, str(REPO))
    import companion  # noqa: F401  (vendor on sys.path)
    from src.dataset import Dataset
    from src.set_metrics import SetMetrics

    ds = Dataset()
    ds.open_file(str(REPO / "data" / "Sets" / dataset_name))
    full = {}
    for c in ds.get_card_ratings().values():
        full.setdefault(c["name"].split(" // ")[0], c["name"])

    def cards(names):
        out = [ds.get_data_by_name([full.get(n, n)]) for n in names]
        missing = [n for n, c in zip(names, out) if not c]
        if missing:
            raise SystemExit(f"not in {dataset_name}: {missing}")
        return [c[0] for c in out]

    return SetMetrics(ds), cards


def lanes_text(lanes, k=3):
    return ", ".join(f"{l} {p:.0%}" for l, p in sorted(lanes.items(), key=lambda kv: -kv[1])[:k])


def replay(vid):
    from companion.advisor_v2.engine import DraftAdvisorV2

    rec = json.loads((OUT / vid / "picks.json").read_text(encoding="utf-8"))
    metrics, cards = load_set(rec.get("dataset", f"{rec['set']}_PremierDraft_All_Data.json"))
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    pool, seen, rows = [], [], []
    for p in rec["picks"]:
        pack = cards(p["pack_cards"])
        # Packs seen so far including this one, as the replay harness (tools/replay/run.py) and the live session pass them.
        seen.append((p["pack"], p["pick"], pack))
        adv = DraftAdvisorV2(metrics, list(pool), seen_packs=list(seen))
        recs = adv.evaluate_pack(pack, p["pick"], current_pack=p["pack"])
        scores = {}
        for r in recs:
            scores.setdefault(r.card_name, r)
        order = sorted(scores.values(), key=lambda r: -r.contextual_score)
        names = [r.card_name for r in order]
        his = scores.get(p["taken"])
        pool.append(cards([p["taken"]])[0])
        after = DraftAdvisorV2(metrics, list(pool), seen_packs=list(seen))
        rows.append({"p": p, "order": order, "his": his, "rank": names.index(p["taken"]) + 1 if p["taken"] in names else None,
                     "lanes_before": adv.lanes, "lanes_after": after.lanes})
    report = render(rec, rows, commit, pool)
    out = OUT / vid / f"replay-{commit}.md"
    out.write_text(report, encoding="utf-8")
    write_engine(vid, rec, rows, commit, pool)
    print(f"{vid}: top-1 {sum(r['rank'] == 1 for r in rows)}/{len(rows)}, top-3 {sum((r['rank'] or 99) <= 3 for r in rows)}/{len(rows)} -> {out}")


def write_engine(vid, rec, rows, commit, pool):
    """engine/<commit>.json and engine/latest.json, read by the /videos page (companion/api/videos.py)."""
    import datetime

    def deck(d):
        return {"main": d.get("main"), "splash": d.get("splash"), "deck_wr": d["deck_wr"], "label": d.get("label"),
                "cards": [{"name": c["name"], "count": c["count"]} for c in d["deck"]]}

    picks = []
    for r in rows:
        p, o, h, la = r["p"], r["order"], r["his"], r["lanes_after"]
        picks.append({"pack": p["pack"], "pick": p["pick"], "taken": p["taken"],
                      "top3": [{"name": x.card_name, "score": round(x.contextual_score, 2)} for x in o[:3]],
                      "his_score": h and round(h.contextual_score, 2), "his_rank": r["rank"],
                      "reasons_top": list(o[0].reasoning) if o else [], "reasons_his": list(h.reasoning) if h else [],
                      "lanes_after": {k: round(v, 3) for k, v in sorted(la.items(), key=lambda kv: -kv[1])[:4]}})
    try:
        deck_rec = json.loads((OUT / vid / "deck.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        deck_rec = None
    ours, his = builds(pool, deck_rec)
    data = {"commit": commit, "date": datetime.date.today().isoformat(), "dataset": rec.get("dataset"),
            "top1": sum(r["rank"] == 1 for r in rows), "top3": sum((r["rank"] or 99) <= 3 for r in rows), "n": len(rows),
            "picks": picks, "builder": [deck({**b, "deck_wr": b["model_wr"]}) for b in ours], "his_deck": his and deck(his)}
    d = OUT / vid / "engine"
    d.mkdir(exist_ok=True)
    for name in (commit, "latest"):
        (d / f"{name}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")


def builds(pool, deck_rec):
    """Our builder's best decks from his pool, and the deck model's predicted WR for his own 40; None if the builder is off."""
    from companion import session

    ours = session.own_builds("FRA", pool) or []
    for b in ours:
        b["model_wr"] = session.deck_win_rate("FRA", b)  # the mock's yardstick (tools/mock_draft.py deck_wr); b["deck_wr"] is the builder's own objective
    his = None
    if deck_rec:
        names = {**deck_rec["main"], **deck_rec["lands"]}
        by = {c["name"]: c for c in pool}
        deck = []
        for n, k in names.items():
            c = by.get(n) or {"name": n, "types": ["Basic", "Land"], "colors": [], "mana_cost": ""}
            deck.append({**session._trim(c), "count": k})
        his = {"deck": deck, **session.mana_identity(deck)}
        his["deck_wr"] = session.deck_win_rate("FRA", his)
    return ours, his


def fmt_deck(deck):
    spells = [c for c in deck if "Land" not in (c.get("types") or [])]
    lands = [c for c in deck if "Land" in (c.get("types") or [])]
    f = lambda cs: ", ".join(f"{c['count']} {c['name']}" if c["count"] > 1 else c["name"] for c in cs)
    return f"{sum(c['count'] for c in spells)} spells: {f(spells)}. {sum(c['count'] for c in lands)} lands: {f(lands)}."


def render(rec, rows, commit, pool):
    L = [f"# Replay of {rec['player']}, \"{rec['title']}\" ({rec['url']}) at {commit}", "",
         f"Engine: DraftAdvisorV2 at {commit}, dataset {rec.get('dataset')}. His pool rebuilt pick by pick; seen_packs = every pack so far including the current one.", ""]
    L += ["## Agreement", "", "| Pack | Picks | v2 top-1 | v2 top-3 |", "|---|---|---|---|"]
    for pk in (1, 2, 3, None):
        rs = [r for r in rows if pk is None or r["p"]["pack"] == pk]
        n, t1, t3 = len(rs), sum(r["rank"] == 1 for r in rs), sum((r["rank"] or 99) <= 3 for r in rs)
        L.append(f"| {pk or 'all'} | {n} | {t1} ({t1 / n:.0%}) | {t3} ({t3 / n:.0%}) |")
    forced = [r for r in rows if len(r["p"]["pack_cards"]) <= 2]
    rs = [r for r in rows if len(r["p"]["pack_cards"]) > 2]
    L += ["", f"Without the {len(forced)} picks from packs of one or two cards: top-1 {sum(r['rank'] == 1 for r in rs)}/{len(rs)}, "
              f"top-3 {sum((r['rank'] or 99) <= 3 for r in rs)}/{len(rs)}.", ""]

    L += ["## Colour path", "",
          "His main colours by pick (from picks.json `plan` and his picks) next to v2's lane distribution after the pick is added to the pool "
          "(DraftAdvisorV2(...).lanes, top three).", "",
          "| Pick | He took | His plan | v2 top lane | v2 lanes after the pick |", "|---|---|---|---|---|"]
    for r in rows:
        p, la = r["p"], r["lanes_after"]
        L.append(f"| P{p['pack']}p{p['pick']} | {p['taken']} | {p['plan'] or '-'} | {max(la, key=la.get)} | {lanes_text(la)} |")
    gw = [(f"P{r['p']['pack']}p{r['p']['pick']}", sum(v for l, v in r["lanes_after"].items() if "G" in l)) for r in rows]
    L += ["", "Share of v2's lane mass on lanes containing green, after each pick: " + ", ".join(f"{k} {v:.0%}" for k, v in gw) + ".", ""]

    L += ["## Picks", "", "| Pick | His pick | v2 #1 | v2 #2 | v2 #3 | His rank | v2 score for his | Gap | v2 first reason (#1) | His reasoning |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p, o, h = r["p"], r["order"], r["his"]
        top = [f"{x.card_name} {x.contextual_score:.2f}" for x in o[:3]] + ["-"] * 3
        gap = f"{o[0].contextual_score - h.contextual_score:.2f}" if h else "-"
        reason = (o[0].reasoning[0] if o and o[0].reasoning else "").replace("|", "/")
        L.append(f"| P{p['pack']}p{p['pick']} | {p['taken']} | {top[0]} | {top[1]} | {top[2]} | {r['rank'] or '-'} "
                 f"| {h.contextual_score:.2f} | {gap} | {reason} | {p['reasoning'].replace('|', '/')} |" if h else
                 f"| P{p['pack']}p{p['pick']} | {p['taken']} | {top[0]} | {top[1]} | {top[2]} | - | - | - | {reason} | {p['reasoning']} |")

    L += ["", "## Deck", ""]
    try:
        deck_rec = json.loads((OUT / rec["id"] / "deck.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        deck_rec = None
    ours, his = builds(pool, deck_rec)
    if his:
        L.append(f"His 40 ({his['main']}" + (f" + {his['splash']}" if his["splash"] else "") + f"), deck model WR {his['deck_wr']} (session.deck_win_rate, the mock's deck_wr yardstick): {fmt_deck(his['deck'])}")
    if not ours:
        L.append("\nOur builder returned nothing (OWN_BUILDER off, or no deck model / targets for FRA).")
    for b in ours:
        L += ["", f"Builder: {b['label']}, deck model WR {b['model_wr']} (builder objective {b['deck_wr']}), {b['creatures']} creatures. {fmt_deck(b['deck'])}"]
        if his:
            mine, theirs = ({c["name"]: c["count"] for c in d if "Basic" not in (c.get("types") or [])} for d in (b["deck"], his["deck"]))
            only_b = sorted(n for n in mine if n not in theirs)
            only_h = sorted(n for n in theirs if n not in mine)
            L.append(f"Only in the builder's: {', '.join(only_b) or '-'}. Only in his: {', '.join(only_h) or '-'}.")
    return "\n".join(L) + "\n"


def cmd_replay(a):
    ids = [v["id"] for v in json.loads((OUT / "index.json").read_text(encoding="utf-8"))] if a.all else [a.id]
    for vid in ids:
        replay(vid)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames")
    f.add_argument("id")
    f.add_argument("what", choices=["tiles", "grab"])
    f.add_argument("times", nargs="*")
    f.add_argument("--start", type=float, default=0)
    f.add_argument("--end", type=float, default=1000)
    f.add_argument("--crop", default="300:56:850:100")
    f.add_argument("--every", type=float, default=2)
    f.add_argument("--prefix", default="b")
    f.add_argument("--grab-crop", default="", help="grab only: crop W:H:X:Y, e.g. 1300:420:320:165 for the layout-b pack")
    t = sub.add_parser("transcript")
    t.add_argument("id")
    t.add_argument("--start", type=float, default=0)
    t.add_argument("--end", type=float, default=1e9)
    r = sub.add_parser("replay")
    r.add_argument("id", nargs="?")
    r.add_argument("--all", action="store_true")
    a = p.parse_args()
    {"frames": cmd_frames, "transcript": cmd_transcript, "replay": cmd_replay}[a.cmd](a)


if __name__ == "__main__":
    main()
