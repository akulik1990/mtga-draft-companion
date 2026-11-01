"""Ledger of comparison runs, for people and agents reviewing later.

results/runs.jsonl        one JSON object per run, append-only (see results/README.md for the fields)
results/<stamp>-<tool>-<set>.md   the run's full report

Both comparison tools call record() when they finish. Nothing here is needed
by the app.
"""

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from companion import ROOT  # noqa: E402

RESULTS = ROOT / "results"


def engine_config():
    """Every v2 constant (upper-case class attribute) and module switch at run time."""
    from companion.advisor_v2 import composition, lanes, signals, synergy
    from companion.advisor_v2.engine import DraftAdvisorV2 as V

    out = {k: getattr(V, k) for k in dir(V) if k.isupper()}
    out.update(composition=composition.ENABLED, synergy=synergy.ENABLED, pips=getattr(lanes, "USE_PIPS", False),
               signal_scale=getattr(signals, "SCALE", "z"), signal_quality_power=getattr(signals, "QUALITY_POWER", 1))
    return out


def git_commit():
    try:
        out = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "companion", "tools"], capture_output=True, text=True).stdout.strip()
        return out + ("+dirty" if dirty else "")
    except Exception:
        return "unknown"


def record(tool, set_code, params, metrics, report_md, note="", stamp=None, commit=None, config=None):
    """Append one run to the ledger and save its report. Returns the record."""
    RESULTS.mkdir(exist_ok=True)
    when = stamp or datetime.now(timezone.utc)
    name = f"{when.strftime('%Y%m%d-%H%M%S')}-{tool}-{set_code}.md"
    n = 2
    while (RESULTS / name).exists():  # parallel runs can finish in the same second
        name = f"{when.strftime('%Y%m%d-%H%M%S')}-{tool}-{set_code}-{n}.md"
        n += 1
    (RESULTS / name).write_text(report_md.rstrip() + "\n", encoding="utf-8")
    rec = {
        "time": when.isoformat(timespec="seconds"),
        "tool": tool,
        "set": set_code,
        "commit": commit or git_commit(),
        "engine": config or engine_config(),
        "params": params,
        "metrics": metrics,
        "report": name,
        "note": note,
    }
    with open(RESULTS / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def mock_metrics(rows, engines=("upstream", "v2")):
    """Headline numbers from tools/mock_draft.py rows, per engine, plus the paired difference."""
    import statistics

    a, b = engines
    ra = [r[a]["rating"] or 0 for r in rows]
    rb = [r[b]["rating"] or 0 for r in rows]
    diffs = [y - x for x, y in zip(ra, rb)]

    def rate(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    def sig(e, pack, field):
        vals = [c[field] for r in rows for c in r[e]["signals"] if c["pack"] == pack]
        return rate(vals) if field == "top_signal_open" else (round(statistics.mean(vals), 3) if vals else None)

    out = {"tables": len(rows), "diverging_picks": round(statistics.mean(len(r["divergences"]) for r in rows), 2)}
    for e in engines:
        out[e] = {
            "deck_rating": round(statistics.mean([r[e]["rating"] or 0 for r in rows]), 3),
            "lane_score": round(statistics.mean([r[e].get("lane_score") or 0 for r in rows]), 3),
            "lane_matches_deck": {k: rate(r[e]["lane_matches_deck"][k] for r in rows) for k in ("P2P1", "P3P1")},
            "lane_changes": round(statistics.mean(r[e].get("lane_changes", 0) for r in rows), 2),
            "pool": ({k: round(statistics.mean(r[e]["pool"][k] for r in rows), 2) for k in rows[0][e]["pool"]}
                     if all("pool" in r[e] for r in rows) else None),
            "open_colour": {f"P{p}": {"spearman": sig(e, p, "spearman"), "top_open": sig(e, p, "top_signal_open")} for p in (1, 2, 3)},
        }
    out["v2_minus_upstream"] = {
        "deck_rating": round(statistics.mean(diffs), 3),
        "se": round(statistics.stdev(diffs) / len(diffs) ** 0.5, 3) if len(diffs) > 1 else None,
        "better": sum(d > 0 for d in diffs), "worse": sum(d < 0 for d in diffs),
    }
    ld = [(r[b].get("lane_score") or 0) - (r[a].get("lane_score") or 0) for r in rows]
    out["v2_minus_upstream"]["lane_score"] = round(statistics.mean(ld), 3)
    out["v2_minus_upstream"]["lane_score_se"] = round(statistics.stdev(ld) / len(ld) ** 0.5, 3) if len(ld) > 1 else None
    out["v2_minus_upstream"]["lane_better"] = sum(d > 0 for d in ld)
    out["v2_minus_upstream"]["lane_worse"] = sum(d < 0 for d in ld)
    return out


def harness_metrics(report_md):
    """Headline numbers out of upstream's report.py Markdown: top-1/top-3 rows and the regret line."""
    out = {}
    for label in ("base", "head"):
        m = re.search(rf"^\| {label} \| ([\d.]+) / ([\d.]+) \| ([\d.]+) / ([\d.]+) \|", report_md, re.M)
        if m:
            out[label] = {"top1": float(m[1]), "top3": float(m[2]), "strong_top1": float(m[3]), "strong_top3": float(m[4])}
    m = re.search(r"^\| regret[^|]*\| ([-\d.]+) \| ([-\d.]+) \| \*\*([-+\d.]+)\*\* \(([\d.]+)\)", report_md, re.M)
    if m:
        out["regret"] = {"base": float(m[1]), "head": float(m[2]), "diff": float(m[3]), "se": float(m[4])}
    m = re.search(r"Top-1, head minus base: \*\*([-+\d.]+) points\*\* \(paired bootstrap SE ([\d.]+)\)", report_md)
    if m:
        out["top1_diff"] = {"points": float(m[1]), "se": float(m[2])}
    return out


# ── card kinds: where an engine disagrees with good players ─────────────────

KINDS = ("mono", "gold2", "gold3", "hybrid", "dual land", "other land", "colourless")


def card_kind(card):
    """mono / gold2 / gold3 / hybrid / dual land / other land / colourless, from types, colours and pips."""
    from companion.advisor_v2.lanes import pips

    types, colours = card.get("types") or [], card.get("colors") or []
    if "Land" in types:
        return "dual land" if len(colours) >= 2 else "other land"
    if any(len(o) > 1 for o in pips(card)):
        return "hybrid"
    if len(colours) >= 3:
        return "gold3"
    if len(colours) == 2:
        return "gold2"
    return "mono" if colours else "colourless"


def harness_by_kind(base_path, head_path, drafts_path, dataset_path):
    """Strong drafters (60%+ over 50+ games, upstream's definition): top-1 and mean per-pick regret by card kind and by pack.

    Regret is upstream's: (engine's top score − score of the card taken) / SD of the pack's scores.
    """
    import gzip
    import statistics
    from collections import defaultdict

    base = json.loads(Path(base_path).read_text())["drafts"]
    head = {d["id"]: d for d in json.loads(Path(head_path).read_text())["drafts"]}
    with gzip.open(drafts_path, "rt", encoding="utf-8") as f:
        drafts = {d["id"]: d for d in json.load(f)}
    opener = gzip.open if str(dataset_path).endswith(".gz") else open
    with opener(dataset_path, "rt", encoding="utf-8") as f:
        ratings = json.load(f)["card_ratings"]
    by_name = {}
    for c in ratings.values():
        by_name.setdefault(c["name"], c)
        by_name.setdefault(c["name"].split(" // ")[0], c)

    acc = defaultdict(lambda: {"n": 0, "base_top1": 0, "head_top1": 0, "base_regret": 0.0, "head_regret": 0.0})
    for b in base:
        d, h = drafts.get(b["id"]), head.get(b["id"])
        if not d or not h or (d.get("skill") or 0) < 0.6 or (d.get("games") or 0) < 50:
            continue
        taken = {(p[0], p[1]): p[2] for p in d["picks"]}
        hp = {(x[0], x[1]): x for x in h["picks"]}
        for x in b["picks"]:
            y = hp.get((x[0], x[1]))
            if not y:
                continue
            kind = card_kind(by_name.get(taken.get((x[0], x[1]), ""), {}))
            for key in (f"kind:{kind}", f"pack:{x[0]}"):
                a = acc[key]
                a["n"] += 1
                a["base_top1"] += x[2] == 0
                a["head_top1"] += y[2] == 0
                a["base_regret"] += x[3]
                a["head_regret"] += y[3]
    out = {}
    for key, a in acc.items():
        n = a["n"]
        out[key] = {"n": n, "base_top1": round(100 * a["base_top1"] / n, 1), "head_top1": round(100 * a["head_top1"] / n, 1),
                    "base_regret": round(a["base_regret"] / n, 3), "head_regret": round(a["head_regret"] / n, 3)}
    return out


def by_kind_markdown(by_kind):
    lines = ["## Strong drafters by card kind and pack", "",
             "Top-1 % and mean per-pick regret (lower is closer to what the player took) for base (upstream) and head (v2).", "",
             "| slice | picks | top-1 base | top-1 head | regret base | regret head |", "|---|---|---|---|---|---|"]
    order = [f"kind:{k}" for k in KINDS] + [f"pack:{p}" for p in (1, 2, 3)]
    for key in order:
        if key in by_kind:
            v = by_kind[key]
            lines.append(f"| {key.split(':', 1)[1]} | {v['n']} | {v['base_top1']} | {v['head_top1']} | {v['base_regret']} | {v['head_regret']} |")
    return "\n".join(lines)
