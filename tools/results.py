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
