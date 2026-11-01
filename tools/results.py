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
            "deck_wr": (round(statistics.mean([r[e]["deck_wr"] for r in rows]), 3)
                        if all(r[e].get("deck_wr") is not None for r in rows) else None),
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


# ── top drafters: agreement and outcome lift (the main measure since the retarget) ──

OUTCOME_BUCKETS = {"lose": (0, 2), "mid": (3, 4), "win": (5, 7)}


def top_drafter_metrics(run_path, drafts_path, boot=500, seed=0):
    from collections import defaultdict

    """One engine's replay of a top-drafter sample, scored against what those drafters took.

    top1/top3: share of picks where the engine's first / first three choices include the card taken.
    by_outcome: top1 split by the drafter's event result (0-2, 3-4, 5-7 wins).
    outcome_lift: top1 in 5-7 win events minus top1 in 0-2 win events, bootstrap SE over drafts.
      Positive: the engine sides with picks that went with winning, away from ones that went with losing.
    weighted_top1: top1 with each pick weighted by its event's wins + 1.
    regret_slope: event wins per +1 SD of the draft's mean per-pick regret (finished events, controls for
      skill and games buckets); more negative means straying from the engine's pick cost wins.
    """
    import gzip

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "mtga_draft_17lands" / "Tools" / "AdvisorReplay"))
    import report as up_report

    with gzip.open(drafts_path, "rt", encoding="utf-8") as f:
        info = {d["id"]: d for d in json.load(f)}
    run = json.loads(Path(run_path).read_text())["drafts"]
    stage = defaultdict(lambda: [0, 0, 0, 0])  # (pack, early|late) -> [hits1, picks, hits1 in 5-7 win events, picks in 5-7 win events]
    rows = []  # per draft: (hits1, hits3, picks, wins, mean regret, finished, draft)
    for d in run:
        meta = info.get(d["id"])
        if not meta or not d["picks"] or (meta.get("skill") or 0) < 0.60 or (meta.get("games") or 0) < 50:
            continue
        for p in d["picks"]:
            for key in (f"P{p[0]}", f"P{p[0]} {'early' if p[1] <= 5 else 'late'}", f"P{p[0]}P{p[1]}"):
                st = stage[key]
                st[0] += p[2] == 0
                st[1] += 1
                if (meta.get("wins") or 0) >= 5:
                    st[2] += p[2] == 0
                    st[3] += 1
        ranks = [p[2] for p in d["picks"]]
        rows.append((sum(r == 0 for r in ranks), sum(r < 3 for r in ranks), len(ranks), meta.get("wins") or 0,
                     float(np.mean([p[3] for p in d["picks"]])), meta.get("wins") == 7 or meta.get("losses") == 3, meta))
    if not rows:
        return {}
    h1 = np.array([r[0] for r in rows], float); h3 = np.array([r[1] for r in rows], float)
    n = np.array([r[2] for r in rows], float); w = np.array([r[3] for r in rows], float)

    def top1(idx):
        return 100 * h1[idx].sum() / n[idx].sum() if n[idx].sum() else float("nan")

    def bucket(name, idx=None):
        idx = np.arange(len(rows)) if idx is None else idx
        lo, hi = OUTCOME_BUCKETS[name]
        return idx[(w[idx] >= lo) & (w[idx] <= hi)]

    def lift(idx):
        return top1(bucket("win", idx)) - top1(bucket("lose", idx))

    rng = np.random.default_rng(seed)
    lifts = [lift(rng.integers(0, len(rows), len(rows))) for _ in range(boot)]
    every = np.arange(len(rows))
    out = {
        "drafts": len(rows), "picks": int(n.sum()),
        "top1": round(top1(every), 2), "top3": round(100 * h3.sum() / n.sum(), 2),
        "top1_se": round(float(np.std([top1(rng.integers(0, len(rows), len(rows))) for _ in range(boot)])), 2),
        "by_outcome": {k: {"drafts": int(len(bucket(k))), "top1": round(top1(bucket(k)), 2)} for k in OUTCOME_BUCKETS},
        "outcome_lift": round(lift(every), 2), "outcome_lift_se": round(float(np.nanstd(lifts)), 2),
        "weighted_top1": round(100 * float((h1 * (w + 1)).sum() / (n * (w + 1)).sum()), 2),
        "mean_regret": round(float(np.average([r[4] for r in rows], weights=n)), 3),
        "by_stage": {k: {"picks": v[1], "top1": round(100 * v[0] / v[1], 2) if v[1] else None,
                         "top1_winners": round(100 * v[2] / v[3], 2) if v[3] else None} for k, v in sorted(stage.items())},
    }
    done = [r for r in rows if r[5]]
    if len(done) > 20:
        ctrl = up_report.controls([r[6] for r in done], ["x"] * len(done))
        out["regret_slope"] = round(float(up_report.wins_slope(np.array([r[4] for r in done]), np.array([r[3] for r in done], float), ctrl)), 3)
    return out


def top_markdown(set_code, split, metrics):
    v, u = metrics["v2"], metrics["upstream_ref"]
    rows = [("top-1 % (SE)", f"{v['top1']} ({v['top1_se']})", f"{u['top1']}"), ("top-3 %", v["top3"], u["top3"]),
            ("top-1 in 5-7 win events", v["by_outcome"]["win"]["top1"], u["by_outcome"]["win"]["top1"]),
            ("top-1 in 3-4 win events", v["by_outcome"]["mid"]["top1"], u["by_outcome"]["mid"]["top1"]),
            ("top-1 in 0-2 win events", v["by_outcome"]["lose"]["top1"], u["by_outcome"]["lose"]["top1"]),
            ("outcome lift (SE)", f"{v['outcome_lift']:+} ({v['outcome_lift_se']})", f"{u['outcome_lift']:+}"),
            ("wins-weighted top-1 %", v["weighted_top1"], u["weighted_top1"]),
            ("mean per-pick regret", v["mean_regret"], u["mean_regret"]),
            ("regret slope (more negative is better)", v.get("regret_slope"), u.get("regret_slope"))]
    stages = [k for k in v.get("by_stage", {}) if " " in k]
    import re as _re

    def pick_key(k):
        m = _re.fullmatch(r"P(\d)P(\d+)", k)
        return (int(m[1]), int(m[2])) if m else None

    positions = sorted((k for k in v.get("by_stage", {}) if pick_key(k)), key=pick_key)
    pick_rows = ["", "By pick: top-1 match % with all top drafters (v2 / upstream ref)", "",
                 "| pick | " + " | ".join(f"P{n}" for n in (1, 2, 3)) + " |", "|---|---|---|---|"]
    for k in range(1, 15):
        cells = []
        for n in (1, 2, 3):
            key = f"P{n}P{k}"
            a, b = v["by_stage"].get(key, {}).get("top1"), u.get("by_stage", {}).get(key, {}).get("top1")
            cells.append(f"{a:.0f} / {b:.0f}" if a is not None and b is not None else "-")
        pick_rows.append(f"| {k} | " + " | ".join(cells) + " |")
    stage_rows = ["", "By pack and stage (picks 1-5 early, 6+ late): top-1 % with all top drafters / with those who went 5-7", "",
                  "| stage | v2 | upstream (ref) |", "|---|---|---|"] + [
        f"| {k} | {v['by_stage'][k]['top1']} / {v['by_stage'][k]['top1_winners']} | {u['by_stage'].get(k, {}).get('top1')} / {u['by_stage'].get(k, {}).get('top1_winners')} |" for k in stages]
    return "\n".join([f"## Top drafters, {set_code} {split} sample ({v['drafts']} drafts, {v['picks']} picks)", "",
                      "Agreement with top drafters (60%+ over 50+ games) and with the ones whose events went well. Upstream is a reference, not the target.", "",
                      "| measure | v2 | upstream (ref) |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows] + stage_rows + pick_rows)
