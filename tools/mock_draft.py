"""Drive the app's mock drafts from outside: head-to-head comparisons, or one draft to watch.

    # Head-to-head: same seeds, one in-process app per engine, each following its own engine's top pick
    uv run python tools/mock_draft.py compare FRA --seeds 40

    # One draft into a running app, following an engine, slowly enough to watch in the browser
    uv run python tools/mock_draft.py play FRA --app http://127.0.0.1:17173 --follow v2 --delay 2

The app owns the mock (companion/mock.py: tables, bots, replayed 17Lands drafts)
and feeds it to its own scanner as a Player.log; this tool only reads the app's
state and sends picks, over HTTP or in-process through the same DraftSession
the API serves. Mock and real Arena differ only in who writes the log.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import companion  # noqa: E402,F401
from companion import DATA_DIR, LAUNCH_DIR  # noqa: E402

COLOURS = "WUBRG"
ENGINES = ("upstream", "v2")


class InProcess:
    """The app's DraftSession, no HTTP. Same code the API serves."""

    def __init__(self, session):
        self.session = session

    @classmethod
    def fresh(cls, name="mock"):
        from src import constants
        from src.configuration import read_configuration
        from src.limited_sets import LimitedSets
        from src.log_scanner import ArenaScanner
        from companion.session import DraftSession

        log = Path(constants.TEMP_FOLDER) / "mock" / f"{name}.Player.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("MTGA mock log\n", encoding="utf-8")
        config, _ = read_configuration()
        return cls(DraftSession(ArenaScanner(str(log), LimitedSets().retrieve_limited_sets(), retrieve_unknown=True), config))

    def start(self, set_code, kind="table", seed=None):
        return self.session.start_mock(kind, set_code, seed)

    def state(self):
        self.session.poll()
        return self.session.snapshot()

    def pick(self, name):
        return self.session.mock_pick(name)

    def build(self):
        self.session.poll()  # pick up the last pick before building
        return self.session.build_decks()

    def truth(self, pack_no):
        return self.session.mock.source.openness(pack_no)


class Http:
    """A running app over its API."""

    def __init__(self, url):
        import requests

        self.url, self.http = url.rstrip("/"), requests

    def start(self, set_code, kind="table", seed=None):
        r = self.http.post(f"{self.url}/api/mock/start", json={"kind": kind, "set": set_code, "seed": seed}, timeout=60)
        r.raise_for_status()
        return r.json()

    def state(self):
        return self.http.get(f"{self.url}/api/state", timeout=30).json()

    def pick(self, name):
        r = self.http.post(f"{self.url}/api/mock/pick", json={"card_name": name}, timeout=30)
        r.raise_for_status()
        return r.json()

    def build(self):
        return self.http.post(f"{self.url}/api/deck/build", timeout=300).json()

    def truth(self, pack_no):
        return None


def wait_for(app, pack_no, pick, tries=100, delay=0.0):
    """The snapshot once the app shows that pick with both rankings."""
    snap = None
    for _ in range(tries):
        snap = app.state()
        if (snap["pack_number"], snap["pick_number"]) == (pack_no, pick) and snap["pack"] and snap["advisors"].get("v2"):
            return snap
        time.sleep(delay)
    raise RuntimeError(f"app shows P{snap['pack_number']}P{snap['pick_number']} ({len(snap['pack'])} cards), wanted P{pack_no}P{pick}")


def choose(snap, follow, mock):
    if follow == "human" and mock.get("human"):
        return mock["human"]
    if follow == "gihwr":
        return max(snap["pack"], key=lambda c: (c.get("deck_colors") or {}).get("All Decks", {}).get("gihwr") or 0)["name"]
    recs = snap["advisors"].get(follow) or []
    return recs[0]["card_name"] if recs else snap["pack"][0]["name"]


def lane_of(snap, engine):
    if engine == "v2":
        lanes = snap.get("lanes") or {}
        return max(lanes, key=lanes.get) if lanes else ""
    for r in snap["advisors"].get("upstream") or []:
        fit = r.get("archetype_fit") or ""
        if len(fit) == 2 and set(fit) <= set(COLOURS):
            return fit
    return ""


def run_draft(app, set_code, follow, kind="table", seed=None, delay=0.0, quiet=True):
    """One mock draft to the end. Returns (pick log, open-colour checks)."""
    mock = app.start(set_code, kind, seed)
    log, checks, last_snap = [], [], None
    while not mock["done"]:
        pack_no, pick = mock["pack_number"], mock["pick_number"]
        snap = wait_for(app, pack_no, pick, delay=0.1 if delay else 0.0)
        name = choose(snap, follow, mock)
        top = {e: (snap["advisors"].get(e) or [{}])[0] for e in ENGINES}
        log.append({"pack": pack_no, "pick": pick, "took": name, "pack_names": sorted(c["name"] for c in snap["pack"]),
                    "top": {e: {"card": top[e].get("card_name"), "reasons": top[e].get("reasoning", [])} for e in ENGINES},
                    "lane": {e: lane_of(snap, e) for e in ENGINES}})
        if not quiet:
            print(f"P{pack_no}P{pick} took {name}   upstream→{top['upstream'].get('card_name')}   v2→{top['v2'].get('card_name')}", flush=True)
        if delay:
            time.sleep(delay)
        mock = app.pick(name)
        if mock["done"] or mock["pack_number"] != pack_no:
            truth = app.truth(pack_no)
            if truth:
                open_top2 = {c for c, _ in sorted(truth.items(), key=lambda kv: -kv[1])[:2]}
                for e in ENGINES:
                    sig = (snap.get("signals_v2") if e == "v2" else snap.get("signals")) or {}
                    top_sig = max(sig, key=sig.get) if any(sig.values()) else None
                    checks.append({"pack": pack_no, "engine": e, "spearman": round(spearman(sig, truth), 3),
                                   "top_signal_open": (top_sig in open_top2) if top_sig else None})
    return log, checks


def spearman(a, b):
    def ranks(d):
        order = sorted(d, key=d.get)
        r, i = {}, 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and d[order[j + 1]] == d[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    if not a:
        return 0.0
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra.values()) / n, sum(rb.values()) / n
    cov = sum((ra[c] - ma) * (rb[c] - mb) for c in a)
    va = sum((ra[c] - ma) ** 2 for c in a) ** 0.5
    vb = sum((rb[c] - mb) ** 2 for c in a) ** 0.5
    return cov / (va * vb) if va and vb else 0.0


LANE_PAIRS = ["WU", "WB", "WR", "WG", "UB", "UR", "UG", "BR", "BG", "RG"]
DECK_SPELLS = 23


def lane_score(pool):
    """Engine- and builder-independent deck quality: (score, pair).

    For each colour pair, take the pool's spells castable in it by upstream's deck
    builder rule (src.advisor.mana_base.is_castable: pips, hybrid aware),
    value each by its 17Lands GIH WR in that pair (All Decks when 17Lands does not
    publish one), and average the best 23, counting missing slots as 45.0 (a
    filler card). The pool's score is its best pair. Nothing here comes from
    either advisor or from upstream's deck builder.
    """
    from src.advisor.mana_base import is_castable

    best = (0.0, "")
    for pair in LANE_PAIRS:
        rates = []
        for card in pool:
            if "Land" in (card.get("types") or []) or not is_castable(card, list(pair), strict=True):
                continue
            dc = card.get("deck_colors") or {}
            wr = float((dc.get(pair) or {}).get("gihwr") or 0.0) or float((dc.get("All Decks") or {}).get("gihwr") or 0.0)
            if wr > 0:
                rates.append(wr)
        rates = sorted(rates, reverse=True)[:DECK_SPELLS]
        score = (sum(rates) + 45.0 * (DECK_SPELLS - len(rates))) / DECK_SPELLS
        best = max(best, (round(score, 3), pair))
    return best


def pool_composition(pool):
    """What the engine actually drafted: lands, gold cards, creatures, mean GIH WR of spells."""
    from tools.results import card_kind

    kinds = [card_kind(c) for c in pool]
    spells = [c for c in pool if "Land" not in (c.get("types") or [])]
    wrs = [float((c.get("deck_colors") or {}).get("All Decks", {}).get("gihwr") or 0) for c in spells]
    wrs = [w for w in wrs if w > 0]
    return {"lands": sum(k.endswith("land") for k in kinds), "dual_lands": kinds.count("dual land"),
            "gold2": kinds.count("gold2"), "gold3": kinds.count("gold3"), "hybrid": kinds.count("hybrid"),
            "creatures": sum("Creature" in (c.get("types") or []) for c in spells),
            "spell_gihwr": round(statistics.mean(wrs), 2) if wrs else 0.0}


# Upstream's side of a table depends only on the seed, the dataset, upstream's code and the bots,
# so it is played once and reused. Bump UPSTREAM_CACHE_VERSION when the stored row's fields change.
UPSTREAM_CACHE_VERSION = 1
USE_UPSTREAM_CACHE = True


def _sha(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upstream_cache_dir(set_code):
    """<data>/Temp/sim/upstream-cache/<SET>-<key>/, key = dataset + vendored upstream + bots + format version."""
    import hashlib
    import subprocess

    from companion import ROOT

    vendor = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD:vendor/mtga_draft_17lands"], capture_output=True, text=True).stdout.strip()
    parts = [_sha(DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json"), vendor or "no-git",
             _sha(ROOT / "companion" / "mock.py"), str(UPSTREAM_CACHE_VERSION)]
    key = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return DATA_DIR / "Temp" / "sim" / "upstream-cache" / f"{set_code}-{key}"


def compare_table(set_code, seed, make_app, engines=ENGINES):
    row, logs = {"seed": seed}, {}
    cache = upstream_cache_dir(set_code) / f"{seed}.json" if USE_UPSTREAM_CACHE and "upstream" in engines else None
    for e in engines:
        if e == "upstream" and cache and cache.exists():
            saved = json.loads(cache.read_text())
            row[e], logs[e] = saved["row"], saved["log"]
            continue
        app = make_app(e)
        log, checks = run_draft(app, set_code, e, "table", seed)
        builds = app.build()
        best = builds[0] if builds else {}
        logs[e] = log
        final_pool = app.state()["pool"]
        score, pair = lane_score(final_pool)
        row[e] = {"rating": best.get("rating"), "lane_score": score, "lane_score_pair": pair, "deck_colours": "".join(best.get("colors", [])[:2]), "label": best.get("label"),
                  "lane_at": {f"P{p}P1": next(x["lane"][e] for x in log if (x["pack"], x["pick"]) == (p, 1)) for p in (2, 3)},
                  "signals": [c for c in checks if c["engine"] == e], "picks": [x["took"] for x in log],
                  "pool": pool_composition(final_pool)}
        row[e]["lane_matches_deck"] = {k: sorted(v) == sorted(row[e]["deck_colours"]) for k, v in row[e]["lane_at"].items()}
        late = [x["lane"][e] for x in log if (x["pack"] - 1) * 14 + x["pick"] >= 9]
        row[e]["lane_changes"] = sum(1 for p, q in zip(late, late[1:]) if p and q and p != q)
        if e == "upstream" and cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"row": row[e], "log": log}))
    a, b = engines
    row["divergences"] = [
        {"pack": x["pack"], "pick": x["pick"], a: (x["took"], x["top"][a]["reasons"]), b: (y["took"], y["top"][b]["reasons"])}
        for x, y in zip(logs[a], logs[b]) if x["took"] != y["took"] and x["pack_names"] == y["pack_names"]
    ]
    row["first_divergence"] = next(((d["pack"], d["pick"]) for d in row["divergences"]), None)
    return row


def report(rows, engines=ENGINES):
    a, b = engines
    la = [r[a].get("lane_score") or 0 for r in rows]
    lb = [r[b].get("lane_score") or 0 for r in rows]
    ldiff = [y - x for x, y in zip(la, lb)]
    lse = statistics.stdev(ldiff) / len(ldiff) ** 0.5 if len(ldiff) > 1 else 0.0
    ra = [r[a]["rating"] or 0 for r in rows]
    rb = [r[b]["rating"] or 0 for r in rows]
    diffs = [y - x for x, y in zip(ra, rb)]
    wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
    se = statistics.stdev(diffs) / len(diffs) ** 0.5 if len(diffs) > 1 else 0.0

    def pct(vals):
        vals = [v for v in vals if v is not None]
        return f"{100 * sum(vals) / len(vals):.0f}%" if vals else "-"

    def sig(e, pack, field):
        vals = [c[field] for r in rows for c in r[e]["signals"] if c["pack"] == pack]
        return pct(vals) if field == "top_signal_open" else (f"{statistics.mean(vals):+.2f}" if vals else "-")

    n_div = [len(r["divergences"]) for r in rows]
    lines = [
        f"## Mock drafts through the app: {len(rows)} tables, {a} vs {b}",
        "",
        "### Final decks",
        "",
        "Lane score is independent of both engines and both builders: mean 17Lands GIH WR of the best 23 spells in the pool's best colour pair, using each card's win rate in that pair. Rating is upstream's deck builder and scorer, kept for comparison.",
        "",
        "| engine | mean lane score | mean upstream rating |", "|---|---|---|",
        f"| {a} | {statistics.mean(la):.2f} | {statistics.mean(ra):.2f} |", f"| {b} | {statistics.mean(lb):.2f} | {statistics.mean(rb):.2f} |",
        "",
        f"Lane score, {b} minus {a}: **{statistics.mean(ldiff):+.3f}** (SE {lse:.3f}); {b} better in {sum(d > 0 for d in ldiff)}, worse in {sum(d < 0 for d in ldiff)}.",
        "",
        f"Upstream rating, {b} minus {a}: **{statistics.mean(diffs):+.2f}** (SE {se:.2f}); {b} better in {wins}, worse in {losses}, equal in {len(rows) - wins - losses}.",
        "",
        "### Picks", "",
        f"Mean picks that differ per table: {statistics.mean(n_div):.1f} of 42. First divergence by pack: "
        + ", ".join(f"P{p} {sum(1 for r in rows if r['first_divergence'] and r['first_divergence'][0] == p)}" for p in (1, 2, 3))
        + f", never {sum(1 for r in rows if not r['first_divergence'])}.",
        "",
        "### Colour suggestion (engine's lane at the start of a pack equals the colours of its own final build; lane changes from pick 9 on)", "",
        "| engine | at P2P1 | at P3P1 | lane changes per draft |", "|---|---|---|---|",
        f"| {a} | {pct(r[a]['lane_matches_deck']['P2P1'] for r in rows)} | {pct(r[a]['lane_matches_deck']['P3P1'] for r in rows)} | {statistics.mean(r[a].get('lane_changes', 0) for r in rows):.1f} |",
        f"| {b} | {pct(r[b]['lane_matches_deck']['P2P1'] for r in rows)} | {pct(r[b]['lane_matches_deck']['P3P1'] for r in rows)} | {statistics.mean(r[b].get('lane_changes', 0) for r in rows):.1f} |",
        "",
        "### Open-colour detection (vs what the passing seat took: Spearman over WUBRG; top signal colour among the two most open)", "",
        "| engine | after P1 ρ | after P1 top | after P2 ρ | after P2 top | after P3 ρ | after P3 top |", "|---|---|---|---|---|---|---|",
    ]
    for e in engines:
        lines.append(f"| {e} | " + " | ".join(f"{sig(e, p, 'spearman')} | {sig(e, p, 'top_signal_open')}" for p in (1, 2, 3)) + " |")
    comp_keys = ["lands", "dual_lands", "gold2", "gold3", "hybrid", "creatures", "spell_gihwr"]
    if all("pool" in r[a] for r in rows):
        lines += ["", "### What each engine drafted (mean per 42-pick pool)", "",
                  "| engine | " + " | ".join(comp_keys) + " |", "|---|" + "---|" * len(comp_keys)]
        for e in engines:
            lines.append(f"| {e} | " + " | ".join(f"{statistics.mean(r[e]['pool'][k] for r in rows):.2f}" for k in comp_keys) + " |")
    lines += ["", "### Tables where the engines ended furthest apart", ""]
    for r in sorted(rows, key=lambda r: -abs((r[b]["rating"] or 0) - (r[a]["rating"] or 0)))[:8]:
        d = (r[b]["rating"] or 0) - (r[a]["rating"] or 0)
        fd = r["first_divergence"]
        lines.append(f"- seed {r['seed']}: {d:+.1f}, first diff at P{fd[0]}P{fd[1]}, {a} {r[a]['deck_colours']} vs {b} {r[b]['deck_colours']}" if fd else f"- seed {r['seed']}: identical drafts")
        for div in r["divergences"][:3]:
            lines.append(f"    - P{div['pack']}P{div['pick']}: {a} took {div[a][0]} ({'; '.join(div[a][1][:2])}) | {b} took {div[b][0]} ({'; '.join(div[b][1][:2])})")
    return "\n".join(lines)


def apply_overrides(pairs):
    """--v2 NAME=VALUE sets a DraftAdvisorV2 constant; --v2 module.NAME=VALUE a switch in companion.advisor_v2.<module>.

    The ledger records every constant and switch, so the override is visible there.
    """
    import importlib

    from companion.advisor_v2.engine import DraftAdvisorV2

    for pair in pairs:
        name, value = pair.split("=", 1)
        target = DraftAdvisorV2
        if "." in name:
            module, name = name.split(".", 1)
            target = importlib.import_module(f"companion.advisor_v2.{module}")
        if not hasattr(target, name):
            raise SystemExit(f"{target.__name__} has no {name}")
        current = getattr(target, name)
        setattr(target, name, value.lower() in ("1", "true", "on") if isinstance(current, bool) else type(current)(value))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare"); c.add_argument("set"); c.add_argument("--seeds", type=int, default=200); c.add_argument("--start", type=int, default=1)
    c.add_argument("--out", help="results JSON (default <data>/Temp/sim/<SET>.json)"); c.add_argument("--note", default="", help="what this run is testing; kept in results/runs.jsonl")
    c.add_argument("--no-cache", action="store_true", help="replay upstream's side instead of reusing its cached tables")
    c.add_argument("--v2", action="append", default=[], metavar="NAME=VALUE", help="override a DraftAdvisorV2 constant for this run, e.g. --v2 SPLASH=0; recorded in the ledger")
    g = sub.add_parser("play"); g.add_argument("set"); g.add_argument("--app", help="base URL of a running app; else in-process")
    g.add_argument("--follow", default="v2", choices=("upstream", "v2", "gihwr", "human")); g.add_argument("--kind", default="table", choices=("table", "17lands", "random"))
    g.add_argument("--seed", type=int); g.add_argument("--delay", type=float, default=0.0, help="seconds between picks, to watch in the browser")
    a = p.parse_args(argv)

    if a.cmd == "compare":
        global USE_UPSTREAM_CACHE
        USE_UPSTREAM_CACHE = not a.no_cache
        apply_overrides(a.v2)
        rows = []
        for seed in range(a.start, a.start + a.seeds):
            row = compare_table(a.set, seed, lambda e: InProcess.fresh(e))
            rows.append(row)
            print(f"seed {seed}: " + "  ".join(f"{e} {row[e]['rating'] or 0:.1f} {row[e]['deck_colours']}" for e in ENGINES) + f"  first diff {row['first_divergence']}", flush=True)
        out = (LAUNCH_DIR / a.out) if a.out else DATA_DIR / "Temp" / "sim" / f"{a.set}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=1))
        md = report(rows)
        from tools import results

        rec = results.record("mock", a.set, {"seeds": a.seeds, "start": a.start}, results.mock_metrics(rows), md, a.note)
        print("\n" + md + f"\n\nwrote {out}; recorded in results/runs.jsonl as {rec['report']}")
    else:
        app = Http(a.app) if a.app else InProcess.fresh("play")
        log, _ = run_draft(app, a.set, a.follow, a.kind, a.seed, delay=a.delay, quiet=False)
        builds = app.build()
        best = builds[0] if builds else {}
        print(f"\n{len(log)} picks; best build {best.get('label')} rating {best.get('rating')}")


if __name__ == "__main__":
    main()
