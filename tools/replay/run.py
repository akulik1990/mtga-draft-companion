"""Drive upstream's Tools/AdvisorReplay with a choice of advisor.

    uv run python tools/replay/run.py compare FRA --drafts 300 [--salt x]
    uv run python tools/replay/run.py sample FRA --drafts 300 [--salt x]
    uv run python tools/replay/run.py replay --advisor v2 --drafts D.json.gz --dataset S.json --label head --out r.json
    uv run python tools/replay/run.py report results/*.json

    uv run python tools/replay/run.py sample-top FRA --n 600          # top drafters only, disjoint train/test samples
    uv run python tools/replay/run.py top FRA --split train [--v2 K=500] [--note ...]   # v2 only, ~30 s

`top` is the main measure since the retarget (CLAUDE.md, Validation): v2's agreement with top
drafters (60%+ over 50+ games) and its outcome lift, on a fixed top-drafter sample, with
upstream's numbers as a cached reference. `compare` is the older random-sample comparison: it samples drafts (downloading the
17Lands dump once into <data>/Temp/replay/), replays them through upstream's
advisor as "base" and ours as "head", and prints upstream's report. Vendor code
is used as-is; only the advisor class is swapped.
"""

import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import requests  # upstream dependency; carries certifi, which the uv-managed Python needs on NixOS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # run by path, not only as -m
import companion  # noqa: E402,F401  (vendor on sys.path, cwd = data dir)
from companion import DATA_DIR, LAUNCH_DIR, VENDOR  # noqa: E402

sys.path.insert(0, str(VENDOR / "Tools" / "AdvisorReplay"))
import replay  # noqa: E402
import report  # noqa: E402
import sample  # noqa: E402

CACHE = DATA_DIR / "Temp" / "replay"
DUMP_URL = "https://17lands-public.s3.amazonaws.com/analysis_data/draft_data/draft_data_public.{set}.PremierDraft.csv.gz"
DATASET_URL = "https://unrealities.github.io/MTGA_Draft_17Lands/{set}_PremierDraft_All_Data.json.gz"


def advisor_class(name):
    if name == "upstream":
        from src.advisor.engine import DraftAdvisor

        return DraftAdvisor
    from companion.advisor_v2.engine import DraftAdvisorV2

    return DraftAdvisorV2


_original = {"init": replay.App.__init__, "advise": replay.App.advise, "replay_draft": replay.replay_draft}


def use_advisor(name):
    """Make replay.App construct the chosen advisor. Everything else is upstream's.

    Upstream's harness hands the advisor only the pool and the current pack. v2
    also wants the packs seen earlier in the draft, so for v2 the draft being
    replayed is remembered and its earlier picks' packs are passed as seen_packs.
    """
    cls = advisor_class(name)

    def __init__(self, *a, **k):
        _original["init"](self, *a, **k)
        self.DraftAdvisor = cls
        self.current_draft = None

    def replay_draft(app, draft, skipped):
        app.current_draft = draft
        return _original["replay_draft"](app, draft, skipped)

    def advise(self, pack, pool, pack_no, pick_no):
        if name != "v2":
            return _original["advise"](self, pack, pool, pack_no, pick_no)
        seen = [
            (p, k, self.cards(names))
            for p, k, _taken, names, _pool in (self.current_draft or {}).get("picks", [])
            if (p, k) <= (pack_no, pick_no)
        ]
        recs = cls(self.metrics, pool, seen_packs=seen).evaluate_pack(pack, pick_no, current_pack=pack_no)
        scores = {}
        for r in recs:
            scores.setdefault(r.card_name, r.contextual_score)
        return scores

    replay.App.__init__ = __init__
    replay.App.advise = advise
    replay.replay_draft = replay_draft


def fetch(url, dest):
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {url}")
        with requests.get(url, stream=True, timeout=60) as r, open(dest, "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


def dataset_path(set_code):
    local = DATA_DIR / "Sets" / f"{set_code}_PremierDraft_All_Data.json"
    return local if local.exists() else fetch(DATASET_URL.format(set=set_code), CACHE / local.name.replace(".json", ".json.gz"))


def cmd_sample(a):
    dump = fetch(DUMP_URL.format(set=a.set), CACHE / f"draft_data_public.{a.set}.PremierDraft.csv.gz")
    out = CACHE / f"{a.set}.{a.drafts}{'.' + a.salt if a.salt else ''}.drafts.json.gz"
    if not out.exists():
        sample.main(["--dump", str(dump), "--drafts", str(a.drafts), "--salt", a.salt, "--out", str(out)])
    return out


# ── top-drafter samples ──────────────────────────────────────────────────────

TOP_SKILL, TOP_GAMES = 0.60, 50  # upstream's "strong drafter": 60%+ game win rate over 50+ games


def top_dump(set_code):
    """The set's dump cut down to drafts by top drafters, cached as ...PremierDraft.top.csv.gz."""
    import csv
    import gzip

    src = fetch(DUMP_URL.format(set=set_code), CACHE / f"draft_data_public.{set_code}.PremierDraft.csv.gz")
    out = CACHE / f"draft_data_public.{set_code}.PremierDraft.top.csv.gz"
    if out.exists():
        return out
    tmp = out.with_suffix(".tmp")
    with gzip.open(src, "rt", newline="", encoding="utf-8") as fi, gzip.open(tmp, "wt", newline="", encoding="utf-8") as fo:
        reader, writer = csv.reader(fi), csv.writer(fo)
        header = next(reader)
        col = {h: i for i, h in enumerate(header)}
        if "user_game_win_rate_bucket" not in col:
            raise SystemExit(f"{set_code}'s dump has no skill buckets; top-drafter samples need them")
        writer.writerow(header)
        skill, games = col["user_game_win_rate_bucket"], col["user_n_games_bucket"]
        kept = 0
        for row in reader:
            try:
                if float(row[skill] or 0) >= TOP_SKILL and float(row[games] or 0) >= TOP_GAMES:
                    writer.writerow(row)
                    kept += 1
            except ValueError:
                continue
    tmp.rename(out)
    print(f"kept {kept} pick rows from top drafters")
    return out


def top_sample_paths(set_code, n):
    return {split: CACHE / f"{set_code}.top{n}.{split}.drafts.json.gz" for split in ("train", "test")}


def cmd_sample_top(a):
    """2n top-drafter drafts sampled with a fixed salt, split alternately into disjoint train and test sets."""
    import gzip
    import json

    paths = top_sample_paths(a.set, a.n)
    if all(p.exists() for p in paths.values()):
        return paths
    with gzip.open(top_dump(a.set), "rt", newline="", encoding="utf-8") as f:
        drafts = sample.sample_drafts(f, 2 * a.n, "top")
    for i, split in enumerate(("train", "test")):
        with gzip.open(paths[split], "wt", encoding="utf-8") as f:
            json.dump(drafts[i::2], f)
        print(f"{split}: {len(drafts[i::2])} top-drafter drafts -> {paths[split].name}")
    return paths


def cmd_sample_wins(a):
    """n top-drafter drafts that went exactly `wins` wins, none from the train or test samples."""
    import gzip
    import json

    out = CACHE / f"{a.set}.win{a.wins}_{a.n}.drafts.json.gz"
    if out.exists():
        return out
    used = set()
    for path in top_sample_paths(a.set, 600).values():
        if path.exists():
            with gzip.open(path, "rt", encoding="utf-8") as f:
                used |= {d["id"] for d in json.load(f)}
    with gzip.open(top_dump(a.set), "rt", newline="", encoding="utf-8") as f:
        pool = sample.sample_drafts(f, 20 * a.n, f"win{a.wins}")
    chosen = [d for d in pool if d["wins"] == a.wins and d["id"] not in used][: a.n]
    with gzip.open(out, "wt", encoding="utf-8") as f:
        json.dump(chosen, f)
    print(f"{len(chosen)} drafts with {a.wins} wins (of {len(pool)} sampled), none from train/test -> {out.name}")
    return out


def replay_one(set_code, drafts, advisor, out):
    use_advisor(advisor)
    replay.main(["--src", str(VENDOR), "--drafts", str(drafts), "--dataset", str(dataset_path(set_code)),
                 "--set", set_code, "--label", "base" if advisor == "upstream" else "head", "--out", str(out)])
    return out


def cmd_top(a):
    """v2 on a top-drafter sample; upstream's replay of the same sample is cached and shown as ref."""
    from tools import results as ledger
    from tools.mock_draft import apply_overrides

    apply_overrides(a.v2)
    if a.sample:
        drafts = Path(a.sample) if Path(a.sample).is_absolute() else CACHE / a.sample
        stem = drafts.name.replace(".drafts.json.gz", "")
        a.split = stem
    else:
        drafts = top_sample_paths(a.set, a.n)[a.split]
        if not drafts.exists():
            cmd_sample_top(a)
        stem = f"{a.set}.top{a.n}.{a.split}"
    dataset = dataset_path(a.set)
    base = CACHE / f"{stem}.base.json"
    if not base_is_current(base, drafts, dataset):
        replay_one(a.set, drafts, "upstream", base)
        base.with_suffix(".key").write_text(base_key(drafts, dataset))
    head = replay_one(a.set, drafts, "v2", CACHE / f"{stem}.head.{os.getpid()}.json")
    metrics = {"v2": ledger.top_drafter_metrics(head, drafts), "upstream_ref": ledger.top_drafter_metrics(base, drafts)}
    head.unlink()
    md = ledger.top_markdown(a.set, a.split, metrics)
    if getattr(a, "json", False):
        import json

        print("METRICS " + json.dumps(metrics["v2"]))
        return metrics
    print(md)
    if not a.no_record:
        rec = ledger.record("top", a.set, {"split": a.split, "n": a.n}, metrics, md, a.note)
        print(f"recorded in results/runs.jsonl as {rec['report']}")
    return metrics


def cmd_replay(a):
    use_advisor(a.advisor)
    here = lambda p: str(LAUNCH_DIR / p)  # the package chdirs into the data dir
    replay.main(["--src", str(VENDOR), "--drafts", here(a.drafts), "--dataset", here(a.dataset),
                 "--set", a.set, "--label", a.label, "--out", here(a.out)])


def base_key(drafts, dataset):
    """Upstream's replay depends only on the sampled drafts, the dataset and upstream's code."""
    import subprocess

    from tools.mock_draft import _sha

    vendor = subprocess.run(["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD:vendor/mtga_draft_17lands"],
                            capture_output=True, text=True).stdout.strip()
    return "|".join([_sha(drafts), _sha(dataset), vendor or "no-git"])


def base_is_current(out, drafts, dataset):
    key = out.with_suffix(".key")
    return out.exists() and key.exists() and key.read_text() == base_key(drafts, dataset)


def cmd_compare(a):
    from tools.mock_draft import apply_overrides

    apply_overrides(a.v2)
    drafts = cmd_sample(a)
    dataset = dataset_path(a.set)
    results = []
    for label, advisor in (("base", "upstream"), ("head", "v2")):
        out = CACHE / f"{a.set}.{a.drafts}{'.' + a.salt if a.salt else ''}.{label}.json"
        if label == "base" and not a.no_cache and base_is_current(out, drafts, dataset):
            print(f"reusing upstream's replay {out.name}")
            results.append(str(out))
            continue
        use_advisor(advisor)
        replay.main(["--src", str(VENDOR), "--drafts", str(drafts), "--dataset", str(dataset),
                     "--set", a.set, "--label", label, "--out", str(out)])
        results.append(str(out))
        if label == "base":
            out.with_suffix(".key").write_text(base_key(drafts, dataset))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report.main(results)
    from tools import results as ledger

    by_kind = ledger.harness_by_kind(results[0], results[1], drafts, dataset)
    md = buf.getvalue().rstrip() + "\n\n" + ledger.by_kind_markdown(by_kind) + "\n"
    print(md)
    metrics = ledger.harness_metrics(md)
    metrics["by_kind"] = by_kind
    rec = ledger.record("harness", a.set, {"drafts": a.drafts, "salt": a.salt}, metrics, md, a.note)
    print(f"recorded in results/runs.jsonl as {rec['report']}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("set"); s.add_argument("--drafts", type=int, default=300); s.add_argument("--salt", default="")
    s.set_defaults(fn=cmd_sample)
    r = sub.add_parser("replay"); r.add_argument("--advisor", choices=("upstream", "v2"), required=True)
    for opt in ("--drafts", "--dataset", "--label", "--out"):
        r.add_argument(opt, required=True)
    r.add_argument("--set", default=""); r.set_defaults(fn=cmd_replay)
    c = sub.add_parser("compare"); c.add_argument("set"); c.add_argument("--drafts", type=int, default=300); c.add_argument("--salt", default="")
    c.add_argument("--note", default="", help="what this run is testing; kept in results/runs.jsonl")
    c.add_argument("--v2", action="append", default=[], metavar="NAME=VALUE", help="override a v2 constant or module.SWITCH, as in tools/mock_draft.py")
    c.add_argument("--no-cache", action="store_true", help="replay upstream again instead of reusing its saved replay of this sample")
    c.set_defaults(fn=cmd_compare)
    st = sub.add_parser("sample-top"); st.add_argument("set"); st.add_argument("--n", type=int, default=600); st.set_defaults(fn=cmd_sample_top)
    sw = sub.add_parser("sample-wins"); sw.add_argument("set"); sw.add_argument("--n", type=int, default=600); sw.add_argument("--wins", type=int, default=7)
    sw.set_defaults(fn=cmd_sample_wins)
    t = sub.add_parser("top"); t.add_argument("set"); t.add_argument("--split", choices=("train", "test"), default="train")
    t.add_argument("--sample", help="another drafts file in the replay cache (e.g. FRA.win7_600.drafts.json.gz) instead of train/test")
    t.add_argument("--n", type=int, default=600); t.add_argument("--note", default="")
    t.add_argument("--v2", action="append", default=[], metavar="NAME=VALUE"); t.add_argument("--no-record", action="store_true", help="tuning runs: print only")
    t.add_argument("--json", action="store_true", help="print v2's metrics as one METRICS line (for tools/tune.py)")
    t.set_defaults(fn=cmd_top)
    rp = sub.add_parser("report"); rp.add_argument("results", nargs="+"); rp.set_defaults(fn=lambda a: report.main(a.results))
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
