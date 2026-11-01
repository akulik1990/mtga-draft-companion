"""Drive upstream's Tools/AdvisorReplay with a choice of advisor.

    uv run python tools/replay/run.py compare FRA --drafts 300 [--salt x]
    uv run python tools/replay/run.py sample FRA --drafts 300 [--salt x]
    uv run python tools/replay/run.py replay --advisor v2 --drafts D.json.gz --dataset S.json --label head --out r.json
    uv run python tools/replay/run.py report results/*.json

`compare` is the gate for scorer changes: it samples drafts (downloading the
17Lands dump once into <data>/Temp/replay/), replays them through upstream's
advisor as "base" and ours as "head", and prints upstream's report. Vendor code
is used as-is; only the advisor class is swapped.
"""

import argparse
import sys
from pathlib import Path

import requests  # upstream dependency; carries certifi, which the uv-managed Python needs on NixOS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # run by path, not only as -m
import companion  # noqa: E402,F401  (vendor on sys.path, cwd = data dir)
from companion import DATA_DIR, VENDOR  # noqa: E402

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


def use_advisor(name):
    """Make replay.App construct the chosen advisor. Everything else is upstream's."""
    cls = advisor_class(name)
    original = replay.App.__init__

    def __init__(self, *a, **k):
        original(self, *a, **k)
        self.DraftAdvisor = cls

    replay.App.__init__ = __init__


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


def cmd_replay(a):
    use_advisor(a.advisor)
    replay.main(["--src", str(VENDOR), "--drafts", a.drafts, "--dataset", a.dataset,
                 "--set", a.set, "--label", a.label, "--out", a.out])


def cmd_compare(a):
    drafts = cmd_sample(a)
    dataset = dataset_path(a.set)
    results = []
    for label, advisor in (("base", "upstream"), ("head", "v2")):
        out = CACHE / f"{a.set}.{a.drafts}{'.' + a.salt if a.salt else ''}.{label}.json"
        use_advisor(advisor)
        replay.main(["--src", str(VENDOR), "--drafts", str(drafts), "--dataset", str(dataset),
                     "--set", a.set, "--label", label, "--out", str(out)])
        results.append(str(out))
    report.main(results)


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
    c.set_defaults(fn=cmd_compare)
    rp = sub.add_parser("report"); rp.add_argument("results", nargs="+"); rp.set_defaults(fn=lambda a: report.main(a.results))
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
