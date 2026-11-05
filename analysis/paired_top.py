"""Paired top-1 / top-3 agreement with top drafters between two replay logs of the same drafts (analysis/replay_log.py), clustered
by draft: per-draft match rates, then a paired t over drafts (the decision rule in CLAUDE.md).

    python3 analysis/paired_top.py <log_a.json> <log_b.json> <drafts.json.gz>
"""
import gzip, json, math, sys
from collections import defaultdict
from statistics import mean, stdev

a, b = (json.load(open(p)) for p in sys.argv[1:3])
human = {(d["id"], p[0], p[1]): p[2] for d in json.load(gzip.open(sys.argv[3], "rt")) for p in d["picks"]}


def per_draft(log, k):
    out = defaultdict(list)
    for e in log:
        h = human.get(tuple(e[:3]))
        if e[3] and h is not None:
            out[e[0]].append(h in [n for n, _ in e[3][:k]])
    return {d: mean(v) for d, v in out.items()}


for k in (1, 3):
    A, B = per_draft(a, k), per_draft(b, k)
    diffs = [B[d] - A[d] for d in A if d in B]
    m, se = mean(diffs), stdev(diffs) / math.sqrt(len(diffs))
    p = math.erfc(abs(m / se) / math.sqrt(2)) if se else 1.0  # normal approximation; 600 drafts
    print(f"top-{k}: {100 * mean(A.values()):.2f} -> {100 * mean(B.values()):.2f}  diff {100 * m:+.2f} (SE {100 * se:.2f}, p {p:.3f}, {len(diffs)} drafts)")
