"""Prints the new deck-model terms (deckmodel_v2 = v2s_tw1, skill-fixed) with SEs in per-SD units and WR points per unit (at 56% x 44% x 100 = 24.6 points per logit). Plain python."""
import json
from pathlib import Path

ST = Path(__file__).resolve().parents[1] / "data" / "Temp" / "dm1106" / "stage2"
K = 100 * 0.56 * 0.44
for tag in ("v1_tw1", "v2_tw1", "v2s_tw1", "v2s_tw3"):
    m = json.loads((ST / f"{tag}.json").read_text())
    print(f"\n{tag}: l2 {m['l2']}, held-out loss {m['heldout_logloss']:.5f} (context only {m['context_only_logloss']:.5f})")
    print(f"  {'term':14s} {'per SD':>8s} {'SE':>6s} {'z':>6s} {'SD (units)':>10s} {'WR pts per unit':>16s}")
    for n, b, se, sd in zip(m["shape_names"], m["shape"], m["shape_se"], m["shape_sd"] + [1, 1, 1]):
        print(f"  {n:14s} {b:+8.3f} {se:6.3f} {b / se:+6.1f} {sd:10.3f} {K * b / sd:+16.2f}")
