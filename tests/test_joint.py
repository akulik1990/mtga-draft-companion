"""tools/joint.py: variants on top of the base, all values together, with a fake harness."""

from tools import joint


def test_variants_apply_together_and_sort(monkeypatch, capsys):
    seen = []

    def fake(set_code, sample, overrides):
        seen.append(dict(overrides))
        v = 50.0 + (1.0 if overrides.get("PREF_WEIGHT") == "8" else 0.0) - (0.5 if overrides.get("K") == "1200" else 0.0)
        return {"top1": v, "top1_se": 1.0, "top3": v + 30, "picks": 4200, "by_stage": {"P1 early": {"top1": v}}}

    monkeypatch.setattr(joint, "run", fake)
    monkeypatch.setattr(joint, "ensure_sample", lambda s, n: n)
    joint.main(["FRA", "--v2", "GAMMA=0", "--try", "PREF_WEIGHT=8 K=1200", "--try", "PREF_WEIGHT=8", "--jobs", "1"])
    assert {"GAMMA": "0", "PREF_WEIGHT": "8", "K": "1200"} in seen  # the base under every variant, values together
    out = capsys.readouterr().out.splitlines()
    assert out[2].startswith("PREF_WEIGHT=8 ") and "+1.00" in out[2] and "+0.50" in out[3]
