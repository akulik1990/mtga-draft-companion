"""Per-set engine settings in three layers: manual > stats > baseline, with runtime overrides on top.

    sets/baseline.json        every adjustable value with its default (generated from the code: tools/sets.py baseline)
    sets/<SET>/stats.json     values worked out from that set's data, plus the fitted files beside it
    sets/<SET>/manual.json    only what a person chose to change for that set
    MTGA_V2_MANUAL=path       use another manual file (to test an adjusted layer against the current one)

Each layer is {"values": {NAME: value}, "pairs": {...}, "types": {...}, "cards": {...}, "combos": [...]}. NAME is an
engine constant (STEER) or a module switch (lanes.USE_PIPS). The tables adjust card values in WR points:
pairs {"UB": 1.0}, types {"creature": 0.5, "land": 0, "other": 0}, cards {"Card name": 2.0},
combos [{"cards": ["A", "B"], "points": 1.5}] (each gets the points when the other is in the pool).
Runtime overrides (--v2 NAME=VALUE, the app's engine switches) win over every layer.
"""

import importlib
import json
import os
from pathlib import Path

SETS = Path(__file__).with_name("sets")
MODULES = ("lanes", "signals", "composition", "synergy")
STRUCTURAL = {"lanes.ALL", "signals.COLORS"}  # names, not settings
TABLES = ("pairs", "types", "cards", "combos")

BASELINE = {}  # NAME -> value as the code defines it; filled by snapshot() when the engine module loads
RUNTIME = {}  # NAME -> value set at runtime; wins over every layer
ACTIVE = {"set": None, "tables": {t: ([] if t == "combos" else {}) for t in TABLES}, "source": {}, "applied": set()}


def set_file(set_code, name):
    """A fitted per-set file: sets/<SET>/<name>.json."""
    return SETS / set_code / f"{name}.json"


def _target(name):
    from companion.advisor_v2.engine import DraftAdvisorV2

    if "." in name:
        module, attr = name.split(".", 1)
        return importlib.import_module(f"companion.advisor_v2.{module}"), attr
    return DraftAdvisorV2, name


def snapshot(engine_cls):
    """Every scalar setting the code defines: engine constants and module switches."""
    out = {}
    for attr, value in vars(engine_cls).items():
        if attr.isupper() and isinstance(value, (bool, int, float, str)):
            out[attr] = value
    for module in MODULES:
        mod = importlib.import_module(f"companion.advisor_v2.{module}")
        for attr, value in vars(mod).items():
            name = f"{module}.{attr}"
            if attr.isupper() and name not in STRUCTURAL and isinstance(value, (bool, int, float, str)):
                out[name] = value
    return out


def layer(set_code, which):
    path = Path(os.environ["MTGA_V2_MANUAL"]) if which == "manual" and os.environ.get("MTGA_V2_MANUAL") else set_file(set_code, which)
    return json.loads(path.read_text()) if set_code and path.exists() else {}


def resolve(set_code):
    """(values, tables, source): each value from runtime, manual, stats or baseline, whichever has it first."""
    stats, manual = layer(set_code, "stats"), layer(set_code, "manual")
    values, source = {}, {}
    for name, base in BASELINE.items():
        for where, layer_values in (("runtime", RUNTIME), ("manual", manual.get("values") or {}), ("stats", stats.get("values") or {})):
            if name in layer_values:
                values[name], source[name] = layer_values[name], where
                break
        else:
            values[name], source[name] = base, "baseline"
    unknown = set((manual.get("values") or {}) | (stats.get("values") or {})) - set(BASELINE)
    if unknown:
        raise ValueError(f"{set_code}: unknown settings {sorted(unknown)}")
    tables = {t: (stats.get(t) or ([] if t == "combos" else {})) for t in TABLES}
    for t in TABLES:
        if t == "combos":
            tables[t] = list(tables[t]) + list(manual.get(t) or [])
        else:
            tables[t] = {**tables[t], **(manual.get(t) or {})}
    return values, tables, source


def activate(set_code, force=False):
    """Make the engine use this set's settings. Cheap when the set is already active.

    Only values a stats or manual layer sets are written; anything the previous set wrote goes back to baseline.
    Everything else is left as it is, so runtime changes (and tests' monkeypatches) are not undone.
    """
    if not set_code or (set_code == ACTIVE["set"] and not force):
        return
    values, tables, source = resolve(set_code)
    layered = {name for name, where in source.items() if where in ("manual", "stats")}
    for name in ACTIVE["applied"] - layered - set(RUNTIME):
        target, attr = _target(name)
        setattr(target, attr, BASELINE[name])
    for name in layered:
        target, attr = _target(name)
        setattr(target, attr, values[name])
    ACTIVE.update(set=set_code, tables=tables, source=source, applied=layered)


def override(name, value):
    """A runtime override (--v2, app switches): applied now and kept across set changes."""
    target, attr = _target(name)
    setattr(target, attr, value)
    RUNTIME[name] = value
