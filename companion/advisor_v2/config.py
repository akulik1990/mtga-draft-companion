"""Per-set engine settings in three layers: manual > fitted > baseline, with runtime overrides on top.

    sets/baseline.json        every adjustable value with its default (generated from the code: tools/sets.py baseline)
    sets/<SET>/fitted.json    everything worked out from that set's data: tuned values, and sections the fitting tools
                              write (prefs, branch, branch_colour, closecalls, deckmodel), with "sources" saying how
    sets/<SET>/manual.json    what comes from anywhere else (people, videos, guides): only what was changed
    MTGA_V2_MANUAL=path       use another manual file (to test an adjusted layer against the current one)

Each layer is {"values": {NAME: value}, "pairs": {...}, "types": {...}, "cards": {...}, "combos": [...]}. NAME is an
engine constant (STEER) or a module switch (lanes.USE_PIPS). The tables adjust card values in WR points:
pairs {"UB": 1.0}, types {"creature": 0.5, "land": 0, "other": 0}, cards {"Card name": 2.0},
combos [{"cards": ["A", "B"], "points": 1.5}] (each gets the points when the other is in the pool).
Runtime overrides (--v2 NAME=VALUE, the app's engine switches) win over every layer.
"""

import datetime
import importlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

SETS = Path(__file__).with_name("sets")
MODULES = ("lanes", "signals", "composition", "synergy", "deckbuilder")
STRUCTURAL = {"lanes.ALL", "signals.COLORS"}  # names, not settings
TABLES = ("pairs", "types", "cards", "combos")

BASELINE = {}  # NAME -> value as the code defines it; filled by snapshot() when the engine module loads
RUNTIME = {}  # NAME -> value set at runtime; wins over every layer
ACTIVE = {"set": None, "tables": {t: ([] if t == "combos" else {}) for t in TABLES}, "source": {}, "applied": set()}


def set_file(set_code, name):
    """sets/<SET>/<name>.json (fitted, manual)."""
    return SETS / set_code / f"{name}.json"


@lru_cache(maxsize=16)
def _fitted(path, mtime):
    return json.loads(Path(path).read_text())


def fitted(set_code, section):
    """One section of the set's fitted.json, {} when the set or the section has none."""
    path = set_file(set_code, "fitted") if set_code else None
    if not path or not path.exists():
        return {}
    return _fitted(str(path), path.stat().st_mtime).get(section) or {}


def write_fitted(set_code, section, data, how):
    """Store a fitting tool's output as one section of the set's fitted.json, and record how it was made."""
    path = set_file(set_code, "fitted")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = json.loads(path.read_text()) if path.exists() else {"values": {}, "sources": {}}
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    doc[section] = data
    doc.setdefault("sources", {})[section] = {"how": how, "date": datetime.date.today().isoformat(), "commit": commit}
    path.write_text(json.dumps(doc, indent=1) + "\n")
    return path


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


def tags(card, set_code=None):
    """The card's dataset tags with the set's manual corrections ("tag_fixes": {"remove": {tag: [names]}, "add": {...}}).
    The dataset takes its tags from Scryfall's automatic tagging, which calls face damage and counterspells "removal"."""
    out = list(card.get("tags") or [])
    fixes = _tag_fixes(set_code or card.get("set") or ACTIVE["set"])
    if fixes:
        names = {card.get("name"), (card.get("name") or "").split(" // ")[0]}
        for tag, cards in (fixes.get("remove") or {}).items():
            if names & set(cards):
                out = [t for t in out if t != tag]
        for tag, cards in (fixes.get("add") or {}).items():
            if names & set(cards) and tag not in out:
                out.append(tag)
    return out


@lru_cache(maxsize=8)
def _tag_fixes(set_code):
    return layer(set_code, "manual").get("tag_fixes") or {} if set_code else {}


def layer(set_code, which):
    path = Path(os.environ["MTGA_V2_MANUAL"]) if which == "manual" and os.environ.get("MTGA_V2_MANUAL") else set_file(set_code, which)
    return json.loads(path.read_text()) if set_code and path.exists() else {}


def resolve(set_code):
    """(values, tables, source): each value from runtime, manual, fitted or baseline, whichever has it first."""
    fitted_layer, manual = layer(set_code, "fitted"), layer(set_code, "manual")
    values, source = {}, {}
    for name, base in BASELINE.items():
        for where, layer_values in (("runtime", RUNTIME), ("manual", manual.get("values") or {}), ("fitted", fitted_layer.get("values") or {})):
            if name in layer_values:
                values[name], source[name] = layer_values[name], where
                break
        else:
            values[name], source[name] = base, "baseline"
    unknown = set((manual.get("values") or {}) | (fitted_layer.get("values") or {})) - set(BASELINE)
    if unknown:
        raise ValueError(f"{set_code}: unknown settings {sorted(unknown)}")
    tables = {t: (fitted_layer.get(t) or ([] if t == "combos" else {})) for t in TABLES}
    for t in TABLES:
        if t == "combos":
            tables[t] = list(tables[t]) + list(manual.get(t) or [])
        else:
            tables[t] = {**tables[t], **(manual.get(t) or {})}
    return values, tables, source


def activate(set_code, force=False):
    """Make the engine use this set's settings. Cheap when the set is already active.

    Only values a fitted or manual layer sets are written; anything the previous set wrote goes back to baseline.
    Everything else is left as it is, so runtime changes (and tests' monkeypatches) are not undone.
    """
    if not set_code or (set_code == ACTIVE["set"] and not force):
        return
    values, tables, source = resolve(set_code)
    layered = {name for name, where in source.items() if where in ("manual", "fitted")}
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
