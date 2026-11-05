"""Criteria search, part 1: what in a deck's make-up carries results, all fields jointly, on the clean design.

    uv run python tools/criteria_search.py FRA all|first      # about 5-8 min each; writes <data>/tables/SET/10_criteria_<outcome>_*.csv

See tools/criteria_lib.py for the design. Outputs:
  _models.csv   held-out gain over the context model for quality / quality+ / colour / structure / combinations, each half
                separately (the "other half" confirmation), and structure's increment over quality with its SE
  _groups.csv   per theme group (curve, creatures, removal, card flow, evasion, defence, tricks, mana, colour, mechanics, types):
                gain over quality alone, and loss of gain when the group is removed from the full model (unique contribution)
  _axes.csv     how many latent deck axes (PCA of the structure fields, within colour combination) carry the structure gain
  _fields.csv   permutation importance per field (held-out half, 3 repeats), with the group
The model "full" = quality + colour combination + structure fields (+ derived splits). "quality+" adds mean / weak / strong GIH WR
of the spells (card-level, from the public set data, which contains these games: a mild leak, so reported next to plain quality).
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from criteria_lib import *  # noqa: E402,F403

GROUPS = {
    "curve": "spells lands avg_mv 1 2 3 4 5 6+ finishers",
    "creatures": "creature power toughness role_creatures creature_tokens tokens cr_early cr_late cr_big cr_blockers cr_late_share noncreature",
    "removal": "role_removal destroy exile_target damage fight shrink tap_down bounce counterspell edict any_target drain rm_cheap rm_late mech_fight",
    "card flow": "draw scry surveil loot cycling role_card_advantage flow_cheap",
    "evasion": "flying menace trample haste role_evasion double_strike mech_flying mech_haste mech_trample mech_double_strike",
    "defence": "lifelink gain_life defender reach deathtouch first_strike vigilance indestructible hexproof ward tag_protection mech_deathtouch "
               "mech_first_strike mech_lifelink mech_vigilance",
    "tricks": "pump tag_combat_trick flash anthem plus_counters prowess",
    "mana": "makes_mana treasure tag_fixing_ramp role_mana mana_sink tag_mana_sink land_mono land_multi land_colourless col_colourless col_multi "
            "double_pip mech_domain mech_landfall",
    "colour": "col_W lc_W land_W col_U lc_U land_U col_B lc_B land_B col_R lc_R land_R col_G lc_G land_G",
    "mechanics": "etb dies attack_trigger recursion copy empower_jace prepared threshold flashback noncombat_payoff combat_damage_trigger sacrifice mill "
                 "discard mech_behold mech_convoke mech_exhaust mech_proliferate mech_replicate tag_hate",
    "types": "artifact enchantment instant sorcery planeswalker legendary equipment aura double_faced",
}
GROUP_OF = {f: g for g, s in GROUPS.items() for f in s.split()}
DERIVED = ("cr_early", "cr_late", "cr_big", "cr_blockers", "cr_late_share", "rm_cheap", "rm_late", "flow_cheap", "finishers", "noncreature")


def build(D):
    F = card_features(D)
    fields, _ = merge_fields(D["rows"], [k for k in D["rows"][0] if k not in TEXT and k not in META])
    quality = [f for f in fields if f.startswith("rank_") or f in ("common", "uncommon", "rare", "mythic")]
    structure = [f for f in fields if f not in quality]
    cols = {f: np.array([r[f] for r in D["rows"]], float) for f in structure + quality}
    cols.update({f: F[f] for f in DERIVED})
    structure += [f for f in DERIVED if f not in structure]
    base = np.array([r["combo"].split("+")[0] for r in D["rows"]])
    combos = [c for c in sorted(set(base)) if (base == c).sum() >= 300]
    colour = np.array([[b == c for c in combos[1:]] for b in base], float)
    sets = dict(quality=standardise(np.column_stack([cols[f] for f in quality])),
                quality_plus=standardise(np.column_stack([cols[f] for f in quality] + [F["gih_mean"], F["weak"], F["strong"]])),
                colour=colour, structure=standardise(np.column_stack([cols[f] for f in structure])))
    return sets, structure, quality, cols, base, combos, F


def main():
    set_code = sys.argv[1] if len(sys.argv) > 1 else "FRA"
    outcome = sys.argv[2] if len(sys.argv) > 2 else "all"
    t0 = time.time()
    D = load_data(set_code, outcome)
    Xc = context(D, skill=True)
    sets, structure, quality, cols, base, combos, F = build(D)
    print(f"{outcome}: {len(D['w'])} decks, {int(D['n'].sum())} games, win rate {100 * D['w'].sum() / D['n'].sum():.1f}%, "
          f"{len(structure)} structure fields, {len(quality)} quality fields, {len(combos)} colour combinations ({time.time() - t0:.0f}s)", flush=True)
    name = lambda s: tables.path(set_code, f"10_criteria_{outcome}_{s}.csv")
    z0, _ = crossfit(Xc, np.zeros((len(D["w"]), 0)), D, 0)
    halves = [D["half"] == h for h in (0, 1)]

    def stack(*keys):
        return np.hstack([sets[k] for k in keys])

    models = [("quality", ("quality",)), ("quality+", ("quality_plus",)), ("colour combination", ("colour",)), ("structure", ("structure",)),
              ("quality + structure", ("quality", "structure")), ("quality + colour + structure", ("quality", "colour", "structure")),
              ("quality+ + colour + structure", ("quality_plus", "colour", "structure"))]
    res, zs = [], {}
    for label, keys in models:
        Xd = stack(*keys)
        lam = choose_lam(Xc, Xd, D)
        z, zd = crossfit(Xc, Xd, D, lam)
        zs[label] = (z, zd, lam, keys)
        g, se = gain(z0, z, D)
        gh = [gain(z0, z, D, m) for m in halves]
        spread = 100 * sigmoid(z0.mean() + zd).std() if Xd.shape[1] else 0
        res.append(dict(model=label, lam=lam, gain=round(g, 5), se=round(se, 5), gain_half0=round(gh[0][0], 5), se_half0=round(gh[0][1], 5),
                        gain_half1=round(gh[1][0], 5), se_half1=round(gh[1][1], 5), spread_sd_points=round(float(spread), 2)))
        print(f"  {label:32} lam {lam:6} gain {g:.5f} (SE {se:.5f})  halves {gh[0][0]:.5f} / {gh[1][0]:.5f}  spread {spread:.1f} pts  [{time.time() - t0:.0f}s]", flush=True)
    for a, b in (("quality", "quality + structure"), ("quality", "quality + colour + structure"), ("quality+", "quality+ + colour + structure"),
                 ("colour combination", "quality + colour + structure")):
        g, se = gain(zs[a][0], zs[b][0], D)
        res.append(dict(model=f"increment: {b} over {a}", gain=round(g, 5), se=round(se, 5)))
        print(f"  increment {b} over {a}: {g:.5f} (SE {se:.5f}, z {g / se:.1f})", flush=True)
    write(name("models"), res)

    # Groups: unique contribution inside the full model, and gain over quality alone
    full_keys = ("quality", "colour", "structure")
    lam_full = zs["quality + colour + structure"][2]
    z_full = zs["quality + colour + structure"][0]
    z_q = zs["quality"][0]
    gi = {g: [i for i, f in enumerate(structure) if GROUP_OF.get(f) == g] for g in GROUPS}
    other = [f for f in structure if f not in GROUP_OF]
    print("  fields in no group:", other, flush=True)
    rows = []
    S = sets["structure"]
    for g, idx in gi.items():
        keep = [i for i in range(S.shape[1]) if i not in idx]
        Xd = np.hstack([sets["quality"], sets["colour"], S[:, keep]])
        z_wo, _ = crossfit(Xc, Xd, D, lam_full)
        Xo = np.hstack([sets["quality"], S[:, idx]])
        z_on, _ = crossfit(Xc, Xo, D, lam_full)
        uniq, use = gain(z_wo, z_full, D)
        alone, sea = gain(z_q, z_on, D)
        rows.append(dict(group=g, fields=len(idx), unique_gain=round(uniq, 5), unique_se=round(use, 5), unique_z=round(uniq / use, 1),
                         over_quality_gain=round(alone, 5), over_quality_se=round(sea, 5), over_quality_z=round(alone / sea, 1)))
        print(f"  group {g:10} ({len(idx):2} fields): unique {uniq:+.5f} (z {uniq / use:+.1f})  over quality {alone:+.5f} (z {alone / sea:+.1f})  [{time.time() - t0:.0f}s]", flush=True)
    other_i = [i for i, f in enumerate(structure) if f in other]
    write(name("groups"), rows)

    # Latent axes: PCA of structure (colour-identity fields out, demeaned within colour combination)
    keep_i = [i for i, f in enumerate(structure) if GROUP_OF.get(f) != "colour"]
    M = S[:, keep_i].copy()
    for c in combos:
        m = base == c
        M[m] -= M[m].mean(0)
    M[~np.isin(base, combos)] -= M[~np.isin(base, combos)].mean(0)
    M = standardise(M)
    U, sv, Vt = np.linalg.svd(M, full_matrices=False)
    ev = sv ** 2 / (sv ** 2).sum()
    fl = [structure[i] for i in keep_i]
    axes = []
    for k in range(12):
        top = np.argsort(-np.abs(Vt[k]))[:8]
        axes.append(dict(axis=k + 1, variance_share=round(float(ev[k]), 3), top_fields=" ".join(f"{fl[j]}{Vt[k][j]:+.2f}" for j in top)))
        print(f"  axis {k + 1:2} {ev[k]:.3f}: " + " ".join(f"{fl[j]}{Vt[k][j]:+.2f}" for j in top), flush=True)
    Z = U * sv
    full_gain = gain(zs["quality + colour"] if "quality + colour" in zs else z_q, z_full, D)[0]
    base_qc = crossfit(Xc, np.hstack([sets["quality"], sets["colour"]]), D, lam_full)[0]
    tot = gain(base_qc, z_full, D)[0]
    for k in (2, 4, 6, 8, 12, 20, 40):
        Xd = np.hstack([sets["quality"], sets["colour"], standardise(Z[:, :k])])
        z_k, _ = crossfit(Xc, Xd, D, lam_full)
        g, se = gain(base_qc, z_k, D)
        axes.append(dict(axis=f"top {k} axes, gain beyond quality+colour", variance_share=round(float(ev[:k].sum()), 3), top_fields=f"gain {g:.5f} (SE {se:.5f}) of full-structure gain {tot:.5f} = {100 * g / tot:.0f}%"))
        print(f"  top {k:2} axes: gain beyond quality+colour {g:.5f} ({100 * g / tot:.0f}% of the full structure's {tot:.5f})  [{time.time() - t0:.0f}s]", flush=True)
    write(name("axes"), axes)

    # Permutation importance per field on the held-out half
    X = np.hstack([Xc, sets["quality"], sets["colour"], S])
    lam = lams(Xc.shape[1], X.shape[1] - Xc.shape[1], lam_full)
    k0 = Xc.shape[1] + sets["quality"].shape[1] + sets["colour"].shape[1]
    rng = np.random.default_rng(0)
    imp = np.zeros(len(structure))
    tot_n = 0.0
    for h in (0, 1):
        b = fit(X, D["w"], D["n"], lam, D["half"] != h)
        te = np.where(D["half"] == h)[0]
        base_dev = dev(X[te] @ b, D["w"][te], D["n"][te]).sum()
        tot_n += D["n"][te].sum()
        for j in range(len(structure)):
            d = 0.0
            for _ in range(3):
                Xp = X[te].copy()
                Xp[:, k0 + j] = Xp[rng.permutation(len(te)), k0 + j]
                d += dev(Xp @ b, D["w"][te], D["n"][te]).sum() - base_dev
            imp[j] += d / 3
    imp /= tot_n
    order = np.argsort(-imp)
    write(name("fields"), [dict(field=structure[j], group=GROUP_OF.get(structure[j], "other"), importance_loss_increase=round(float(imp[j]), 6)) for j in order])
    print("  top fields by permutation importance (loss increase when shuffled, x1e5):")
    for j in order[:30]:
        print(f"    {structure[j]:22} {GROUP_OF.get(structure[j], 'other'):10} {1e5 * imp[j]:6.1f}")
    print(f"done in {time.time() - t0:.0f}s")


def write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys, restval="")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
