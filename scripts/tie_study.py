"""Honest study of exact/near tie-breaking rules (research only).

Only cached public train/validation features are used.  No test path is opened.
Model families are compared with five-fold scene OOF on train.  Public
validation is evaluated only once, after selecting a family by train OOF.

The route-cost bank stores, for every first move, its cost gap from the best
move (in edge-length units).  ``epsilon=0`` is a true tie under a hypothesis;
positive epsilon deliberately tests whether apparent tie-breaking is really a
small but systematic route preference.
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "ml2"
PERMS = list(itertools.permutations(range(4)))
EPSILONS = (0.0, 0.05, 0.10, 0.20, 0.50, 1.0)
DIR_NAMES = "UDLR"
REL_NAMES = "SRUL"  # straight, right, U-turn, left


def load():
    # These filenames are an explicit allow-list: test_private is never read.
    tr = pickle.loads((OUT / "feats_train.pkl").read_bytes())
    va = pickle.loads((OUT / "feats_validation.pkl").read_bytes())
    st = pickle.loads((OUT / "states_train.pkl").read_bytes())
    sv = pickle.loads((OUT / "states_validation.pkl").read_bytes())
    return tr, va, st, sv


def rel_table():
    # Direction encoding is U,D,L,R = 0,1,2,3.  Reproduce policy.relative_turn
    # without importing any loader that might inspect another split.
    clockwise = (0, 3, 1, 2)
    return np.asarray([[((clockwise.index(d) - clockwise.index(h)) % 4)
                        for d in range(4)] for h in range(4)], np.int8)


REL = rel_table()


def rank_tables(kind):
    tabs = np.empty((24, 4, 4), np.int8)
    for pi, p in enumerate(PERMS):
        for h in range(4):
            for d in range(4):
                key = int(REL[h, d]) if kind == "rel" else d
                tabs[pi, h, d] = p.index(key)
    return tabs


RTAB, ATAB = rank_tables("rel"), rank_tables("abs")


def contexts(d, states):
    return {
        "global": np.zeros(len(d["Y"]), np.int8),
        "heading": np.asarray([x["heading"] for x in states["scenes"]], np.int8),
        "rain": d["conds"][:, 0].astype(np.int8),
        "urgent": d["conds"][:, 1].astype(np.int8),
        "fragile": d["conds"][:, 2].astype(np.int8),
        "via": d["has_via"].astype(np.int8),
        "night": np.asarray([s == "night" for s in d["style"]], np.int8),
    }


FAMILIES = {
    "relative_global": ("rel", "global"),
    "absolute_global": ("abs", "global"),
    "relative_by_heading": ("rel", "heading"),
    "absolute_by_heading": ("abs", "heading"),
    "relative_by_rain": ("rel", "rain"),
    "relative_by_urgent": ("rel", "urgent"),
    "relative_by_fragile": ("rel", "fragile"),
    "relative_by_via": ("rel", "via"),
    "relative_by_night": ("rel", "night"),
}


def candidate_sets(data, robot, bank, epsilon):
    legged = robot == 4
    X = data["res"][legged]["X"]
    legal = data["res"][legged]["legal"]
    gap = X[:, :, 3 * bank]
    return legal & (gap <= epsilon + 1e-7)


def order_choices(eligible, heading, kind):
    """Return chosen direction [24,n] for all priority orders."""
    tab = RTAB if kind == "rel" else ATAB
    ranks = tab[:, heading, :]  # [24,n,4]
    score = np.where(eligible[None, :, :], ranks, 99)
    return score.argmin(2).astype(np.int8)


def fit_orders(eligible, heading, labels, groups, train_idx, kind):
    choices = order_choices(eligible, heading, kind)
    correct = choices == labels[None, :]
    fallback = int(correct[:, train_idx].sum(1).argmax())
    selected = {"fallback": fallback}
    for g in np.unique(groups[train_idx]):
        ii = train_idx[groups[train_idx] == g]
        selected[str(int(g))] = int(correct[:, ii].sum(1).argmax()) if len(ii) else fallback
    return selected, choices


def apply_orders(choices, groups, selected, idx):
    out = np.empty(len(idx), np.int8)
    for j, i in enumerate(idx):
        p = selected.get(str(int(groups[i])), selected["fallback"])
        out[j] = choices[p, i]
    return out


def fit_candidate(data, states, robot, bank, eps, family, train_idx, test_idx):
    kind, group_name = FAMILIES[family]
    ctx = contexts(data, states)
    eligible = candidate_sets(data, robot, bank, eps)
    selected, choices = fit_orders(eligible, ctx["heading"], data["Y"][:, robot],
                                   ctx[group_name], train_idx, kind)
    pred = apply_orders(choices, ctx[group_name], selected, test_idx)
    return pred, selected


def search(data, states, robot, family, train_idx):
    y = data["Y"][:, robot]
    ctx = contexts(data, states)
    kind, group_name = FAMILIES[family]
    best = (-1, None)
    for bank in range(len(data["bank"])):
        for eps in EPSILONS:
            eligible = candidate_sets(data, robot, bank, eps)
            selected, choices = fit_orders(eligible, ctx["heading"], y, ctx[group_name], train_idx, kind)
            pred = apply_orders(choices, ctx[group_name], selected, train_idx)
            score = int((pred == y[train_idx]).sum())
            # Prefer the simpler exact-tie explanation when scores are equal.
            key = (score, -eps, -bank)
            if best[1] is None or key > best[0]:
                best = (key, (bank, eps, selected))
    return best[1]


def prepare_choices(data, states, robot):
    """Cache all candidate/order predictions; about 76 MB per robot."""
    heading = contexts(data, states)["heading"]
    specs = [(b, e) for b in range(len(data["bank"])) for e in EPSILONS]
    out = {"rel": np.empty((len(specs), 24, len(heading)), np.int8),
           "abs": np.empty((len(specs), 24, len(heading)), np.int8)}
    for ci, (bank, eps) in enumerate(specs):
        eligible = candidate_sets(data, robot, bank, eps)
        out["rel"][ci] = order_choices(eligible, heading, "rel")
        out["abs"][ci] = order_choices(eligible, heading, "abs")
    return specs, out


def search_cached(choices, specs, y, groups, train_idx, allowed=None):
    """Jointly select bank/epsilon and one priority order per context group."""
    allowed = np.arange(len(specs)) if allowed is None else np.asarray(allowed)
    subchoices = choices[allowed]
    correct = subchoices[:, :, train_idx] == y[train_idx][None, None, :]
    total = np.zeros((len(allowed),), np.int32)
    group_orders = {}
    for g in np.unique(groups[train_idx]):
        mask = groups[train_idx] == g
        score = correct[:, :, mask].sum(2)
        order = score.argmax(1)
        total += score[np.arange(len(allowed)), order]
        group_orders[str(int(g))] = order
    # exact ties and earlier banks are deterministic simplicity tie-breaks
    best_score = total.max()
    candidates = np.flatnonzero(total == best_score)
    subi = min(candidates, key=lambda z: (specs[allowed[z]][1], specs[allowed[z]][0]))
    ci = int(allowed[subi])
    fallback_score = correct[subi].sum(1)
    selected = {"fallback": int(fallback_score.argmax())}
    selected.update({k: int(v[subi]) for k, v in group_orders.items()})
    return ci, selected


def apply_cached(choices, ci, groups, selected, idx):
    out = np.empty(len(idx), np.int8)
    for j, i in enumerate(idx):
        p = selected.get(str(int(groups[i])), selected["fallback"])
        out[j] = choices[ci, p, i]
    return out


def order_text(kind, p):
    alphabet = REL_NAMES if kind == "rel" else DIR_NAMES
    return "".join(alphabet[x] for x in PERMS[p])


def diagnostics(data, states, robot, family, bank, eps, selected):
    ctx = contexts(data, states)
    kind, group_name = FAMILIES[family]
    eligible = candidate_sets(data, robot, bank, eps)
    exact = candidate_sets(data, robot, bank, 0.0)
    choices = order_choices(eligible, ctx["heading"], kind)
    idx = np.arange(len(data["Y"]))
    pred = apply_orders(choices, ctx[group_name], selected, idx)
    y = data["Y"][:, robot]
    neligible = eligible.sum(1)
    nexact = exact.sum(1)
    orders = {k: order_text(kind, v) for k, v in selected.items()}
    return {
        "accuracy": round(float(np.mean(pred == y)), 6),
        "exact_tie_scenes": int((nexact > 1).sum()),
        "near_tie_scenes": int((neligible > 1).sum()),
        "expanded_beyond_exact": int(((neligible > nexact) & (neligible > 1)).sum()),
        "label_in_candidate_set": round(float(np.mean(eligible[np.arange(len(y)), y])), 6),
        "orders": orders,
    }


def main():
    tr, va, st, sv = load()
    robots = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(1, 10))
    folds = list(KFold(5, shuffle=True, random_state=2026).split(np.arange(len(tr["Y"]))))
    result = {
        "protocol": "5-fold scene OOF on public train; family selected by OOF; validation after selection; no test read",
        "epsilons_edge_units": list(EPSILONS),
        "direction_encoding": "absolute UDLR; relative SRUL = straight/right/U-turn/left",
        "robots": {},
    }
    for robot in robots:
        t0 = time.time()
        specs, cached = prepare_choices(tr, st, robot)
        exact_candidates = [i for i, (_, e) in enumerate(specs) if e == 0]
        trctx = contexts(tr, st)
        fam_records = {}
        fam_oof = {}
        for family in FAMILIES:
            oof = np.full(len(tr["Y"]), -1, np.int8)
            oof_exact = np.full(len(tr["Y"]), -1, np.int8)
            picks = []
            kind, group_name = FAMILIES[family]
            for train_idx, test_idx in folds:
                ci, selected = search_cached(cached[kind], specs, tr["Y"][:, robot],
                                             trctx[group_name], train_idx)
                bank, eps = specs[ci]
                pred = apply_cached(cached[kind], ci, trctx[group_name], selected, test_idx)
                oof[test_idx] = pred
                eci, eselected = search_cached(cached[kind], specs, tr["Y"][:, robot],
                                               trctx[group_name], train_idx, exact_candidates)
                oof_exact[test_idx] = apply_cached(cached[kind], eci, trctx[group_name],
                                                   eselected, test_idx)
                picks.append({"bank": bank, "epsilon": eps,
                              "train_orders": diagnostics(tr, st, robot, family, bank, eps, selected)["orders"]})
            fam_records[family] = {
                "oof": round(float(np.mean(oof == tr["Y"][:, robot])), 6),
                "oof_exact_only": round(float(np.mean(oof_exact == tr["Y"][:, robot])), 6),
                "fold_accuracy": [round(float(np.mean(oof[b] == tr["Y"][b, robot])), 6) for _, b in folds],
                "fold_choices": picks,
            }
            # This is deliberately after OOF. Validation never selects the family.
            fci, forders = search_cached(cached[kind], specs, tr["Y"][:, robot],
                                         trctx[group_name], np.arange(len(tr["Y"])))
            fbank, feps = specs[fci]
            vctx = contexts(va, sv)
            veligible = candidate_sets(va, robot, fbank, feps)
            vchoices = order_choices(veligible, vctx["heading"], kind)
            vpred_family = apply_orders(vchoices, vctx[group_name], forders,
                                        np.arange(len(va["Y"])))
            fam_records[family]["validation_sanity"] = round(
                float(np.mean(vpred_family == va["Y"][:, robot])), 6)
            fam_records[family]["full_train_bank"] = fbank
            fam_records[family]["full_train_epsilon"] = feps
            fam_oof[family] = oof
        # Complexity-aware choice: OOF accuracy first; global families win exact ties.
        complexity = {"relative_global": 0, "absolute_global": 0}
        selected_family = max(FAMILIES, key=lambda f: (fam_records[f]["oof"], -complexity.get(f, 1)))
        kind, group_name = FAMILIES[selected_family]
        ci, orders = search_cached(cached[kind], specs, tr["Y"][:, robot], trctx[group_name],
                                   np.arange(len(tr["Y"])))
        bank, eps = specs[ci]
        trdiag = diagnostics(tr, st, robot, selected_family, bank, eps, orders)
        vc = contexts(va, sv)
        veligible = candidate_sets(va, robot, bank, eps)
        vchoices = order_choices(veligible, vc["heading"], kind)
        vpred = apply_orders(vchoices, vc[group_name], orders, np.arange(len(va["Y"])))
        vdiag = diagnostics(va, sv, robot, selected_family, bank, eps, orders)
        vdiag["accuracy"] = round(float(np.mean(vpred == va["Y"][:, robot])), 6)
        result["robots"][str(robot)] = {
            "families": fam_records,
            "selected_by_oof": selected_family,
            "final_bank": bank,
            "bank_weights": tr["bank"][bank],
            "epsilon": eps,
            "train_fit_diagnostics": trdiag,
            "validation_sanity": vdiag,
            "selected_oof_predictions": fam_oof[selected_family].astype(int).tolist(),
        }
        (OUT / "tie_study.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"tie R{robot}: {selected_family} OOF={fam_records[selected_family]['oof']:.4f} "
              f"bank={bank} eps={eps:g} val={vdiag['accuracy']:.4f} ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
