"""OOF evaluation of structured path rules for robots R3--R8.

The study compares exact lexicographic objectives and explicit detour budgets.
Candidate selection is performed inside each scene-level fold.  Public
validation is used only as a final sanity check after OOF model selection.
No test/private path is referenced by this script.
"""
from __future__ import annotations

import itertools
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from exactB_common import env_of, load, relative_turn  # noqa: E402

OUT = ROOT / "work" / "ml2"
NAMES = "UDLR"
METRIC_COL = {"E": 0, "cr": 1, "nc": 2, "lm": 3, "R": 4, "L": 5,
              "U": 6, "stairs": 7, "oneway": 8}


def candidate_specs():
    # Shortest-route families test secondary preferences only.  Budgeted
    # families test whether a robot accepts b extra edges for a primary goal.
    specs = []
    shortest = [
        ("E",), ("E", "cr"), ("E", "nc"), ("E", "lm"),
        ("E", "turns"), ("E", "U"), ("E", "cr", "nc"),
        ("E", "nc", "cr"), ("E", "lm", "cr"), ("E", "lm", "nc"),
        ("E", "turns", "cr"), ("E", "turns", "nc"),
    ]
    specs.extend((0, seq) for seq in shortest)
    primary = [
        ("cr", "E"), ("nc", "E"), ("lm", "E"), ("turns", "E"),
        ("U", "E"), ("cr", "nc", "E"), ("nc", "cr", "E"),
        ("lm", "cr", "E"), ("lm", "nc", "E"),
        ("cr", "turns", "E"), ("nc", "turns", "E"),
        ("turns", "cr", "E"), ("turns", "nc", "E"),
        ("oneway", "E"), ("stairs", "E"), ("stairs", "cr", "E"),
        ("stairs", "nc", "E"),
    ]
    for budget in (0, 1, 2, 3, 4, 6):
        specs.extend((budget, seq) for seq in primary)
    # Preserve order while removing equivalent duplicates.
    return list(dict.fromkeys(specs))


def metric(row, name):
    if name == "turns":
        return row[:, 4] + row[:, 5] + row[:, 6]
    return row[:, METRIC_COL[name]]


def orders():
    out = []
    for kind in ("relative", "absolute"):
        for order in itertools.permutations(range(4)):
            out.append((kind, order))
    return out


def ordered_dirs(heading, spec):
    kind, order = spec
    if kind == "absolute":
        return order
    ranked = [None] * 4
    for direction in range(4):
        ranked[relative_turn(heading, direction)] = direction
    return tuple(ranked[x] for x in order)


def route_keys(item, budget, sequence):
    result = {}
    shortest = item["shortest"]
    if shortest is None:
        return result
    for direction, rows in item["fronts"].items():
        rows = rows[rows[:, 0] <= shortest + budget]
        if not len(rows):
            continue
        cols = [metric(rows, name) for name in sequence]
        index = np.lexsort(tuple(reversed(cols)))[0]
        result[int(direction)] = tuple(int(col[index]) for col in cols)
    return result


def all_predictions(payload, scenes):
    specs, ties = candidate_specs(), orders()
    n = len(scenes)
    pred = np.full((len(specs) * len(ties), n), -1, dtype=np.int8)
    names = []
    for si, (budget, sequence) in enumerate(specs):
        for oi, tie in enumerate(ties):
            names.append({"budget": budget, "lex": list(sequence),
                          "tie": [tie[0], list(tie[1])]})
        for i, (item, scene) in enumerate(zip(payload["fronts"], scenes)):
            keys = route_keys(item, budget, sequence)
            if not keys:
                continue
            best = min(keys.values())
            tied = {direction for direction, key in keys.items() if key == best}
            for oi, tie in enumerate(ties):
                pred[si * len(ties) + oi, i] = next(
                    direction for direction in ordered_dirs(scene["m"].heading, tie) if direction in tied)
    return pred, names


def context_arrays(scenes, payload):
    env = [env_of(scene) for scene in scenes]
    flags = np.asarray([[x["R"], x["U"], x["F"], x["V"], x["N"]] for x in env], dtype=np.int8)
    ambiguous = np.asarray([x["ambiguous_goal"] for x in payload["fronts"]], dtype=np.int8)
    style = np.asarray([{"classic": 0, "night": 1, "print": 2, "sketch": 3}[x["S"]] for x in env])
    return {"global": np.zeros(len(scenes), np.int16),
            "rain": flags[:, 0], "urgent": flags[:, 1], "fragile": flags[:, 2],
            "via": flags[:, 3], "night": flags[:, 4], "ambiguity": ambiguous,
            "conditions": flags[:, 0] * 4 + flags[:, 1] * 2 + flags[:, 2],
            "conditions_via": flags[:, 0] * 8 + flags[:, 1] * 4 + flags[:, 2] * 2 + flags[:, 3],
            "style": style}


def choose(hits, idx, groups, alpha=20.0):
    """Choose a candidate per group, shrunk toward its global OOF accuracy."""
    global_acc = hits[:, idx].mean(1)
    selected = {}
    for group in np.unique(groups):
        use = idx[groups[idx] == group]
        score = (hits[:, use].sum(1) + alpha * global_acc) / (len(use) + alpha)
        selected[int(group)] = int(np.argmax(score))
    return selected


def selector_oof(pred, labels, groups, folds, alpha):
    hits = pred == labels[None, :]
    out = np.full(len(labels), -1, np.int8)
    selections = []
    all_idx = np.arange(len(labels))
    for train_idx, test_idx in folds:
        picked = choose(hits, train_idx, groups, alpha)
        fallback = int(np.argmax(hits[:, train_idx].mean(1)))
        for i in test_idx:
            out[i] = pred[picked.get(int(groups[i]), fallback), i]
        selections.append(picked)
    return out, selections


def breakdown(pred, labels, contexts):
    hit = pred == labels
    result = {}
    for name in ("rain", "urgent", "fragile", "via", "night", "ambiguity"):
        result[name] = {}
        for value in (0, 1):
            mask = contexts[name] == value
            result[name][str(value)] = {"n": int(mask.sum()), "acc": round(float(hit[mask].mean()), 6)}
    return result


def stack_oof(base_train, base_val, rule_train, rule_val, labels_train, folds):
    """Conservative diagnostic blend: baseline probabilities + rule one-hot."""
    eye = np.eye(4)
    xtr = np.concatenate([base_train, eye[rule_train], base_train.max(1, keepdims=True)], 1)
    xva = np.concatenate([base_val, eye[rule_val], base_val.max(1, keepdims=True)], 1)
    oof = np.full(len(labels_train), -1, np.int8)
    for train_idx, test_idx in folds:
        model = LogisticRegression(C=0.1, max_iter=500).fit(xtr[train_idx], labels_train[train_idx])
        oof[test_idx] = model.predict(xtr[test_idx])
    final = LogisticRegression(C=0.1, max_iter=500).fit(xtr, labels_train)
    return oof, final.predict(xva)


def main():
    train_scenes, val_scenes = load("train"), load("validation")
    baseline = pickle.loads((OUT / "baseline_v2_probs.pkl").read_bytes())
    folds = list(KFold(5, shuffle=True, random_state=20261009).split(np.arange(len(train_scenes))))
    results = {"protocol": "5-fold scene OOF selection on train; validation sanity only; no private test",
               "robots": {}}
    diagnostics = {}
    route_cache = {}
    for robot in range(3, 9):
        kind = "legged" if robot == 4 else "wheeled"
        if kind not in route_cache:
            train_payload = pickle.loads((OUT / f"route_study_fronts_train_{kind}_s6.pkl").read_bytes())
            val_payload = pickle.loads((OUT / f"route_study_fronts_validation_{kind}_s6.pkl").read_bytes())
            ptr, names = all_predictions(train_payload, train_scenes)
            pva, _ = all_predictions(val_payload, val_scenes)
            route_cache[kind] = (train_payload, val_payload, ptr, pva, names)
        train_payload, val_payload, ptr, pva, names = route_cache[kind]
        ytr = np.asarray([scene["y"][robot] for scene in train_scenes], np.int8)
        yva = np.asarray([scene["y"][robot] for scene in val_scenes], np.int8)
        ctr, cva = context_arrays(train_scenes, train_payload), context_arrays(val_scenes, val_payload)
        structures = {}
        structure_validation = {}
        predictions = {}
        candidate_hits = ptr == ytr[None, :]
        for context_name in ("global", "rain", "urgent", "fragile", "via", "night", "ambiguity",
                             "conditions", "conditions_via", "style"):
            for alpha in (10.0, 30.0, 100.0):
                key = f"{context_name}|a={int(alpha)}"
                poof, _ = selector_oof(ptr, ytr, ctr[context_name], folds, alpha)
                structures[key] = float(np.mean(poof == ytr))
                predictions[key] = poof
                choice = choose(candidate_hits, np.arange(len(ytr)), ctr[context_name], alpha)
                fallback_choice = int(np.argmax(candidate_hits.mean(1)))
                pv = np.asarray([pva[choice.get(int(group), fallback_choice), i]
                                 for i, group in enumerate(cva[context_name])], np.int8)
                structure_validation[key] = float(np.mean(pv == yva))
        best_key = max(structures, key=structures.get)
        context_name, alpha_text = best_key.split("|a=")
        alpha = float(alpha_text)
        hits = ptr == ytr[None, :]
        selected = choose(hits, np.arange(len(ytr)), ctr[context_name], alpha)
        fallback = int(np.argmax(hits.mean(1)))
        val_rule = np.asarray([pva[selected.get(int(group), fallback), i]
                               for i, group in enumerate(cva[context_name])], np.int8)
        oof_rule = predictions[best_key]
        base_oof = np.asarray(baseline[robot]["oof"]).argmax(1)
        base_val = np.asarray(baseline[robot]["val"]).argmax(1)
        blend_oof, blend_val = stack_oof(np.asarray(baseline[robot]["oof"]),
                                         np.asarray(baseline[robot]["val"]),
                                         oof_rule, val_rule, ytr, folds)
        selected_named = {str(group): names[index] for group, index in selected.items()}
        results["robots"][str(robot)] = {
            "candidate_count": len(names), "selected_structure": best_key,
            "structure_oof": {k: round(v, 6) for k, v in sorted(structures.items())},
            "structure_validation": {k: round(v, 6) for k, v in sorted(structure_validation.items())},
            "rule_oof": round(float(np.mean(oof_rule == ytr)), 6),
            "rule_validation": round(float(np.mean(val_rule == yva)), 6),
            "baseline_oof": round(float(np.mean(base_oof == ytr)), 6),
            "baseline_validation": round(float(np.mean(base_val == yva)), 6),
            "blend_oof": round(float(np.mean(blend_oof == ytr)), 6),
            "blend_validation": round(float(np.mean(blend_val == yva)), 6),
            "selected_rules": selected_named,
            "rule_breakdown_oof": breakdown(oof_rule, ytr, ctr),
            "rule_breakdown_validation": breakdown(val_rule, yva, cva),
            "complementarity_oof": {
                "both_correct": int(((oof_rule == ytr) & (base_oof == ytr)).sum()),
                "rule_only": int(((oof_rule == ytr) & (base_oof != ytr)).sum()),
                "baseline_only": int(((oof_rule != ytr) & (base_oof == ytr)).sum()),
                "both_wrong": int(((oof_rule != ytr) & (base_oof != ytr)).sum()),
            },
        }
        diagnostics[str(robot)] = {
            "rule_oof_errors": np.flatnonzero(oof_rule != ytr).tolist(),
            "baseline_oof_errors": np.flatnonzero(base_oof != ytr).tolist(),
            "rule_validation_errors": np.flatnonzero(val_rule != yva).tolist(),
            "via_ambiguous_counts_train": dict(Counter(
                f"via={int(ctr['via'][i])},amb={int(ctr['ambiguity'][i])}" for i in range(len(ytr)))),
        }
        print(f"R{robot}: rule {results['robots'][str(robot)]['rule_oof']:.4f}/"
              f"{results['robots'][str(robot)]['rule_validation']:.4f}  baseline "
              f"{results['robots'][str(robot)]['baseline_oof']:.4f}/"
              f"{results['robots'][str(robot)]['baseline_validation']:.4f}  blend "
              f"{results['robots'][str(robot)]['blend_oof']:.4f}/"
              f"{results['robots'][str(robot)]['blend_validation']:.4f}", flush=True)
        (OUT / "route_study_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        (OUT / "route_study_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
