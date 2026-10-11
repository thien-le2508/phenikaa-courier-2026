"""ML2 learned policy fallback (see work/ML2_NOTES.md).

predict_all_ml2(graph, mission, bundle) -> [10 ints]
  R0: exact rule from courier.policy.
  R1..R9: candidate ranker. For each legal first move the features are:
    - route-cost summaries under a bank of linear cost hypotheses, solved by Bellman-Ford over
      (node, heading, via-stage) states: relative cost, is-min, number of minimisers;
    - optional "inverse experts": per-robot per-condition weights fitted by inverse optimisation;
    - scene/move context: rain, urgent, fragile, via, style one-hot (night = hidden condition!),
      relative and absolute direction, first-edge attributes;
    - optionally the v1 bank features from courier.policy.move_features.
bundle = pickle from work/ml2/policy_ml2_{train,full}.pkl.
`graph["style"]` must be set (predicted by the style model at test time), as for predict_all_ml.
"""
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import ml2_core as core  # noqa: E402
import ml2_feats as mf  # noqa: E402
from courier.policy import Map, move_features, predict_all  # noqa: E402

_STYLES = mf.STYLES


def _scene_dict(graph, mission):
    o = core.build_scene(graph, mission)
    m = Map(graph)
    conds = np.asarray([[bool(m.rain), bool(mission.get("urgent")), bool(mission.get("fragile"))]], bool)
    style = graph.get("style") or "classic"
    return dict(scenes=[o], conds=conds, style=[style], has_via=np.asarray([bool(o["vias"])]),
                Y=np.zeros((1, 10), np.int64))


def _inv_Q(P, d, spec, legged):
    """Q features from per-group inverse-fitted weights: spec = {keys, groups: {bits: {w, order}}}."""
    import ml2_inv2 as inv
    g = int(inv.group_ids(d, spec["keys"])[0])
    bits = format(g, f"0{max(1, len(spec['keys']))}b")
    w = spec["groups"][bits]["w"]
    Q = mf.bank_Q(P, [w], legged, dev=P["succ"].device.type)
    return Q


def scene_features(graph, mission, bundle, device="cpu"):
    """{legged(bool): (X [4,f], legal [4])} for the ML2 bank (+ context)."""
    d = _scene_dict(graph, mission)
    P = core.pack(d["scenes"], device)
    ctx = mf.context(d)
    fs = mf.first_step(P)
    out = {}
    for legged in (False, True):
        Q = mf.bank_Q(P, bundle["bank"], legged, dev=device)
        X = np.concatenate([mf.move_matrix(Q), ctx, fs], -1).astype(np.float32)
        out[legged] = (X[0], mf.legal_mask(P, legged)[0], P, d)
    return out


def robot_matrix(feats, robot, spec, v1rows=None, extra=None):
    X, legal, P, d = feats[robot == 4]
    legal = legal.copy()
    parts = [X]
    if spec.get("inv") is not None:
        Q = _inv_Q(P, d, spec["inv"], robot == 4)
        parts.append(mf.move_matrix(Q)[0].astype(np.float32))
    if spec.get("v1"):
        k = spec["v1_dim"]
        Z = np.zeros((4, k), np.float32)
        for dd in range(4):
            if dd in v1rows:
                Z[dd] = v1rows[dd]
            else:
                legal[dd] = False
        parts.append(Z)
    return np.concatenate(parts, -1), legal


def predict_all_ml2(graph, mission, bundle, device="cpu"):
    base = predict_all(graph, mission)
    out = [base[0]]
    feats = scene_features(graph, mission, bundle, device)
    v1 = None
    if any(s.get("v1") for s in bundle["robots"].values()):
        v1 = move_features(graph, mission)
    for r in range(1, 10):
        spec = bundle["robots"].get(r)
        if spec is None or spec.get("use") == "exact":
            out.append(base[r])
            continue
        X, legal = robot_matrix(feats, r, spec, v1[r] if v1 else None)
        if not legal.any():
            out.append(base[r])
            continue
        idx = np.nonzero(legal)[0]
        p = np.mean([m.predict_proba(X[idx])[:, 1] for m in spec["models"]], 0)
        out.append(int(idx[int(np.argmax(p))]))
    return out
