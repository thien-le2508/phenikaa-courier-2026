"""Train the selected research policy candidates on public train+validation.

This is the deployment refit used by predict_v2.py.  Selection was completed
with train OOF before this script; no private-test input is read here.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "work" / "ml2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import conditional_models as cm  # noqa: E402
import ml2_core as core  # noqa: E402
import ml2_nip as nip  # noqa: E402
from ml2_inv2 import group_ids  # noqa: E402


def rows(feats, labels, robot):
    x, y = [], []
    for scene, label in zip(feats, labels):
        for direction, row in scene[robot].items():
            x.append(row)
            y.append(direction == label[robot])
    return np.asarray(x, np.float32), np.asarray(y, np.int8)


def fit_hgb(x, y):
    return HistGradientBoostingClassifier(
        max_iter=800, learning_rate=.08, max_leaf_nodes=31,
        l2_regularization=1., early_stopping=False, random_state=0
    ).fit(x, y)


def combine_states(train, validation):
    return {
        "scenes": train["scenes"] + validation["scenes"],
        "Y": np.concatenate([train["Y"], validation["Y"]]),
        "conds": np.concatenate([train["conds"], validation["conds"]]),
        "has_via": np.concatenate([train["has_via"], validation["has_via"]]),
        "style": list(train["style"]) + list(validation["style"]),
    }


def main():
    feature_data = pickle.loads((ROOT / "work" / "policy_features_v1.pkl").read_bytes())
    feature_v2 = pickle.loads((ROOT / "work" / "policy_features_v2.pkl").read_bytes())
    ft, yt = feature_data["train"]
    fv, yv = feature_data["validation"]
    feats_v1 = ft + fv
    labels = np.concatenate([np.asarray(yt), np.asarray(yv)])
    feats_v2 = feature_v2["train"][0] + feature_v2["validation"][0]

    v1_models, v2_models = {}, {}
    for robot in range(1, 10):
        x, y = rows(feats_v1, labels, robot)
        v1_models[robot] = fit_hgb(x, y)
        print(f"trained v1 R{robot}", flush=True)
    for robot in (4, 5, 6, 7):
        x, y = rows(feats_v2, labels, robot)
        v2_models[robot] = fit_hgb(x, y)
        print(f"trained v2 R{robot}", flush=True)

    # R9 conditional structured utility, selected by train OOF.
    st, sv = core.cache("train"), core.cache("validation")
    all_states = combine_states(st, sv)
    ctx = np.column_stack([all_states["conds"].astype(float), all_states["has_via"].astype(float)])
    styles = np.asarray([[s == name for name in cm.STYLES] for s in all_states["style"]], float)
    x9, legal9 = cm.action_tensor(feats_v1, 9)
    design9, blocks9 = cm.design(x9, ctx, styles, "main_style")
    model9, scaler9, mult9 = cm.fit_ranker(
        design9, legal9, labels[:, 9], np.arange(len(labels)), .1, 1.0, x9.shape[2]
    )
    conditional9 = {
        "model": model9, "scaler": scaler9, "mult": mult9,
        "variant": "main_style", "blocks": blocks9, "base_width": int(x9.shape[2]),
    }
    print("trained conditional R9", flush=True)

    # NIP is needed only by the selected R4/R5/R6 blends.  Refit one final
    # model per robot on train+validation after model selection was frozen.
    packed = core.pack(all_states["scenes"], "cuda")
    groups = group_ids(all_states, nip.KEYS)
    nip_states = {}
    for robot in (4, 5, 6):
        model = nip.train_model(packed, all_states["Y"][:, robot], groups,
                                robot == 4, epochs=40, seed=0)
        nip_states[robot] = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"trained NIP R{robot}", flush=True)
    del packed
    torch.cuda.empty_cache()

    conditional_results = json.loads((OUT / "conditional_results.json").read_text(encoding="utf-8"))
    temperatures = {}
    for robot in (5, 6):
        temperatures[robot] = {
            name: float(conditional_results[str(robot)]["source_calibration"][name]["full_temp"])
            for name in ("baseline_v1", "baseline_v2", "nip")
        }

    # The style classifier in policy_best was already refit on train+validation.
    production = pickle.loads((ROOT / "work" / "policy_best.pkl").read_bytes())
    bundle = {
        "version": 2,
        "selection": {
            0: "exact", 1: "v1", 2: "v1", 3: "v1", 4: "v2+nip",
            5: "calibrated .2*v1+.3*v2+.5*nip",
            6: "calibrated .3*v1+.4*v2+.3*nip",
            7: "v2", 8: "v1", 9: "conditional_main_style",
        },
        "v1_models": v1_models,
        "v2_models": v2_models,
        "nip_states": nip_states,
        "conditional9": conditional9,
        "temperatures": temperatures,
        "blend_weights": {5: {"v1": .2, "v2": .3, "nip": .5},
                          6: {"v1": .3, "v2": .4, "nip": .3}},
        "style_model": production["style_model"],
        "v1_dim": len(next(iter(feats_v1[0][1].values()))),
    }
    path = ROOT / "work" / "policy_v2_final.pkl"
    path.write_bytes(pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL))
    print(f"saved {path} ({path.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
