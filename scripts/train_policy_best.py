"""Train and save the selected A/B policy rankers and image-style model."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from courier.data import DATA, load_split  # noqa: E402
from courier.style import style_features  # noqa: E402


def candidate_xy(feats, labels, robot):
    x, y = [], []
    for scene_feat, scene_label in zip(feats, labels):
        for direction, row in scene_feat[robot].items():
            x.append(row)
            y.append(direction == scene_label[robot])
    return np.asarray(x, np.float32), np.asarray(y, np.int8)


def fit_a(x, y):
    return HistGradientBoostingClassifier(max_iter=800, learning_rate=.08, max_leaf_nodes=31,
                                          l2_regularization=1., random_state=0).fit(x, y)


def fit_b(x, y):
    return HistGradientBoostingClassifier(max_iter=600, learning_rate=.10, max_leaf_nodes=63,
                                          min_samples_leaf=5, l2_regularization=.5,
                                          random_state=0).fit(x, y)


def style_data(split):
    scenes = json.loads((DATA / split / "scenes.json").read_text(encoding="utf-8"))
    x = np.asarray([style_features(DATA / split / s["image"]) for s in scenes], np.float32)
    y = np.asarray([s["style"] for s in scenes])
    return x, y


def main():
    cache = ROOT / "work" / "policy_features_v1.pkl"
    with cache.open("rb") as f:
        data = pickle.load(f)
    with (ROOT / "work" / "policy_features_v2.pkl").open("rb") as f:
        data_v2 = pickle.load(f)
    ft, yt = data["train"]
    fv, yv = data["validation"]

    xt, st = style_data("train")
    xv, sv = style_data("validation")
    style_model = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=1,
                                       max_features=.75, n_jobs=4, random_state=0).fit(xt, st)
    print(f"style validation accuracy={np.mean(style_model.predict(xv) == sv):.4f}", flush=True)
    style_model.fit(np.concatenate((xt, xv)), np.concatenate((st, sv)))

    feats = ft + fv
    labels = np.concatenate((yt, yv))
    models_a, models_b = {}, {}
    for robot in range(1, 10):
        x, y = candidate_xy(feats, labels, robot)
        models_a[robot] = fit_a(x, y)
        models_b[robot] = fit_b(x, y)
        print(f"trained policy R{robot}", flush=True)

    # Confirmed structured upgrades: full path summaries for R7 and local
    # lookahead for R9. Keep feature masks in the bundle for exact inference.
    feats_v2 = data_v2["train"][0] + data_v2["validation"][0]
    labels_v2 = np.concatenate((data_v2["train"][1], data_v2["validation"][1]))
    v2_models = {}
    keep7 = list(range(319))
    x7, y7 = candidate_xy(
        [{7: {d: list(np.asarray(row)[keep7]) for d, row in sf[7].items()}} for sf in feats_v2],
        labels_v2, 7)
    v2_models[7] = {"mode": "b", "keep": keep7, "b": fit_b(x7, y7)}
    keep9 = list(range(106)) + list(range(290, 319))
    x9, y9 = candidate_xy(
        [{9: {d: list(np.asarray(row)[keep9]) for d, row in sf[9].items()}} for sf in feats_v2],
        labels_v2, 9)
    v2_models[9] = {"mode": "ab", "keep": keep9, "a": fit_a(x9, y9), "b": fit_b(x9, y9)}
    print("trained structured policy R7/R9", flush=True)

    bundle = {
        "version": 1,
        "models_a": models_a,
        "models_b": models_b,
        "choice": {1: "a", 2: "ab", 3: "b", 4: "a", 5: "a",
                   6: "a", 7: "a", 8: "ab", 9: "b"},
        "style_model": style_model,
        "v2_models": v2_models,
        "styles": ["classic", "night", "print", "sketch"],
    }
    out = ROOT / "work" / "policy_best.pkl"
    with out.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {out} ({out.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
