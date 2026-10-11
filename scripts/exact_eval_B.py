"""EXACT-B evaluation: per-robot accuracy for R6..R9 modules on GT graph + GT mission (refs resolved by exact resolver).
    PYTHONIOENCODING=utf-8 COURIER_DATA=delivery_public python scripts/exact_eval_B.py [robots] [--fails N]
"""
import importlib, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from courier.policy import Map, graph_from_scene, mission_from_scene, targets, fallback_move  # noqa

DATA = ROOT / os.environ.get("COURIER_DATA", "delivery_public")


def run(robots, nfail=20):
    for split in ("train", "validation"):
        S = json.loads((DATA / split / "scenes.json").read_text(encoding="utf-8"))
        Y = json.loads((DATA / split / "labels.json").read_text(encoding="utf-8"))
        for r in robots:
            mod = importlib.import_module(f"courier.exact.r{r}")
            ok, fails = 0, []
            for i, s in enumerate(S):
                g = graph_from_scene(s)
                g["style"] = s.get("style")
                mi = mission_from_scene(s)
                m = Map(g)
                m.style = s.get("style")
                goals, vias = targets(m, mi)
                p = mod.move(m, mi, goals, vias)
                if p is None:
                    p = fallback_move(m)
                if p == Y[10 * i + r]:
                    ok += 1
                else:
                    fails.append(i)
            print(f"{split:10s} R{r}: {ok / len(S):.4f} ({len(S) - ok} fails) {fails[:nfail]}", flush=True)


if __name__ == "__main__":
    rs = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else [6, 7, 8, 9]
    nf = int(sys.argv[sys.argv.index("--fails") + 1]) if "--fails" in sys.argv else 20
    run(rs, nf)
