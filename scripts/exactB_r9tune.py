import sys, json, itertools
sys.path.insert(0, "scripts")
from exactB_common import *
import courier.exact.r9 as R9
splits = {s: load(s) for s in ("train", "validation")}
for sp in splits.values():
    for sc in sp:
        sc["m"].style = sc["s"]["style"]
grid = eval(sys.argv[1])
keys = list(grid)
for combo in itertools.product(*grid.values()):
    for k, v in zip(keys, combo):
        setattr(R9, k, v)
    out = []
    for s, D in splits.items():
        fails = [sc["i"] for sc in D if R9.move(sc["m"], sc["mi"], sc["goals"], sc["vias"]) != sc["y"][9]]
        out.append(f"{s} {1 - len(fails) / len(D):.4f} {fails[:12]}")
    print(dict(zip(keys, combo)), " | ".join(out), flush=True)
