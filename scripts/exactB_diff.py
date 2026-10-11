"""List scenes where config A has label in argmin but B doesn't."""
import sys, pickle
import numpy as np
from exactB_common import *
robot, cond, split, wa, wb = int(sys.argv[1]), sys.argv[2], sys.argv[3], eval(sys.argv[4]), eval(sys.argv[5])
D = load(split)
F = pickle.load(open(ROOT / f"work/exactB_pareto_{split}_s6.pkl", "rb"))
def q(i, w):
    sc = D[i]
    Wv = np.array([w.get(k, 0) for k in ["E", "cr", "nc", "lm", "tr", "tl", "tu"]], float); Wv[0] = w.get("E", 100)
    fmc = {0: 0, 1: w.get("fr", 0), 2: w.get("fu", 0), 3: w.get("fl", 0)}
    return {d: (fr @ Wv).min() + fmc[relative_turn(sc["m"].heading, d)] for d, fr in F[i].items()}
out = []
for i, sc in enumerate(D):
    mi = sc["mi"]; m = sc["m"]
    if not eval(cond, {}, dict(U=mi["urgent"], F=mi["fragile"], R=m.rain, N=sc["s"]["style"] == "night")) or not F[i]:
        continue
    lab = sc["y"][robot]
    if lab in argmins(q(i, wa)) and lab not in argmins(q(i, wb)):
        out.append(i)
print(len(out), out[:60])
