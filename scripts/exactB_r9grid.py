import pickle, sys, itertools
import numpy as np
from exactB_common import *
split, cond = sys.argv[1], sys.argv[2]
grid = eval(sys.argv[3]); top = int(sys.argv[4]) if len(sys.argv) > 4 else 15
D = load(split); F = pickle.load(open(ROOT / f"work/exactB_r9feat_{split}.pkl", "rb"))
KEYS = ["man", "lm", "crowded", "covered", "fR", "fL", "fU", "istarget", "bfs", "euc2", "deg"]
rows, seg, labs = [], [], []
for i, (sc, fm) in enumerate(zip(D, F)):
    mi = sc["mi"]
    if not eval(cond, {}, dict(U=mi["urgent"], F=mi["fragile"], R=sc["m"].rain, N=sc["s"]["style"] == "night", V=bool(sc["vias"]))): continue
    legal = {d: f for d, f in fm.items() if not f["stairs"]}
    if not legal: continue
    for d, f in legal.items():
        rows.append([f["man"], f["lm"], f["crowded"], f["covered"], f["rel"] == 1, f["rel"] == 3, f["rel"] == 2, f["istarget"], min(f["bfs"], 99), f["euc2"], f["deg"]])
        seg.append(len(labs)); 
    labs.append(None)
    # mark label rows
X = np.array(rows, float)
seg = np.array(seg)
# recompute labels per row
lab_row = []
j = 0
for i, (sc, fm) in enumerate(zip(D, F)):
    mi = sc["mi"]
    if not eval(cond, {}, dict(U=mi["urgent"], F=mi["fragile"], R=sc["m"].rain, N=sc["s"]["style"] == "night", V=bool(sc["vias"]))): continue
    legal = {d: f for d, f in fm.items() if not f["stairs"]}
    if not legal: continue
    for d in legal: lab_row.append(d == sc["y"][9])
lab_row = np.array(lab_row)
starts = np.r_[0, np.flatnonzero(np.diff(seg)) + 1]
combos = list(itertools.product(*[grid.get(k, [0]) for k in KEYS]))
W = np.array(combos, float).T
S = X @ W
best = np.minimum.reduceat(S, starts, axis=0)
eq = S <= best[seg] + 1e-9
nt = np.add.reduceat(eq.astype(int), starts, axis=0)
ok = np.zeros(best.shape, bool); ok[seg[lab_row]] = eq[lab_row]
a = ok.mean(0); u = (ok & (nt == 1)).mean(0)
print("scenes", len(starts))
for j in np.lexsort((-u, -a))[:top]:
    print({k: v for k, v in zip(KEYS, combos[j]) if v}, f"in={a[j]:.4f} uni={u[j]:.4f}")
