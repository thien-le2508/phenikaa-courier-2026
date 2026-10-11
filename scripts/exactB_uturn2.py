import sys, pickle
import numpy as np
from collections import Counter
from exactB_common import *
robot = int(sys.argv[1]); split = "train"
D = load(split); F = pickle.load(open(ROOT / f"work/exactB_pareto_{split}_s6.pkl", "rb"))
rows = []
for i, sc in enumerate(D):
    m = sc["m"]; f = F[i]
    if not f: continue
    U = [d for d in f if relative_turn(m.heading, d) == 2]
    N = [d for d in f if relative_turn(m.heading, d) != 2]
    if not U or not N: continue
    eu = f[U[0]][:, 0].min(); en = min(f[d][:, 0].min() for d in N)
    if en - eu != 2: continue
    def summ(ds, E):
        best = None
        for d in ds:
            fr = f[d]; fr = fr[fr[:, 0] == E] if (fr[:, 0] == E).any() else None
            if fr is None: continue
            v = fr[np.lexsort((fr[:, 1], fr[:, 2], fr[:, 3]))][0]  # lm, nc, cr
            best = v if best is None or tuple(v[[3, 2, 1]]) < tuple(best[[3, 2, 1]]) else best
        return best
    a = summ(U, eu); b = summ(N, en)
    rows.append((sc["y"][robot] == U[0], a, b, sc["i"]))
c = Counter()
for lab, a, b, i in rows:
    c[("lm", int(np.sign(a[3] - b[3]))), lab] += 1
    c[("cr", int(np.sign(a[1] - b[1]))), lab] += 1
    c[("nc", int(np.sign(a[2] - b[2]))), lab] += 1
    c[("turns", int(np.sign(a[4:].sum() - b[4:].sum()))), lab] += 1
for k in sorted({k for k, _ in c}):
    print(k, "U chosen", c[k, True], "not", c[k, False])
