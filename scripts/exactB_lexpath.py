"""Lexicographic key-sequence search on Pareto fronts. Row fields: E cr nc lm R L U ; move-level: fU fR fL (first move rel).
args: robot split cond prim pool maxk [gexpr]"""
import pickle, sys, itertools, os
import numpy as np
from collections import defaultdict
from exactB_common import *
robot, split, cond, prim = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
pool = sys.argv[5].split(",") if len(sys.argv) > 5 and sys.argv[5] else []
maxk = int(sys.argv[6]) if len(sys.argv) > 6 else 2
gexpr = sys.argv[7] if len(sys.argv) > 7 else "0"
D = load(split); F = pickle.load(open(ROOT / f"work/exactB_pareto_{split}_s6.pkl", "rb"))
S = []
for i, sc in enumerate(D):
    mi = sc["mi"]
    env = env_of(sc)
    if not eval(cond, {}, env) or not F[i]: continue
    S.append((i, sc["m"].heading, F[i], sc["y"][robot], env))
orders = [("rel", p) for p in itertools.permutations(range(4))] + [("abs", p) for p in itertools.permutations(range(4))]


def keyvals(expr, fr, rel, env):
    E, cr, nc, lm, R, L, U = (fr[:, k] for k in range(7))
    return eval(expr, {"np": np}, dict(env, E=E, cr=cr, nc=nc, lm=lm, tR=R, tL=L, tU=U, turns=R + L + U,
                                         fU=int(rel == 2), fR=int(rel == 1), fL=int(rel == 3), cov=E - nc)) * np.ones(len(fr))


res = []
for k in range(0, maxk + 1):
    for seq in itertools.permutations(pool, k):
        exprs = [prim] + list(seq)
        groups = defaultdict(list); miss = 0
        sets = []
        for i, h, f, lab, env in S:
            q = {}
            for d, fr in f.items():
                rel = relative_turn(h, d)
                cols = [keyvals(e, fr, rel, env) for e in exprs]
                idx = np.lexsort(cols[::-1])[0]
                q[d] = tuple(c[idx] for c in cols)
            sets.append((h, argmins(q), lab, eval(gexpr, {}, env)))
        for kind, p in orders:
            okg = defaultdict(int)
            for h, am, lab, g in sets:
                o = rel_order(h, p) if kind == "rel" else p
                okg[g] += next(d for d in o if d in am) == lab
            for g, v in okg.items():
                res.append((g, v, exprs, kind + " " + "".join(("SRUL" if kind == "rel" else "UDLR")[x] for x in p)))
        inn = sum(lab in am for h, am, lab, g in sets)
        print(exprs, "inargmin", inn, "/", len(sets), "ties", sum(len(am) > 1 for h, am, lab, g in sets), flush=True)
from collections import Counter
cnt = Counter(g for *_, g in [(0, 0, 0, eval(gexpr, {}, env)) for i, h, f, lab, env in S])
for g in sorted(cnt, key=str):
    rr = sorted([r for r in res if r[0] == g], key=lambda r: (-r[1], len(r[2])))[:4]
    print("group", g, "n", cnt[g])
    for r in rr: print("   ", r[1:])
