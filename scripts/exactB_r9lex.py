"""Search lexicographic key sequences + tie order for R9 in a condition group."""
import pickle, sys, itertools
from exactB_common import *
split, cond = sys.argv[1], sys.argv[2]
prim = sys.argv[3] if len(sys.argv) > 3 else "man"
pool = (sys.argv[4] if len(sys.argv) > 4 else "euc2,lm,crowded,-crowded,covered,-covered,deg,-deg,lmany,istarget,bfs").split(",")
maxk = int(sys.argv[5]) if len(sys.argv) > 5 else 3
robot = int(os.environ.get("ROBOT", 9))
D = load(split); F = pickle.load(open(ROOT / f"work/exactB_r9feat_{split}.pkl", "rb"))
S = []
for sc, fm in zip(D, F):
    mi = sc["mi"]
    env = dict(U=mi["urgent"], F=mi["fragile"], R=sc["m"].rain, N=sc["s"]["style"] == "night", V=bool(sc["vias"]))
    if not eval(cond, {}, env): continue
    legal = {d: f for d, f in fm.items() if not f["stairs"]}
    if not legal: continue
    S.append((sc["m"].heading, legal, sc["y"][robot], env))
orders = [("rel", p) for p in itertools.permutations(range(4))] + [("abs", p) for p in itertools.permutations(range(4))]
res = []
for k in range(0, maxk + 1):
    for seq in itertools.permutations(pool, k):
        exprs = [prim] + list(seq)
        keyf = [compile(e, "k", "eval") for e in exprs]
        sets = []
        for h, legal, lab, env in S:
            q = {d: tuple(eval(c, {}, dict(f, **env)) for c in keyf) for d, f in legal.items()}
            sets.append((h, argmins(q), lab))
        for kind, p in orders:
            ok = 0
            for h, am, lab in sets:
                o = rel_order(h, p) if kind == "rel" else p
                ok += next(d for d in o if d in am) == lab
            res.append((ok, exprs, kind + " " + "".join(("SRUL" if kind == "rel" else "UDLR")[x] for x in p)))
res.sort(key=lambda x: (-x[0], len(x[1])))
print("n", len(S))
for r in res[:12]: print(r)
