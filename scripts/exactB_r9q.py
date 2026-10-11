import pickle, sys
from collections import Counter
from exactB_common import *
split = sys.argv[1]; expr = sys.argv[2]
D = load(split); F = pickle.load(open(ROOT / f"work/exactB_r9feat_{split}.pkl", "rb"))
c = Counter()
for sc, fm in zip(D, F):
    legal = {d: f for d, f in fm.items() if not f["stairs"]}
    if not legal: continue
    q = {d: eval(expr, {}, dict(f, R=sc["m"].rain, U=sc["mi"]["urgent"], Fr=sc["mi"]["fragile"])) for d, f in legal.items()}
    am = argmins(q); lab = sc["y"][9]
    c["in"] += lab in am; c["uni"] += am == {lab}; c["n"] += 1
print(expr, f"in={c['in']/c['n']:.4f} uni={c['uni']/c['n']:.4f}")
