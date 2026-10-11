import pickle, sys, itertools
from collections import defaultdict
from exactB_common import *
split, cond, expr = sys.argv[1], sys.argv[2], sys.argv[3]
gexpr = sys.argv[4] if len(sys.argv) > 4 else "0"
secs = ["0", "crowded", "-crowded", "covered", "-covered", "lm", "-lm", "deg", "-deg", "euc2", "-euc2", "bfs", "-bfs", "istarget"]
D = load(split); F = pickle.load(open(ROOT / f"work/exactB_r9feat_{split}.pkl", "rb"))
groups = defaultdict(list); miss = []
for sc, fm in zip(D, F):
    mi = sc["mi"]
    env = dict(U=mi["urgent"], F=mi["fragile"], R=sc["m"].rain, N=sc["s"]["style"] == "night", V=bool(sc["vias"]), H=sc["m"].heading, S=sc["s"]["style"])
    if not eval(cond, {}, env): continue
    legal = {d: f for d, f in fm.items() if not f["stairs"]}
    if not legal: continue
    q = {d: eval(expr, {}, dict(f, **env)) for d, f in legal.items()}
    am = argmins(q); lab = sc["y"][9]
    if lab not in am: miss.append(sc["i"]); continue
    if len(am) > 1: groups[eval(gexpr, {}, env)].append((sc["m"].heading, {d: legal[d] for d in am}, lab))
print("miss", len(miss), miss[:30])
for g, L in sorted(groups.items(), key=str):
    res = []
    for sec in secs:
        for perm in itertools.permutations(range(4)):
            okr = oka = 0
            for h, am, lab in L:
                sv = {d: eval(sec, {}, f) for d, f in am.items()}
                mn = min(sv.values()); am2 = {d for d in am if sv[d] == mn}
                okr += next(d for d in rel_order(h, perm) if d in am2) == lab
                oka += next(d for d in perm if d in am2) == lab
            res.append((okr, sec, "rel " + "".join("SRUL"[k] for k in perm)))
            res.append((oka, sec, "abs " + "".join("UDLR"[k] for k in perm)))
    res.sort(key=lambda x: -x[0])
    print("group", g, "ties", len(L), res[:5])
