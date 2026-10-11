import pickle, sys
from exactB_common import *
split = sys.argv[1]; expr = sys.argv[2]; n = int(sys.argv[3]); mode = sys.argv[4] if len(sys.argv) > 4 else "miss"
D = load(split); F = pickle.load(open(ROOT / f"work/exactB_r9feat_{split}.pkl", "rb"))
k = 0
for sc, fm in zip(D, F):
    legal = {d: f for d, f in fm.items() if not f["stairs"]}
    if not legal: continue
    q = {d: eval(expr, {}, dict(f, R=sc["m"].rain, U=sc["mi"]["urgent"], Fr=sc["mi"]["fragile"])) for d, f in legal.items()}
    am = argmins(q); lab = sc["y"][9]
    bad = (lab not in am) if mode == "miss" else (len(am) > 1 and lab in am)
    if not bad: continue
    print(render(sc))
    for d, f in legal.items():
        print("  ", NAMES[d], "SRUL"[f["rel"]], {k2: f[k2] for k2 in ["man", "euc2", "bfs", "lm", "lmt", "crowded", "covered", "deg", "istarget"]}, "q=", q[d], "<-LABEL" if d == lab else "")
    k += 1
    if k >= n: break
