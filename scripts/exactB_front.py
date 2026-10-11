import sys, pickle
from exactB_common import *
split = sys.argv[1]
D = load(split)
F = pickle.load(open(ROOT / f"work/exactB_pareto_{split}_s6.pkl", "rb"))
for i in map(int, sys.argv[2].split(",")):
    sc = D[i]
    print(render(sc))
    print("text:", sc["s"]["mission"]["text"], "| style", sc["s"]["style"])
    for d, fr in sorted(F[i].items()):
        fr = sorted(map(tuple, fr))[:8]
        print(" ", NAMES[d], "rel", "SRUL"[relative_turn(sc["m"].heading, d)], "front[E,cr,nc,lm,R,L,U]:", fr)
    print()
