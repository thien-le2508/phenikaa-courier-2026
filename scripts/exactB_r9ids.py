import pickle, sys
from exactB_common import *
split = sys.argv[1]
D = load(split); F = pickle.load(open(ROOT / f"work/exactB_r9feat_{split}.pkl", "rb"))
for i in map(int, sys.argv[2].split(",")):
    sc = D[i]; print(render(sc))
    for d, f in F[i].items():
        print("  ", NAMES[d], "SRUL"[f["rel"]], {k: f[k] for k in ["man", "euc2", "bfs", "lm", "lmt", "crowded", "covered", "deg", "stairs"]}, "<-LABEL" if d == sc["y"][9] else "")
