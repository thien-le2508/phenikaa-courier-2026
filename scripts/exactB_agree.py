from collections import Counter
import numpy as np
from exactB_common import *
for split in ("train",):
    D = load(split)
    for r in (6, 7, 8, 9):
        for cname, cf in [("U", lambda sc: sc["mi"]["urgent"]), ("notU", lambda sc: not sc["mi"]["urgent"]), ("F", lambda sc: sc["mi"]["fragile"]), ("notF", lambda sc: not sc["mi"]["fragile"])]:
            S = [sc for sc in D if cf(sc)]
            ag = [np.mean([sc["y"][r] == sc["y"][k] for sc in S]) for k in range(10)]
            print(r, cname, len(S), " ".join(f"{a:.2f}" for a in ag))
