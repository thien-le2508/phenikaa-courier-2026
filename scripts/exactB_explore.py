import sys
from collections import Counter
from exactB_common import *
for split in ("train", "validation"):
    D = load(split)
    for r in (6, 7, 8, 9):
        c = Counter()
        for sc in D:
            mi = sc["mi"]
            key = (sc["m"].rain, mi["urgent"], mi["fragile"])
            c[key, sc["y"][r] == sc["y"][0]] += 1
        print(split, r, sorted(c.items()))
