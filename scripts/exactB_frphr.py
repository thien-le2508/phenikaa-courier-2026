import re
from collections import Counter
from exactB_common import *
D = load("train")
cf, cn = Counter(), Counter()
for sc in D:
    t = strip_acc(sc["s"]["mission"]["text"])
    for sent in re.split(r"[.!?;()]", t):
        sent = sent.strip()
        if sent: (cf if sc["mi"]["fragile"] else cn)[sent] += 1
tot = cf + cn
for s, c in tot.most_common(300):
    if c >= 5 and (cf[s] / c > 0.9 or cf[s] / c < 0.1) and len(s) < 60 and not any(k in s for k in URG_PHR):
        print(f"{cf[s]:4d} {cn[s]:4d}  {s}")
