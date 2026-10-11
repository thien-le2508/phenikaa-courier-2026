import re, json, unicodedata
from collections import Counter
from exactB_common import *
def strip(s):
    s = unicodedata.normalize("NFD", s.lower()); s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d")
D = load("train")
cu, cn = Counter(), Counter()
for sc in D:
    t = strip(sc["s"]["mission"]["text"])
    for sent in re.split(r"[.!?;()]", t):
        sent = sent.strip()
        if not sent: continue
        (cu if sc["mi"]["urgent"] else cn)[sent] += 1
tot = cu + cn
for s, c in tot.most_common(400):
    if c >= 5 and (cu[s] == 0 or cn[s] == 0 or True):
        r = cu[s] / c
        if r > 0.9 or r < 0.1:
            if len(s) < 60: print(f"{cu[s]:4d} {cn[s]:4d}  {s}")
