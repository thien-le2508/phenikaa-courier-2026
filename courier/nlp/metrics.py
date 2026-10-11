"""Chấm bộ phân tích yêu cầu: từng trường, khớp toàn bộ, và "đúng đích" (giải tham chiếu trên bản đồ chuẩn)."""
import math

from courier.nlp.decode import ANCHOR_KINDS, MAP_KINDS

FIELDS = ("goal", "goal_ref", "via", "via_ref", "urgent", "fragile")


def norm_ref(r):
    if not r or not r.get("kind"):
        return None
    return (r["kind"], r.get("anchor") if r["kind"] in ANCHOR_KINDS else None)


def field_eq(f, p, g):
    return norm_ref(p[f]) == norm_ref(g[f]) if f.endswith("_ref") else p[f] == g[f]


def resolve(t, ref, lms):
    """Tập rc mà (loại t, tham chiếu ref) chỉ tới trên bản đồ lms = [(type, (r, c))]."""
    kind = (ref or {}).get("kind")
    if kind in MAP_KINDS:
        if kind == "anchor_near":
            A = [rc for tt, rc in lms if tt == ref.get("anchor")]
            if not A:
                return frozenset()
            a = A[0]
            d = sorted((abs(a[0] - rc[0]) + abs(a[1] - rc[1]), rc) for tt, rc in lms if rc != a)
            return frozenset([d[0][1]]) if d else frozenset()
        key = {"north_most": lambda rc: rc[0], "south_most": lambda rc: -rc[0],
               "west_most": lambda rc: rc[1], "east_most": lambda rc: -rc[1]}[kind]
        return frozenset([min((rc for _, rc in lms), key=key)])
    if t is None:
        return frozenset()
    C = [rc for tt, rc in lms if tt == t]
    if len(C) <= 1 or not kind:
        return frozenset(C)
    if kind in ("north", "south", "west", "east"):
        key = {"north": lambda rc: rc[0], "south": lambda rc: -rc[0],
               "west": lambda rc: rc[1], "east": lambda rc: -rc[1]}[kind]
        return frozenset([min(C, key=key)])
    A = [rc for tt, rc in lms if tt == ref.get("anchor")]
    if not A:
        return frozenset(C)
    a = A[0]
    d = sorted(C, key=lambda rc: math.dist(a, rc))
    return frozenset([d[0] if kind == "near" else d[-1]])


def score(preds, golds, scenes=None):
    """preds/golds: list mission dict (golds theo nhãn NLP). scenes: để chấm đúng đích."""
    n = len(golds)
    acc = {f: sum(field_eq(f, p, g) for p, g in zip(preds, golds)) / n for f in FIELDS}
    ok = [all(field_eq(f, p, g) for f in FIELDS) for p, g in zip(preds, golds)]
    acc["exact"] = sum(ok) / n
    if scenes is not None:
        tg = []
        for p, g, s in zip(preds, golds, scenes):
            lms = [(l["type"], tuple(l["rc"])) for l in s["landmarks"]]
            m = s["mission"]
            gold_goal = frozenset([tuple(m["goal_ref"]["rc"])]) if m.get("goal_ref") else resolve(m["goal"], None, lms)
            gold_via = (frozenset([tuple(m["via_ref"]["rc"])]) if m.get("via_ref") else resolve(m["via"], None, lms)) \
                if m.get("via") else frozenset()
            pg = resolve(p["goal"], p["goal_ref"], lms)
            pv = resolve(p["via"], p["via_ref"], lms) if p["via"] else frozenset()
            tg.append(pg == gold_goal and pv == gold_via and p["urgent"] == g["urgent"] and p["fragile"] == g["fragile"])
        acc["target"] = sum(tg) / n
    return acc, ok
