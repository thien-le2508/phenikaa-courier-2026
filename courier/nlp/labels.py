"""Nhãn NLP từ scenes.json.

Với nơi giao chỉ mô tả qua bản đồ (north_most/.../anchor_near), câu thường KHÔNG cho biết loại địa điểm, nên nhãn
`goal` của NLP là None, trừ khi câu nhắc tới người nhận của đúng loại đó ("thủ thư ở chỗ trên cùng bản đồ").
Việc phát hiện "lộ loại" dùng danh sách vai người nhận + so khớp gần đúng (chịu lỗi gõ, không dấu).
"""
from courier.data import LANDMARKS
from courier.nlp import lexicon as L
from courier.nlp import lexicon3 as L3
from courier.nlp.generate import MAP_KINDS, strip_accents


def _norm(s):
    return " ".join(strip_accents(s.lower()).split())


def _role_strings(t):
    out = set()
    roles = L.pool(L.ROLES[t], True) + L3.ROLES_EXTRA[t] + L3.ROLES_VAL3.get(t, [])
    aliases = L.pool(L.ALIASES[t], True)
    for r in roles:
        if "{ALIAS}" in r:
            out |= {_norm(r.replace("{ALIAS}", a)) for a in aliases}
        else:
            out.add(_norm(r))
    return sorted(out, key=len, reverse=True)


ROLE_STR = {t: _role_strings(t) for t in LANDMARKS}


def approx_find(pat, text, k):
    """True nếu pat xuất hiện trong text với khoảng cách sửa (có hoán vị kề, Damerau) <= k (Sellers)."""
    m = len(pat)
    pp, prev = None, list(range(m + 1))
    last = ""
    for c in text:
        cur = [0]
        for j in range(1, m + 1):
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (pat[j - 1] != c))
            if pp is not None and j > 1 and c == pat[j - 2] and last == pat[j - 1]:
                v = min(v, pp[j - 2] + 1)
            cur.append(v)
        if cur[m] <= k:
            return True
        pp, prev, last = prev, cur, c
    return False


def reveals_type(text, t):
    s = _norm(text)
    for r in ROLE_STR[t]:
        k = 0 if len(r) < 8 else 1 if len(r) < 16 else 2
        if (r in s) if k == 0 else approx_find(r, s, k):
            return True
    return False


def nlp_target(m, text=None):
    """mission (scenes.json) -> nhãn cho NLP: goal None nếu mô tả qua bản đồ và không lộ loại."""
    text = text if text is not None else m["text"]
    g, v = m.get("goal_ref"), m.get("via_ref")
    goal = m["goal"]
    if g and g["kind"] in MAP_KINDS and not reveals_type(text, goal):
        goal = None
    return {"text": text, "goal": goal, "goal_true": m["goal"],
            "goal_ref": {"kind": g["kind"], "anchor": g.get("anchor")} if g else None,
            "via": m.get("via"), "via_ref": {"kind": v["kind"], "anchor": v.get("anchor")} if v else None,
            "urgent": bool(m["urgent"]), "fragile": bool(m["fragile"])}
