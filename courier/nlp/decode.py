"""Giải mã đầu ra NLP (v3) thành mission, có thể ràng buộc theo các loại địa điểm có trên bản đồ.

mission = {"goal": type|None, "goal_ref": None|{"kind", "anchor"}, "via": type|None, "via_ref": None|{"kind","anchor"},
           "urgent": bool, "fragile": bool}
  * goal None chỉ khi goal_ref.kind thuộc MAP_KINDS (nơi giao chỉ mô tả qua bản đồ, câu không lộ loại).
  * kind north/south/west/east/near/far chỉ khi loại nơi giao có 2 bản trên bản đồ; mốc near/far là loại có 1 bản.
  * anchor_near: mốc phải có trên bản đồ (ưu tiên loại có 1 bản).
"""
import math
from collections import Counter

from courier.nlp.model import HEADS

MAP_KINDS = {"north_most", "south_most", "west_most", "east_most", "anchor_near"}
ANCHOR_KINDS = {"near", "far", "anchor_near"}


def _lg(p):
    return math.log(max(p, 1e-12))


def _best(probs, labels, allowed):
    best, bs = None, -math.inf
    for lab, p in zip(labels, probs):
        if lab in allowed and _lg(p) > bs:
            best, bs = lab, _lg(p)
    return best, bs


def decode(p, landmark_types=None):
    """p: dict head -> list xác suất. landmark_types: list loại địa điểm trên bản đồ (có lặp) hoặc None."""
    counts = Counter(landmark_types or [])
    constrained = bool(counts)
    goals, kinds, anchors = HEADS["goal"], HEADS["goal_kind"], HEADS["goal_anchor"]
    on_map = set(counts) if constrained else set(goals) - {None}

    # ---- nơi giao + tham chiếu (giải mã đồng thời)
    best, best_s = None, -math.inf
    for gi, g in enumerate(goals):
        if g is not None and g not in on_map:
            continue
        for ki, k in enumerate(kinds):
            if g is None and k not in MAP_KINDS:
                continue
            if k is not None and k not in MAP_KINDS and constrained and counts[g] < 2:
                continue
            s = _lg(p["goal"][gi]) + _lg(p["goal_kind"][ki])
            a = None
            if k in ANCHOR_KINDS:
                if k == "anchor_near":
                    allowed = {t for t in on_map if counts.get(t, 1) == 1} or set(on_map)
                    allowed -= {g}
                else:
                    allowed = {t for t in on_map if counts.get(t, 1) == 1 and t != g}
                if not allowed:
                    continue
                a, sa = _best(p["goal_anchor"], anchors, allowed)
                s += sa
            if s > best_s:
                best, best_s = (g, k, a), s
    if best is None:  # không có phương án hợp lệ (CV sai?) -> bỏ ràng buộc
        return decode(p, None) if constrained else None
    goal, kind, anchor = best
    goal_ref = None if kind is None else {"kind": kind, "anchor": anchor}

    # ---- điểm ghé
    vias = HEADS["via"]
    allowed_v = {None} | ({t for t in on_map if t != goal})
    via, _ = _best(p["via"], vias, allowed_v)
    via_ref = None
    if via is not None and (not constrained or counts[via] >= 2):
        vk = HEADS["via_kind"]
        cand = []
        for ki, k in enumerate(vk):
            s = _lg(p["via_kind"][ki])
            a = None
            if k in ("near", "far"):
                allowed = {t for t in on_map if counts.get(t, 1) == 1 and t != via}
                if not allowed:
                    continue
                a, sa = _best(p["via_anchor"], anchors, allowed)
                s += sa
            cand.append((s, k, a))
        _, k, a = max(cand, key=lambda x: x[0])
        if k is not None:
            via_ref = {"kind": k, "anchor": a}
    return {"goal": goal, "goal_ref": goal_ref, "via": via, "via_ref": via_ref,
            "urgent": p["urgent"][1] > 0.5, "fragile": p["fragile"][1] > 0.5}


def hypotheses(p, landmark_types=None, **_):
    """Tương thích ngược (courier/vote.py): chỉ trả về phương án tốt nhất."""
    return [(1.0, decode(p, landmark_types))]
