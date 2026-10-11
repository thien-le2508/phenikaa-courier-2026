"""Chiến thuật của 10 robot — vòng private test (v3). BẢN TẠM (interim), xem work/POLICY_NOTES.md.

Đầu vào chung (độc lập với nguồn: scenes.json hay CV/NLP dự đoán):

    graph = {
        "nodes": [(r, c), ...],
        "edges": [{"a": (r, c), "b": (r, c), "status": "normal|crowded|covered|closed",
                   "stairs": bool, "oneway_to": (r, c) | None}, ...],
        "landmarks": [(type, (r, c)), ...],
        "robot": (r, c), "heading": "UP|DOWN|LEFT|RIGHT", "rain": bool,
    }
    mission = {"goal": type | None, "goal_ref": None | {"kind", "anchor"}, "via": type | None,
               "via_ref": None | {"kind", "anchor"}, "urgent": bool, "fragile": bool}

goal_ref.kind ∈ north/south/west/east/near/far (chọn giữa 2 bản của loại `goal`)
             ∪ north_most/south_most/west_most/east_most/anchor_near (mô tả qua bản đồ; `goal` có thể None).

Chi phí là số nguyên, so hòa chính xác.
"""
import heapq
from collections import defaultdict

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DELTA = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
HEADING = {"UP": UP, "DOWN": DOWN, "LEFT": LEFT, "RIGHT": RIGHT}
CLOCKWISE = [UP, RIGHT, DOWN, LEFT]
STRAIGHT, TURN_RIGHT, U_TURN, TURN_LEFT = 0, 1, 2, 3


def relative_turn(heading, d):
    return (CLOCKWISE.index(d) - CLOCKWISE.index(heading)) % 4


def rel_order(heading, rels):
    """Thứ tự hướng tuyệt đối từ thứ tự tương đối, vd (STRAIGHT, TURN_RIGHT, TURN_LEFT, U_TURN)."""
    i = CLOCKWISE.index(heading)
    return [CLOCKWISE[(i + k) % 4] for k in rels]


SRLU = (STRAIGHT, TURN_RIGHT, TURN_LEFT, U_TURN)
RSLU = (TURN_RIGHT, STRAIGHT, TURN_LEFT, U_TURN)
LSRU = (TURN_LEFT, STRAIGHT, TURN_RIGHT, U_TURN)


# ----------------------------------------------------------------------------- bản đồ

class Map:
    def __init__(self, graph):
        self.nodes = {tuple(n) for n in graph["nodes"]}
        self.adj = defaultdict(list)  # u -> [(dir, v, status, stairs)] chỉ gồm bước hợp lệ
        for e in graph["edges"]:
            if e["status"] == "closed":
                continue
            a, b = tuple(e["a"]), tuple(e["b"])
            ow = tuple(e["oneway_to"]) if e.get("oneway_to") else None
            for u, v in ((a, b), (b, a)):
                if ow is not None and ow != v:
                    continue
                dd = (v[0] - u[0], v[1] - u[1])
                d = next((k for k, x in DELTA.items() if x == dd), None)
                if d is None:
                    continue
                self.adj[u].append((d, v, e["status"], bool(e.get("stairs"))))
        self.lm_list = [(t, tuple(rc)) for t, rc in graph["landmarks"]]
        self.landmarks = defaultdict(list)
        for t, rc in self.lm_list:
            self.landmarks[t].append(rc)
        for t in self.landmarks:
            self.landmarks[t].sort()
        self.lm_cells = {rc for _, rc in self.lm_list}
        self.start = tuple(graph["robot"])
        h = graph["heading"]
        self.heading = HEADING[h] if isinstance(h, str) else h
        self.rain = bool(graph["rain"])


# ----------------------------------------------------------------------------- tham chiếu

MOST = {"north_most": lambda p: p[0], "south_most": lambda p: -p[0],
        "west_most": lambda p: p[1], "east_most": lambda p: -p[1]}


def _man(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def resolve(ref, typ, m):
    """Danh sách ô đích cho (loại `typ`, tham chiếu `ref`).

    - north/south/west/east: bản của `typ` xa nhất về phía đó (bắc = hàng nhỏ).
    - near/far: bản của `typ` có khoảng cách Euclid (hàng, cột) tới mốc nhỏ / lớn nhất.
    - *_most: địa điểm cực trị trong TẤT CẢ địa điểm trên bản đồ (không cần `typ`; trên dữ liệu chuẩn luôn
      duy nhất). Hòa (do đọc sai ảnh) -> ưu tiên bản thuộc `typ` nếu biết.
    - anchor_near: địa điểm (khác loại mốc) gần mốc nhất theo Manhattan (không cần `typ`). Hòa -> Euclid,
      rồi ưu tiên `typ`.
    """
    cands = list(m.landmarks.get(typ, [])) if typ else []
    if not ref:
        return cands
    kind = ref.get("kind")
    if kind in MOST:
        f = MOST[kind]
        pool = m.lm_list
        if not pool:
            return cands
        best = min(f(p) for _, p in pool)
        top = sorted({p for t, p in pool if f(p) == best})
        if len(top) > 1 and typ:
            pref = [p for p in top if p in cands]
            if pref:
                top = pref
        return [top[0]]
    if kind == "anchor_near":
        anchors = m.landmarks.get(ref.get("anchor")) or []
        others = [(t, p) for t, p in m.lm_list if t != ref.get("anchor")]
        if not anchors or not others:
            return cands

        def key(tp):
            t, p = tp
            return (min(_man(p, a) for a in anchors),
                    min((p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2 for a in anchors),
                    0 if t == typ else 1, p)
        return [min(others, key=key)[1]]
    if len(cands) < 2:
        return cands
    if kind == "north":
        return [min(cands, key=lambda p: p[0])]
    if kind == "south":
        return [max(cands, key=lambda p: p[0])]
    if kind == "west":
        return [min(cands, key=lambda p: p[1])]
    if kind == "east":
        return [max(cands, key=lambda p: p[1])]
    anchors = m.landmarks.get(ref.get("anchor")) or []
    if not anchors or kind not in ("near", "far"):
        return cands

    def dist(p):
        return min((p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2 for a in anchors)
    return [min(cands, key=dist) if kind == "near" else max(cands, key=dist)]


def targets(m, mission):
    goals = resolve(mission.get("goal_ref"), mission.get("goal"), m)
    vias = None
    if mission.get("via") and (mission["via"] != mission.get("goal") or mission.get("via_ref")):
        vias = resolve(mission.get("via_ref"), mission["via"], m) or None
        if vias and set(vias) & set(goals) == set(vias):
            vias = None
    return goals, vias


# ----------------------------------------------------------------------------- Dijkstra

def first_move_costs(m, goals, vias, ecost, tcost=None, legged=False):
    """Dijkstra ngược trên trạng thái (nút, hướng, đã ghé via). ecost(status, stairs, v) -> int.
    Trả về {hướng: tổng chi phí} cho bước đầu."""
    goals, vias = set(goals), set(vias) if vias else None
    headings = range(4) if tcost else (0,)
    stages = (0, 1) if vias else (1,)
    rev = defaultdict(list)
    for u in m.nodes:
        for d, v, status, st in m.adj[u]:
            if st and not legged:
                continue
            c0 = ecost(status, st, v)
            for h in headings:
                c = c0 + (tcost(h, d) if tcost else 0)
                for s in stages:
                    s2 = 1 if (s == 1 or v in vias) else 0
                    rev[(v, d if tcost else 0, s2)].append(((u, h, s), c))
    dist = {}
    pq = [(0, (g, h, 1)) for g in goals for h in headings]
    heapq.heapify(pq)
    while pq:
        c, s = heapq.heappop(pq)
        if s in dist:
            continue
        dist[s] = c
        for p, w in rev[s]:
            if p not in dist:
                heapq.heappush(pq, (c + w, p))
    s0 = 1 if (not vias or m.start in vias) else 0
    q = {}
    for d, v, status, st in m.adj[m.start]:
        if st and not legged:
            continue
        c = ecost(status, st, v) + (tcost(m.heading, d) if tcost else 0)
        s2 = 1 if (s0 == 1 or v in vias) else 0
        rest = dist.get((v, d if tcost else 0, s2))
        if rest is not None:
            q[d] = c + rest
    return q


def pick(q, order):
    if not q:
        return None
    best = min(q.values())
    return next(d for d in order if q.get(d) == best)


def turn_cost_fn(right, left, uturn):
    table = [0, right, uturn, left]
    return lambda h, d: table[relative_turn(h, d)]


# ----------------------------------------------------------------------------- 10 robot

def old_edge(normal=10, crowded=10, covered=10, stairs=0):
    table = {"normal": normal, "crowded": crowded, "covered": covered}
    return lambda status, st, v: table[status] + (stairs if st else 0)


def r0_cost(m, avoid):
    """R0 (đã giải mã): ít đoạn nhất; hòa -> ít địa điểm (không phải đích) phải đi xuyên qua;
    rồi mưa: nhiều mái che, ít đường đông; khô: ít đường đông."""
    def ec(status, st, v):
        c = 1_000_000 + (10_000 if v in avoid else 0)
        if m.rain:
            c += (100 if status != "covered" else 0) + (1 if status == "crowded" else 0)
        else:
            c += 100 if status == "crowded" else 0
        return c
    return ec


def robot_config(r, m, mission, avoid):
    """(ecost, tcost, legged, tie_rels) của robot r."""
    W = old_edge
    if r == 0:
        if not mission.get("urgent"):
            rels = SRLU
        else:
            rels = LSRU if m.rain else RSLU
        return r0_cost(m, avoid), None, False, rels
    # ---- các robot dưới đây: vẫn là luật vòng trước (CHƯA giải mã lại) ----
    if r in (2, 4):  # tạm: luật của R0 khớp tốt hơn luật cũ (R4 vẫn đi được bậc thang)
        return r0_cost(m, avoid), None, r == 4, SRLU
    if r in LM_SECONDARY:  # tạm: luật cũ + phụ: ít địa điểm phải đi xuyên qua
        e, t, legged, rels = old_config(r, m, mission)
        return ((lambda st, s, v: 100 * e(st, s, v) + (1 if v in avoid else 0)),
                (lambda h, d: 100 * t(h, d)) if t else None, legged, rels)
    return old_config(r, m, mission)


LM_SECONDARY = {1, 3, 5, 7, 8}


def old_config(r, m, mission):
    W = old_edge
    if r == 1:
        return W(crowded=60), None, False, SRLU
    if r == 2:
        return W(covered=5), None, False, SRLU
    if r == 3:
        return (W(covered=2) if m.rain else W(crowded=30)), None, False, SRLU
    if r == 4:
        return W(crowded=20, stairs=40 if m.rain else 0), None, True, SRLU
    if r == 5:
        return W(), turn_cost_fn(30, 30, 300), False, SRLU
    if r == 6:
        return (W() if mission.get("urgent") else W(crowded=50, covered=8)), None, False, SRLU
    if r == 7:
        if not mission.get("fragile"):
            return W(), None, False, SRLU
        return W(crowded=70), turn_cost_fn(15, 15, 150), False, SRLU
    if r == 8:
        return W(), turn_cost_fn(5, 45, 90), False, SRLU
    raise ValueError(r)


def greedy_move(m, goals, vias):
    """Robot 9 (luật vòng trước): đích cố định (via nếu có), bản gần nhất theo Manhattan, bước làm giảm Manhattan."""
    cands = vias if (vias and m.start not in vias) else goals
    if not cands:
        return None
    t = min(sorted(cands), key=lambda p: _man(p, m.start))
    q = {}
    for d, v, status, st in m.adj[m.start]:
        if st:
            continue
        q[d] = 10 * _man(v, t) + (10 if status == "crowded" else 0)
    return pick(q, rel_order(m.heading, SRLU))


def fallback_move(m, legged=False):
    legal = {d for d, _, _, st in m.adj[m.start] if legged or not st}
    for d in rel_order(m.heading, SRLU):
        if d in legal:
            return d
    return m.heading


PENALTY = 50_000_000  # bước "cấm" chỉ dùng khi không còn cách nào khác (đắt hơn ~50 đoạn đường)


def relaxed_map(m, legged):
    """Bản sao của m có thêm các bước bị cấm với trạng thái "penalty" (xem PENALTY). Dùng khi đọc sai ảnh
    khiến robot không có đường tới đích."""
    r = object.__new__(Map)
    r.__dict__.update(m.__dict__)
    r.adj = defaultdict(list)
    for u in m.nodes:
        have = set()
        for d, v, status, st in m.adj[u]:
            r.adj[u].append((d, v, "penalty", False) if (st and not legged) else (d, v, status, st))
            have.add(d)
        for d, (dr, dc) in DELTA.items():
            v = (u[0] + dr, u[1] + dc)
            if d not in have and v in m.nodes:
                r.adj[u].append((d, v, "penalty", False))
    return r


def predict(m, mission, r, goals=None, vias=None):
    if goals is None:
        goals, vias = targets(m, mission)
    if r == 9:
        mv = greedy_move(m, goals, vias)
        return mv if mv is not None else fallback_move(m)
    avoid = m.lm_cells - set(goals) - set(vias or [])
    ecost, tcost, legged, rels = robot_config(r, m, mission, avoid)
    order = rel_order(m.heading, rels)
    mv = pick(first_move_costs(m, goals, vias, ecost, tcost, legged), order) if goals else None
    if mv is None and goals:
        soft = lambda status, st, v: PENALTY if status == "penalty" else ecost(status, st, v)  # noqa: E731
        mv = pick(first_move_costs(relaxed_map(m, legged), goals, vias, soft, tcost, legged), order)
    return mv if mv is not None else fallback_move(m, legged)


def predict_all(graph, mission):
    m = Map(graph)
    goals, vias = targets(m, mission)
    return [predict(m, mission, r, goals, vias) for r in range(10)]


# ----------------------------------------------------------------------------- tiện ích

def graph_from_scene(s):
    return {
        "nodes": [tuple(n["rc"]) for n in s["nodes"]],
        "edges": s["edges"],
        "landmarks": [(lm["type"], tuple(lm["rc"])) for lm in s["landmarks"]],
        "robot": tuple(s["robot"]["rc"]),
        "heading": s["robot"]["heading"],
        "rain": s["weather"] == "rain",
    }


MAP_KINDS = ("north_most", "south_most", "west_most", "east_most", "anchor_near")


def mission_from_scene(s, drop_map_goal_type=False):
    """Yêu cầu chuẩn (không kèm rc đáp án). drop_map_goal_type=True: bỏ `goal` khi nơi giao chỉ mô tả qua bản đồ
    (giả lập NLP không biết loại địa điểm)."""
    m = s["mission"]
    ref = lambda x: {"kind": x["kind"], "anchor": x["anchor"]} if x else None  # noqa: E731
    goal = m["goal"]
    if drop_map_goal_type and m["goal_ref"] and m["goal_ref"]["kind"] in MAP_KINDS:
        goal = None
    return {"goal": goal, "goal_ref": ref(m["goal_ref"]), "via": m["via"],
            "via_ref": ref(m["via_ref"]), "urgent": m["urgent"], "fragile": m["fragile"]}
