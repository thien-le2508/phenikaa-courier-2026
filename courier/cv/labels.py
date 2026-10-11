"""Danh sách lớp và cách đổi scenes.json thành nhãn huấn luyện cho CV."""
import numpy as np

from courier.data import LANDMARKS

HEADINGS = ["UP", "DOWN", "LEFT", "RIGHT"]
NODE_CLASSES = ["plain"] + ["robot_" + h for h in HEADINGS] + LANDMARKS            # 15
LOOKS = ["normal", "crowded", "covered"]                                            # kiểu nét mặc định
EDGE_LOOK = ["none"] + LOOKS + ["closed"]                                           # 5
ONEWAY = ["none", "forward", "backward"]                                            # theo chiều a->b của mảnh
LEGEND_KINDS = LANDMARKS + ["weather", "normal", "crowded", "covered", "closed", "stairs", "oneway", "robot"]  # 18
WEATHER = ["dry", "rain"]
IGNORE = -100


def look_of(status, road_look):
    if status == "closed":
        return "closed"
    return road_look.get(status, status)


def scene_targets(s, tf=None):
    """Nhãn của một cảnh, tọa độ đã qua biến đổi tf (nếu có)."""
    tf = tf or (lambda p: np.asarray(p, float).reshape(-1, 2))
    rcs = [tuple(n["rc"]) for n in s["nodes"]]
    xy = dict(zip(rcs, tf([n["xy"] for n in s["nodes"]])))
    cls = {rc: 0 for rc in rcs}
    for lm in s["landmarks"]:
        cls[tuple(lm["rc"])] = NODE_CLASSES.index(lm["type"])
    cls[tuple(s["robot"]["rc"])] = NODE_CLASSES.index("robot_" + s["robot"]["heading"])
    nodes = [(xy[rc], rc, cls[rc]) for rc in rcs]

    emap = {}
    for e in s["edges"]:
        a, b = tuple(e["a"]), tuple(e["b"])
        emap[(a, b) if a < b else (b, a)] = e
    edges = []
    for a in rcs:
        for b in ((a[0], a[1] + 1), (a[0] + 1, a[1])):
            if b not in xy:
                continue
            e = emap.get((a, b))
            if e is None:
                edges.append((xy[a], xy[b], 0, IGNORE, IGNORE))
                continue
            look = EDGE_LOOK.index(look_of(e["status"], s["road_look"]))
            ow = e["oneway_to"]
            ow = 0 if ow is None else (1 if tuple(ow) == b else 2)
            edges.append((xy[a], xy[b], look, int(e["stairs"]), ow))

    legend = []
    for l in s["legend"]:
        x0, y0, x1, y1 = l["swatch"]
        kind = l["kind"].split(":")[-1]
        k = LEGEND_KINDS.index(kind)
        lk = LOOKS.index(look_of(kind, s["road_look"])) if kind in LOOKS else IGNORE
        legend.append((tf([(x0 + x1) / 2, (y0 + y1) / 2])[0], k, lk))
    x0, y0, x1, y1 = s["weather_box"]
    weather = (tf([(x0 + x1) / 2, (y0 + y1) / 2])[0], WEATHER.index("rain" if s["weather"] == "rain" else "dry"))
    return {"nodes": nodes, "edges": edges, "legend": legend, "weather": weather}
