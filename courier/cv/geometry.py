"""Hình học: khoảng cách lưới, gán hàng/cột, cắt mảnh ảnh."""
import cv2
import numpy as np
from scipy.spatial import cKDTree

NODE_OUT = 80                 # mảnh giao lộ 80x80, cạnh = NODE_SIDE * spacing
NODE_SIDE = 1.1
EDGE_OUT = (48, 112)          # (cao, rộng) mảnh đoạn đường, trục a->b nằm ngang
EDGE_WIDTH = 0.45             # bề ngang mảnh đoạn đường = EDGE_WIDTH * spacing
LEGEND_OUT = (56, 256)        # mảnh dòng chú giải: ô mẫu + chữ
LEGEND_BOX = (-0.04, -0.03, 0.25, 0.03)  # so với tâm ô mẫu, theo bề rộng ảnh
WEATHER_OUT = 64
WEATHER_SIDE = 0.09           # theo bề rộng ảnh


def spacing(points):
    """Khoảng cách lưới ước lượng = trung vị khoảng cách tới điểm gần nhất."""
    P = np.asarray(points, float)
    if len(P) < 2:
        return 100.0
    d, _ = cKDTree(P).query(P, k=2)
    return float(np.median(d[:, 1]))


def grid_angle(P, sp):
    T = cKDTree(P)
    d, i = T.query(P, k=min(5, len(P)))
    vec = (P[i[:, 1:]] - P[:, None]).reshape(-1, 2)
    dist = np.linalg.norm(vec, axis=1)
    vec = vec[(dist > 0) & (dist < 1.4 * sp)]
    if len(vec) == 0:
        return 0.0
    ang = np.arctan2(vec[:, 1], vec[:, 0])
    return float(np.angle(np.mean(np.exp(4j * ang))) / 4)


def assign_rc(points):
    """Gán (hàng, cột) cho các tâm giao lộ: ước lượng góc xoay, xoay thẳng, tách cụm theo khoảng trống."""
    P = np.asarray(points, float)
    sp = spacing(P)
    a = grid_angle(P, sp)
    c, s = np.cos(-a), np.sin(-a)
    Q = P @ np.array([[c, -s], [s, c]]).T

    def cluster(v):
        o = np.argsort(v)
        lab = np.zeros(len(v), int)
        k = 0
        for j in range(1, len(o)):
            if v[o[j]] - v[o[j - 1]] > 0.5 * sp:
                k += 1
            lab[o[j]] = k
        return lab
    return list(zip(cluster(Q[:, 1]).tolist(), cluster(Q[:, 0]).tolist())), sp


def _warp(img, M, out_w, out_h):
    return cv2.warpAffine(img, M, (out_w, out_h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_REPLICATE)


def crop_rect(img, x0, y0, x1, y1, out_w, out_h):
    sx, sy = (x1 - x0) / out_w, (y1 - y0) / out_h
    M = np.array([[sx, 0, x0 + 0.5 * sx], [0, sy, y0 + 0.5 * sy]], np.float32)
    return _warp(img, M, out_w, out_h)


def crop_node(img, c, sp, scale=1.0):
    h = NODE_SIDE * sp * scale / 2
    return crop_rect(img, c[0] - h, c[1] - h, c[0] + h, c[1] + h, NODE_OUT, NODE_OUT)


def crop_edge(img, a, b, sp):
    """Mảnh dọc đoạn a->b: trục ngang của mảnh đi từ tâm a tới tâm b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d)) or 1.0
    u = d / L
    n = np.array([-u[1], u[0]])
    oh, ow = EDGE_OUT
    w = EDGE_WIDTH * sp
    sx, sy = L / ow, w / oh
    o = a + u * 0.5 * sx + n * (0.5 * sy - w / 2)
    M = np.array([[u[0] * sx, n[0] * sy, o[0]], [u[1] * sx, n[1] * sy, o[1]]], np.float32)
    return _warp(img, M, ow, oh)


def crop_legend(img, c, W):
    x0, y0, x1, y1 = LEGEND_BOX
    return crop_rect(img, c[0] + x0 * W, c[1] + y0 * W, c[0] + x1 * W, c[1] + y1 * W, LEGEND_OUT[1], LEGEND_OUT[0])


def crop_weather(img, c, W):
    h = WEATHER_SIDE * W / 2
    return crop_rect(img, c[0] - h, c[1] - h, c[0] + h, c[1] + h, WEATHER_OUT, WEATHER_OUT)
