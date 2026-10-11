"""Dataset cho HeatmapNet và CropNet, sinh nhãn trực tiếp từ scenes.json."""
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from courier.cv import geometry as G
from courier.cv.augment import degrade
from courier.cv.labels import IGNORE, NODE_CLASSES, scene_targets

HEAT_SIZE = 768
STRIDE = 4


def load_image(path):
    return np.asarray(Image.open(path).convert("RGB"))


def letterbox(img, size=HEAT_SIZE):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = cv2.copyMakeBorder(out, 0, size - nh, 0, size - nw, cv2.BORDER_REPLICATE)
    return out, s


def draw_gaussian(hm, u, v, sigma):
    H, W = hm.shape
    r = int(3 * sigma + 1)
    ci, cj = int(np.floor(u)), int(np.floor(v))
    x0, x1, y0, y1 = max(0, ci - r), min(W, ci + r + 1), max(0, cj - r), min(H, cj + r + 1)
    if x0 >= x1 or y0 >= y1:
        return
    xs = np.arange(x0, x1) + 0.5
    ys = np.arange(y0, y1) + 0.5
    g = np.exp(-((xs[None] - u) ** 2 + (ys[:, None] - v) ** 2) / (2 * sigma ** 2))
    hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)
    if 0 <= ci < W and 0 <= cj < H:
        hm[cj, ci] = 1.0


class HeatmapDataset(Dataset):
    def __init__(self, items, train, aug=None):
        self.items, self.train = items, train  # items: [(image_path, scene)]
        self.aug = aug or {}  # tham số cho degrade (rot/blur/noise); phải nằm trong dataset để worker nhận được

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, s = self.items[i]
        rng = np.random.default_rng() if self.train else None
        img = load_image(path)
        tf = None
        if self.train:
            img, tf, _ = degrade(img, rng, **self.aug)
        t = scene_targets(s, tf)
        W = img.shape[1]
        img, sc = letterbox(img)
        n = HEAT_SIZE // STRIDE
        hm = np.zeros((3, n, n), np.float32)
        sp = G.spacing([p for p, _, _ in t["nodes"]])
        for p, _, _ in t["nodes"]:
            draw_gaussian(hm[0], p[0] * sc / STRIDE, p[1] * sc / STRIDE, max(1.0, 0.07 * sp * sc / STRIDE))
        for p, _, _ in t["legend"]:
            draw_gaussian(hm[1], p[0] * sc / STRIDE, p[1] * sc / STRIDE, max(1.0, 0.012 * W * sc / STRIDE))
        p = t["weather"][0]
        draw_gaussian(hm[2], p[0] * sc / STRIDE, p[1] * sc / STRIDE, max(1.0, 0.015 * W * sc / STRIDE))
        return torch.from_numpy(np.ascontiguousarray(img)), torch.from_numpy(hm)


class CropDataset(Dataset):
    """Mỗi phần tử = các mảnh của một cảnh (sau khi làm xấu ảnh)."""

    def __init__(self, items, train, max_plain=10, max_edges=40, aug=None):
        self.items, self.train = items, train
        self.max_plain, self.max_edges = max_plain, max_edges
        self.aug = aug or {}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, s = self.items[i]
        rng = np.random.default_rng() if self.train else np.random.default_rng(i)
        img = load_image(path)
        tf = None
        if self.train:
            img, tf, _ = degrade(img, rng, **self.aug)
        t = scene_targets(s, tf)
        W = img.shape[1]
        sp = G.spacing([p for p, _, _ in t["nodes"]])
        jit = (lambda k: rng.normal(0, k)) if self.train else (lambda k: 0.0)
        out = {k: ([], []) for k in ("node", "edge", "legend", "weather")}

        plain = [n for n in t["nodes"] if n[2] == 0]
        others = [n for n in t["nodes"] if n[2] != 0]
        if self.train:
            plain = [plain[j] for j in rng.permutation(len(plain))[:self.max_plain]]
        for p, _, c in others + plain:
            q = p + np.array([jit(0.04 * sp), jit(0.04 * sp)])
            crop = G.crop_node(img, q, sp, 1.0 + (jit(0.05) if self.train else 0))
            if self.train and c in (1, 2, 3, 4) and rng.random() < 0.5:   # xoay robot, đổi nhãn hướng
                k = int(rng.integers(1, 4))
                crop = np.rot90(crop, k).copy()
                h = NODE_CLASSES[c][6:]
                for _ in range(k):  # rot90 ngược chiều kim đồng hồ: UP->LEFT->DOWN->RIGHT->UP
                    h = {"UP": "LEFT", "LEFT": "DOWN", "DOWN": "RIGHT", "RIGHT": "UP"}[h]
                c = NODE_CLASSES.index("robot_" + h)
            out["node"][0].append(crop)
            out["node"][1].append([c])

        edges = t["edges"]
        if self.train and len(edges) > self.max_edges:
            is_rare = [e[2] in (0, 4) or e[3] == 1 or e[4] in (1, 2) for e in edges]
            rare = [e for e, r in zip(edges, is_rare) if r]
            rest = [e for e, r in zip(edges, is_rare) if not r]
            k = max(0, self.max_edges - len(rare))
            edges = rare + [rest[j] for j in rng.permutation(len(rest))[:k]]
        for a, b, look, st, ow in edges:
            a2 = a + np.array([jit(0.03 * sp), jit(0.03 * sp)])
            b2 = b + np.array([jit(0.03 * sp), jit(0.03 * sp)])
            if self.train and rng.random() < 0.5:   # đảo chiều mảnh: một chiều đổi forward<->backward
                a2, b2 = b2, a2
                ow = {0: 0, 1: 2, 2: 1}.get(ow, ow)
            out["edge"][0].append(G.crop_edge(img, a2, b2, sp))
            out["edge"][1].append([look, st, ow])

        for p, k, lk in t["legend"]:
            q = p + np.array([jit(0.004 * W), jit(0.003 * W)])
            out["legend"][0].append(G.crop_legend(img, q, W))
            out["legend"][1].append([k, lk])

        p, wl = t["weather"]
        q = p + np.array([jit(0.006 * W), jit(0.006 * W)])
        out["weather"][0].append(G.crop_weather(img, q, W))
        out["weather"][1].append([wl])
        return {k: (np.stack(v[0]), np.array(v[1], np.int64)) for k, v in out.items() if v[0]}


def collate_crops(batch):
    out = {}
    for k in ("node", "edge", "legend", "weather"):
        xs = [b[k][0] for b in batch if k in b]
        ys = [b[k][1] for b in batch if k in b]
        if xs:
            out[k] = (torch.from_numpy(np.concatenate(xs)), torch.from_numpy(np.concatenate(ys)))
    return out


def split_items(split):
    from courier.data import DATA, load_split
    _, _, scenes = load_split(split)
    return [(str(DATA / split / s["image"]), s) for s in scenes]


__all__ = ["HeatmapDataset", "CropDataset", "collate_crops", "split_items", "letterbox", "load_image", "IGNORE"]
