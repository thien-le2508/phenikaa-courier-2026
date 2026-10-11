"""Tìm đỉnh trên heatmap (stride 4) và đổi về tọa độ ảnh gốc."""
import numpy as np
import torch
import torch.nn.functional as F

from courier.cv.datasets import STRIDE


def find_peaks(prob, scale, thr=0.3, k=3):
    """prob: tensor HxW (đã sigmoid). Trả về mảng Nx3 (x, y, điểm) theo tọa độ ảnh gốc."""
    mx = F.max_pool2d(prob[None, None], k, 1, k // 2)[0, 0]
    ys, xs = torch.nonzero((prob == mx) & (prob > thr), as_tuple=True)
    P = prob.cpu().numpy()
    H, W = P.shape
    out = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0, y1, x0, x1 = max(0, y - 1), min(H, y + 2), max(0, x - 1), min(W, x + 2)
        w = P[y0:y1, x0:x1]
        gy, gx = np.mgrid[y0:y1, x0:x1]
        u = (w * (gx + 0.5)).sum() / w.sum()
        v = (w * (gy + 0.5)).sum() / w.sum()
        out.append((u * STRIDE / scale, v * STRIDE / scale, float(P[y, x])))
    return np.array(out).reshape(-1, 3)
