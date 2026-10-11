"""Làm xấu ảnh giống ảnh chụp/quét (xoay, đổi cỡ, mờ, JPEG), kèm biến đổi tọa độ tương ứng."""
import cv2
import numpy as np


STRONG = {"rot": 4.0, "blur": 1.6, "noise": 0.0}  # mức mặc định (các model đầu tiên)


def degrade(img, rng, strength=1.0, rot=None, blur=None, noise=None, scale_lo=0.7, p_blur=0.45):
    """Trả về (ảnh mới, hàm biến đổi điểm). Mạnh hơn train để bao được validation/test.
    rot/blur/noise: giới hạn trên của góc xoay, sigma làm mờ, sigma nhiễu hạt (mặc định lấy từ STRONG)."""
    rot = STRONG["rot"] if rot is None else rot
    blur = STRONG["blur"] if blur is None else blur
    noise = STRONG["noise"] if noise is None else noise
    h, w = img.shape[:2]
    M = np.eye(3)
    if rng.random() < 0.7 * strength:
        ang = rng.uniform(-rot, rot)
        R = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        img = cv2.warpAffine(img, R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        M = np.vstack([R, [0, 0, 1]]) @ M
    if rng.random() < 0.5 * strength:
        s = rng.uniform(scale_lo, 1.1)
        nw, nh = int(round(w * s)), int(round(h * s))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        M = np.diag([nw / w, nh / h, 1]) @ M
    if rng.random() < p_blur * strength:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, blur))
    if noise and rng.random() < 0.3 * strength:
        img = np.clip(img.astype(np.float32) + rng.normal(0, rng.uniform(2, noise), img.shape), 0, 255).astype(np.uint8)
    if rng.random() < 0.3 * strength:
        img = np.clip(img.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.uniform(-12, 12), 0, 255).astype(np.uint8)
    if rng.random() < 0.5 * strength:
        ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, int(rng.uniform(30, 95))])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
    img = np.ascontiguousarray(img)

    def tf(pts):
        P = np.asarray(pts, float).reshape(-1, 2)
        Q = np.c_[P, np.ones(len(P))] @ M.T
        return Q[:, :2]
    return img, tf, M
