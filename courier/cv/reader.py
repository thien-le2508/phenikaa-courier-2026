"""Đọc ảnh bản đồ thành đồ thị cho courier.policy.

Các bước sửa lỗi lúc dự đoán (đo trên validation với model chỉ học train: 295/300 -> 300/300 cảnh đúng hoàn toàn,
và bền hơn hẳn khi ảnh mờ/xoay/nhiễu thêm):
  - lọc đỉnh lẻ loi, gộp đỉnh trùng ô, nhặt lại đỉnh yếu lấp ô trống của lưới (giao lộ "vô hình" kiểu print)
  - làm nét mảnh cắt khi ảnh quá mờ (chỉ bật khi chỉ số độ nét thấp hơn mọi ảnh train/val)
  - TTA đảo chiều mảnh đoạn đường; nghiêng nhẹ về "có một chiều / có bậc thang"
  - tinh chỉnh kiểu nét đoạn đường theo "nguyên mẫu" của chính ảnh (các đoạn chắc chắn trong cùng ảnh)
  - ràng buộc mềm loại địa điểm theo chú giải
  - sửa chỉ số hàng/cột khi có hàng/cột giả ở rìa (vd. chữ tiêu đề bị nhận nhầm là giao lộ)
"""
import itertools

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from courier.cv import geometry as G
from courier.cv.datasets import letterbox, load_image
from courier.cv.labels import EDGE_LOOK, HEADINGS, LEGEND_KINDS, LOOKS, NODE_CLASSES, ONEWAY, WEATHER
from courier.cv.models import TASK_HEADS, CropNet, HeatmapNet, normalize
from courier.cv.peaks import find_peaks
from courier.data import LANDMARKS

STATUSES = ["normal", "crowded", "covered"]
DEFAULT_OPTS = {
    "iso_filter": 1.6,          # bỏ đỉnh cách đỉnh gần nhất > 1.6 x khoảng cách lưới
    "mid_filter": True,         # bỏ đỉnh nằm giữa hai giao lộ kề nhau (ký hiệu giữa đoạn đường)
    "sharpen": (1.0, 1.5),      # unsharp mask (sigma, amount) cho mảnh cắt khi ảnh quá mờ
    "sharpen_auto": 0.115,      # ngưỡng chỉ số độ nét (mọi ảnh train/val đều > 0.135)
    "edge_rev": True,
    "edge_proto": True,
    "ow_bias": 0.4,             # nhân xác suất "không một chiều" / "không bậc thang"
    "st_bias": 0.4,
    "compact": True,
    "reassign": True,
    "node_scales": (1.0,),      # TTA: node crops at these scales, probabilities averaged
    "legend_assign": True,      # legend rows -> kinds by one-to-one assignment (no duplicate kinds)
}


def blur_metric(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gm = np.sqrt(cv2.Sobel(g, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1) ** 2)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    return float(np.percentile(lap, 99.9) / (np.percentile(gm, 99.9) + 1))


def _margin(p):
    s = np.sort(p, -1)
    return s[..., -1] - s[..., -2]


def proto_refine(p, f, hi=0.6, lo=0.6, min_n=3, T=0.05, w=0.5, only=(1, 2, 3)):
    """Kiểu nét đoạn đường: đoạn kém chắc chắn được chấm lại theo độ giống (cosine) với đặc trưng trung bình
    của các đoạn chắc chắn cùng lớp trong CHÍNH ảnh đó (quy ước vẽ của từng ảnh)."""
    m, am = _margin(p), p.argmax(1)
    protos, cls = [], []
    for c in only:
        sel = (am == c) & (m >= hi)
        if sel.sum() >= min_n:
            v = f[sel].mean(0)
            protos.append(v / np.linalg.norm(v))
            cls.append(c)
    if len(cls) < 2:
        return p
    S = f @ np.stack(protos).T
    q = np.exp((S - S.max(1, keepdims=True)) / T)
    q /= q.sum(1, keepdims=True)
    out = p.copy()
    low = (m < lo) & np.isin(am, only)
    mass = p[:, cls].sum(1, keepdims=True)
    sub = (1 - w) * p[:, cls] + w * mass * q
    out[np.ix_(low, cls)] = sub[low]
    return out


class MapReader:
    def __init__(self, heat_ckpt, crop_ckpt, device="cuda", recover_thr=0.05, legend_prior=0.5, opts=None):
        self.device = device
        self.recover_thr = recover_thr
        self.legend_prior = legend_prior
        self.opts = dict(DEFAULT_OPTS, **(opts or {}))
        self.legend_prior = self.opts.pop("legend_prior", legend_prior)
        self.recover_thr = self.opts.pop("recover_thr", recover_thr)
        self.heat = HeatmapNet(pretrained=False)
        self.heat.load_state_dict(torch.load(heat_ckpt, map_location="cpu", weights_only=True))
        self.heat.to(device).eval()
        self.crops = []  # one or several CropNets (ensemble: probabilities averaged)
        for ck in ([crop_ckpt] if isinstance(crop_ckpt, (str, bytes)) or hasattr(crop_ckpt, "__fspath__") else crop_ckpt):
            m = CropNet(pretrained=False)
            m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
            self.crops.append(m.to(device).eval())
        self.crop = self.crops[0]

    def n_params(self):
        return sum(p.numel() for m in [self.heat] + self.crops for p in m.parameters())

    @torch.no_grad()
    def _classify(self, crops, task, bs=96):
        """Trả về ({head: xác suất}, đặc trưng chuẩn hóa)."""
        if len(crops) == 0:
            return {}, None
        probs, feats = [], []
        k = len(self.crops)
        for i in range(0, len(crops), bs):
            x = normalize(torch.from_numpy(np.stack(crops[i:i + bs])).to(self.device))
            pr, fs = {}, []
            for m in self.crops:
                with torch.autocast(self.device.split(":")[0], dtype=torch.float16, enabled=self.device != "cpu"):
                    f = F.adaptive_avg_pool2d(m.trunk(x), 1).flatten(1) + m.task_emb[task]
                    out = {h: m.heads[h](f) for h in TASK_HEADS[task]}
                for h, v in out.items():
                    pr[h] = pr.get(h, 0) + torch.softmax(v.float(), -1).cpu().numpy() / k
                fs.append(F.normalize(f.float(), dim=1).cpu().numpy() / np.sqrt(k))
            probs.append(pr)
            feats.append(np.concatenate(fs, 1))
        return {h: np.concatenate([p[h] for p in probs]) for h in probs[0]}, np.concatenate(feats)

    def read(self, path):
        return self.read_img(load_image(path))

    @torch.no_grad()
    def read_img(self, img, force_P=None):
        o = self.opts
        img_in = img
        H, W = img.shape[:2]
        crop_img = img
        if o["sharpen"] and blur_metric(img) < o["sharpen_auto"]:
            sg, am = o["sharpen"]
            bl = cv2.GaussianBlur(img, (0, 0), sg)
            crop_img = np.clip(img.astype(np.float32) * (1 + am) - bl.astype(np.float32) * am, 0, 255).astype(np.uint8)

        lb, sc = letterbox(img)
        with torch.autocast(self.device.split(":")[0], dtype=torch.float16, enabled=self.device != "cpu"):
            hm = torch.sigmoid(self.heat(normalize(torch.from_numpy(lb)[None].to(self.device)))[0].float())
        pts = find_peaks(hm[0], sc)
        swatches = find_peaks(hm[1], sc)
        wpk = find_peaks(hm[2], sc, thr=0.02)
        img = crop_img

        # --- giao lộ
        if o["iso_filter"] and len(pts) > 3:
            dnn = cKDTree(pts[:, :2]).query(pts[:, :2], k=2)[0][:, 1]
            pts = pts[dnn < o["iso_filter"] * G.spacing(pts[:, :2])]
        if o["mid_filter"] and len(pts) > 3:
            pts = self._drop_midpoints(pts)
        if force_P is not None:
            P = force_P
            rcs, sp = G.assign_rc(P)
        else:
            P = pts[:, :2]
            rcs, sp = G.assign_rc(P)
            best = {}  # hai đỉnh rơi cùng một ô (hàng, cột): giữ đỉnh mạnh hơn
            for i, rc in enumerate(rcs):
                if rc not in best or pts[i, 2] > pts[best[rc], 2]:
                    best[rc] = i
            keep = sorted(best.values())
            P, rcs = P[keep], [rcs[i] for i in keep]
            if self.recover_thr:
                P, rcs, sp = self._recover(P, rcs, sp, find_peaks(hm[0], sc, thr=self.recover_thr))
        node_p = 0
        for nsc in o["node_scales"]:  # TTA over crop scales
            node_p = node_p + self._classify([G.crop_node(img, p, sp, nsc) for p in P], "node")[0]["node"]
        node_p = node_p / len(o["node_scales"])

        # --- chú giải: loại dòng + kiểu nét của ô mẫu đường
        leg = self._classify([G.crop_legend(img, p, W) for p in swatches[:, :2]], "legend")[0]
        kind_p = leg.get("legend_kind", np.zeros((0, len(LEGEND_KINDS))))
        look_p = leg.get("legend_look", np.zeros((0, len(LOOKS))))
        if len(kind_p) and o["legend_assign"] and len(kind_p) <= len(LEGEND_KINDS):
            # each legend kind appears at most once in a legend: joint assignment of rows to kinds
            ri_, ci_ = linear_sum_assignment(-np.log(kind_p + 1e-9))
            kinds = [None] * len(kind_p)
            for r_, c_ in zip(ri_, ci_):
                kinds[r_] = LEGEND_KINDS[c_]
        else:
            kinds = [LEGEND_KINDS[k] for k in kind_p.argmax(1)] if len(kind_p) else []
        place_set = {k for k in kinds if k in LANDMARKS}
        look_to_status, legend_conf = self._legend_mapping(kind_p, look_p)

        # --- thời tiết
        if len(wpk):
            wc = wpk[np.argmax(wpk[:, 2]), :2]
            rain_p = float(self._classify([G.crop_weather(img, wc, W)], "weather")[0]["weather"][0, WEATHER.index("rain")])
        else:
            rain_p = 0.0

        # --- robot + địa điểm (ràng buộc mềm theo chú giải: đọc sai chú giải không được kéo sai cả bản đồ)
        robot_cols = [NODE_CLASSES.index("robot_" + h) for h in HEADINGS]
        ri = int(np.argmax(node_p[:, robot_cols].sum(1)))
        heading = HEADINGS[int(np.argmax(node_p[ri, robot_cols]))]
        prior = np.ones(len(NODE_CLASSES))
        if place_set:
            for t in LANDMARKS:
                if t not in place_set:
                    prior[NODE_CLASSES.index(t)] = self.legend_prior
        prior[robot_cols] = 0
        landmarks = []
        for i, p in enumerate(node_p):
            if i == ri:
                continue
            c = int(np.argmax(p * prior))
            if c:
                landmarks.append((NODE_CLASSES[c], rcs[i]))

        # --- đoạn đường giữa các cặp giao lộ kề nhau
        idx = {rc: i for i, rc in enumerate(rcs)}
        pairs = [(rc, nb) for rc in idx for nb in ((rc[0], rc[1] + 1), (rc[0] + 1, rc[1])) if nb in idx]
        ep, feat = self._classify([G.crop_edge(img, P[idx[a]], P[idx[b]], sp) for a, b in pairs], "edge")
        if pairs:
            if o["edge_proto"]:
                ep["edge_look"] = proto_refine(ep["edge_look"], feat)
            if o["edge_rev"]:  # cùng đoạn, cắt theo chiều ngược lại; một chiều đổi forward <-> backward
                er, feat_r = self._classify([G.crop_edge(img, P[idx[b]], P[idx[a]], sp) for a, b in pairs], "edge")
                if o["edge_proto"]:
                    er["edge_look"] = proto_refine(er["edge_look"], feat_r)
                ep = {"edge_look": (ep["edge_look"] + er["edge_look"]) / 2,
                      "edge_stairs": (ep["edge_stairs"] + er["edge_stairs"]) / 2,
                      "edge_oneway": (ep["edge_oneway"] + er["edge_oneway"][:, [0, 2, 1]]) / 2}
            ep["edge_oneway"] = ep["edge_oneway"].copy()
            ep["edge_oneway"][:, 0] *= o["ow_bias"]
            ep["edge_stairs"] = ep["edge_stairs"].copy()
            ep["edge_stairs"][:, 0] *= o["st_bias"]
        edges = []
        for k, (a, b) in enumerate(pairs):
            look = EDGE_LOOK[int(np.argmax(ep["edge_look"][k]))]
            if look == "none":
                continue
            status = "closed" if look == "closed" else look_to_status[look]
            ow = ONEWAY[int(np.argmax(ep["edge_oneway"][k]))] if status != "closed" else "none"
            edges.append({"a": a, "b": b, "status": status,
                          "stairs": bool(ep["edge_stairs"][k, 1] > ep["edge_stairs"][k, 0]),
                          "oneway_to": b if ow == "forward" else (a if ow == "backward" else None)})

        # giao lộ không có đoạn đường nào (hồ nước, chữ tiêu đề...) thì bỏ; bản đồ thật không có nút lẻ
        used = {e["a"] for e in edges} | {e["b"] for e in edges} | {rcs[ri]}
        landmarks = [(t, rc) for t, rc in landmarks if rc in used]
        robot = rcs[ri]
        if o["compact"]:  # hàng/cột giả ở rìa bị bỏ hết -> dịch chỉ số về 0 (chỉ khi là phép dịch thuần)
            rr, cc = sorted({r for r, _ in used}), sorted({c for _, c in used})
            if (rr[0] or cc[0]) and rr == list(range(rr[0], rr[0] + len(rr))) and cc == list(range(cc[0], cc[0] + len(cc))):
                dr, dc = rr[0], cc[0]

                def sh(rc):
                    return (rc[0] - dr, rc[1] - dc)
                edges = [dict(e, a=sh(e["a"]), b=sh(e["b"]), oneway_to=sh(e["oneway_to"]) if e["oneway_to"] else None)
                         for e in edges]
                landmarks = [(t, sh(rc)) for t, rc in landmarks]
                used = {sh(rc) for rc in used}
                rcs = [sh(rc) for rc in rcs]
                robot = sh(robot)
        nodes = [rc for rc in rcs if rc in used]
        graph = {"nodes": nodes, "edges": edges, "landmarks": landmarks,
                 "robot": robot, "heading": heading, "rain": rain_p > 0.5,
                 "grid": (max(r for r, _ in nodes) + 1, max(c for _, c in nodes) + 1)}
        aux = {"points": P, "spacing": sp, "node_p": node_p, "edge_pairs": pairs, "edge_p": ep, "rain_p": rain_p,
               "legend_kinds": kinds, "look_to_status": look_to_status, "legend_conf": legend_conf}

        if o["reassign"] and force_P is None:  # bỏ điểm không có đường rồi gán lại hàng/cột từ đầu
            keep_i = [i for i, rc in enumerate(rcs) if rc in used]
            if 3 < len(keep_i) < len(rcs):
                P0 = P[keep_i]
                rcs2, _ = G.assign_rc(P0)
                if rcs2 != [rcs[i] for i in keep_i]:
                    return self.read_img(img_in, force_P=P0)
        return graph, aux

    @staticmethod
    def _drop_midpoints(pts):
        """Bỏ đỉnh nằm đúng trung điểm của hai đỉnh cách nhau ~1 ô lưới: đó là ký hiệu giữa đoạn đường
        (vạch đóng, bậc thang, mũi tên một chiều), không phải giao lộ."""
        P = pts[:, :2]
        sp = G.spacing(P)
        tree = cKDTree(P)
        drop = set()
        for i, p in enumerate(P):
            nb = [j for j in tree.query_ball_point(p, 0.95 * sp) if j != i]
            for a, b in itertools.combinations(nb, 2):
                d = np.linalg.norm(P[a] - P[b])  # hàng và cột có thể cách nhau khác nhau -> nới khoảng
                if 0.7 * sp < d < 1.8 * sp and np.linalg.norm(p - (P[a] + P[b]) / 2) < 0.12 * d:
                    drop.add(i)
                    break
        return pts[[i for i in range(len(pts)) if i not in drop]]

    @staticmethod
    def _recover(P, rcs, sp, low):
        """Thêm đỉnh yếu nếu nó lấp một ô trống mà không tạo hàng/cột mới."""
        nr, nc = max(r for r, _ in rcs) + 1, max(c for _, c in rcs) + 1
        for q in low[np.argsort(-low[:, 2])] if len(low) else []:
            if np.min(np.linalg.norm(P - q[:2], axis=1)) < 0.4 * sp:
                continue
            P2 = np.vstack([P, q[:2]])
            rcs2, sp2 = G.assign_rc(P2)
            r2, c2 = max(r for r, _ in rcs2) + 1, max(c for _, c in rcs2) + 1
            if (r2, c2) == (nr, nc) and len(set(rcs2)) == len(rcs2):
                P, rcs, sp = P2, rcs2, sp2
        return P, rcs, sp

    @staticmethod
    def _legend_mapping(kind_p, look_p):
        """Chọn hoán vị kiểu nét -> trạng thái hợp lý nhất theo 3 dòng chú giải đường."""
        if len(kind_p) == 0:
            return {s: s for s in STATUSES}, 0.0
        rows = {s: int(np.argmax(kind_p[:, LEGEND_KINDS.index(s)])) for s in STATUSES}
        best, best_score = None, -np.inf
        for perm in itertools.permutations(LOOKS):
            score = sum(np.log(look_p[rows[s], LOOKS.index(l)] + 1e-9) for s, l in zip(STATUSES, perm))
            if score > best_score:
                best, best_score = perm, score
        return {l: s for s, l in zip(STATUSES, best)}, float(np.exp(best_score))


# ---------------------------------------------------------------- clean inference entry point
ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent.parent
DEFAULT_HEAT = ROOT / "work" / "heat_v3.pt"
DEFAULT_CROPS = [ROOT / "work" / "crops_v3.pt"]


def load_models(heat=None, crops=None, device=None, opts=None):
    """Load the CV models once. crops: one path or a list (ensemble). Returns a MapReader."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    crops = crops or DEFAULT_CROPS
    if isinstance(crops, (list, tuple)):
        crops = [str(c) for c in crops]
    else:
        crops = str(crops)
    return MapReader(str(heat or DEFAULT_HEAT), crops, device=device, opts=opts)


def read_image(path, models=None):
    """Map image -> graph dict (same format as courier.policy.graph_from_scene, plus "grid" = (rows, cols)
    estimated from the read nodes). models: a MapReader from load_models() (loaded with defaults if None)."""
    if models is None:
        models = load_models()
    return models.read(str(path))[0]
