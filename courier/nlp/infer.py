"""Dùng bộ phân tích yêu cầu v3 khi dự đoán.

    from courier.nlp import infer
    reader = infer.load("work/nlp_v3_quick.pt")              # cuda nếu có
    missions = infer.parse(reader, texts, landmark_type_lists)  # landmark_type_lists: mỗi câu 1 list loại (có lặp) hoặc None

Mỗi mission = {"goal": type|None, "goal_ref": None|{"kind","anchor"}, "via": type|None,
               "via_ref": None|{"kind","anchor"}, "urgent": bool, "fragile": bool}
goal None <=> nơi giao chỉ được mô tả qua bản đồ (goal_ref.kind ∈ north_most/south_most/west_most/east_most/anchor_near)
và câu không lộ loại địa điểm; khi đó policy giải nơi giao từ bản đồ:
  *_most      -> địa điểm có hàng nhỏ nhất / lớn nhất, cột nhỏ nhất / lớn nhất (trên toàn bộ địa điểm)
  anchor_near -> địa điểm gần mốc nhất theo khoảng cách Manhattan (|dr|+|dc|), không tính chính mốc
                 (đã kiểm tra: đúng 723/723 train, 125/125 val; Euclid sai 6 + 3 cảnh)
"""
import torch

from courier.nlp.decode import decode
from courier.nlp.model import MissionParser, get_tokenizer, predict_probs
from courier.nlp.rules import rule_attrs


class MissionReader:
    def __init__(self, ckpt_path, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "heads" not in ck:
            raise ValueError(f"{ckpt_path}: checkpoint vòng trước (nhãn cũ), không dùng được với courier/nlp v3")
        self.tok = get_tokenizer(ck["encoder"])
        self.model = MissionParser(ck["encoder"])
        self.model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
        self.model.to(device).eval()
        self.device = device

    def probs(self, texts):
        return predict_probs(self.model, self.tok, list(texts), device=self.device)

    def read(self, texts, landmark_types=None, rules=False, tta=False):
        """landmark_types: list (mỗi câu một list loại địa điểm trên bản đồ, có lặp) hoặc None.
        rules: luật phủ định/phân cực (rules.py) ghi đè urgent/fragile khi tín hiệu đủ mạnh (|điểm| >= 1).
        tta: bỏ qua (giữ tương thích với scripts/predict.py)."""
        texts = list(texts)
        ps = self.probs(texts)
        lts = landmark_types or [None] * len(ps)
        out = [decode(p, lt) for p, lt in zip(ps, lts)]
        if rules:
            for m, t in zip(out, texts):
                r = rule_attrs(t)
                for f in ("urgent", "fragile"):
                    if abs(r[f]) >= 1:
                        m[f] = r[f] > 0
        return out


def load(path, device=None):
    return MissionReader(path, device)


def parse(reader, texts, landmark_type_lists=None, rules=False):
    return reader.read(texts, landmark_type_lists, rules=rules)
