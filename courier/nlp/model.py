"""Bộ phân tích yêu cầu: encoder tiếng Việt + 8 đầu phân loại."""
import torch
import torch.nn as nn
import transformers.utils as _tu
import transformers.utils.import_utils as _iu

_iu.is_torchaudio_available = _tu.is_torchaudio_available = lambda: False  # torchaudio trong anaconda bị lỗi DLL
from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: E402

from courier.data import LANDMARKS  # noqa: E402

# v3: thêm các kiểu mô tả nơi giao qua bản đồ; goal có thể None (không lộ loại)
REF_KINDS = ["north", "south", "west", "east", "near", "far",
             "north_most", "south_most", "west_most", "east_most", "anchor_near"]
VIA_KINDS = ["north", "south", "west", "east", "near", "far"]
HEADS = {
    "goal": [None] + LANDMARKS,
    "via": [None] + LANDMARKS,
    "goal_kind": [None] + REF_KINDS,
    "goal_anchor": [None] + LANDMARKS,
    "via_kind": [None] + VIA_KINDS,
    "via_anchor": [None] + LANDMARKS,
    "urgent": [False, True],
    "fragile": [False, True],
}


def encode_labels(m):
    g, v = m.get("goal_ref") or {}, m.get("via_ref") or {}
    vals = {"goal": m["goal"], "via": m.get("via"), "goal_kind": g.get("kind"), "goal_anchor": g.get("anchor"),
            "via_kind": v.get("kind"), "via_anchor": v.get("anchor"),
            "urgent": bool(m["urgent"]), "fragile": bool(m["fragile"])}
    return {h: HEADS[h].index(vals[h]) for h in HEADS}


def load_encoder(name):
    try:
        return AutoModel.from_pretrained(name, use_safetensors=True, add_pooling_layer=False)
    except (OSError, ValueError, EnvironmentError):
        # Chỉ có pytorch_model.bin mà torch < 2.6: tự nạp bằng weights_only=True
        from huggingface_hub import hf_hub_download
        cfg = AutoConfig.from_pretrained(name)
        model = AutoModel.from_config(cfg, add_pooling_layer=False)
        sd = torch.load(hf_hub_download(name, "pytorch_model.bin"), map_location="cpu", weights_only=True)
        sd = {k.split(".", 1)[1] if k.startswith(("roberta.", "bert.")) else k: v for k, v in sd.items()}
        missing, _ = model.load_state_dict(sd, strict=False)
        assert not [k for k in missing if "position_ids" not in k], missing
        return model


class MissionParser(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder_name = encoder_name
        self.enc = load_encoder(encoder_name)
        H = self.enc.config.hidden_size
        self.heads = nn.ModuleDict({h: nn.Sequential(nn.Dropout(0.1), nn.Linear(2 * H, len(v)))
                                    for h, v in HEADS.items()})

    def forward(self, input_ids, attention_mask):
        o = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(o.dtype)
        pooled = torch.cat([o[:, 0], (o * mask).sum(1) / mask.sum(1)], -1)
        return {h: f(pooled) for h, f in self.heads.items()}


def get_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)


@torch.no_grad()
def predict_probs(model, tok, texts, device="cuda", bs=64, max_len=192):
    """Trả về list dict {head: list xác suất}."""
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        probs = {h: torch.softmax(v.float(), -1).cpu().tolist() for h, v in logits.items()}
        out += [{h: probs[h][j] for h in HEADS} for j in range(len(probs["goal"]))]
    return out
