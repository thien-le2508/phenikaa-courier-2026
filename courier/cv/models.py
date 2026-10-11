"""Mạng CV: HeatmapNet (tìm tâm trên toàn ảnh) và CropNet (phân loại mảnh ảnh, nhiều đầu)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from courier.cv.labels import EDGE_LOOK, LEGEND_KINDS, LOOKS, NODE_CLASSES, ONEWAY, WEATHER

HEAT_CHANNELS = ["node", "swatch", "weather"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def normalize(x_uint8):
    """uint8 NHWC hoặc NCHW trên GPU -> float chuẩn hóa ImageNet NCHW."""
    if x_uint8.shape[-1] == 3:
        x_uint8 = x_uint8.permute(0, 3, 1, 2)
    x = x_uint8.float()
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def _resnet18(pretrained):
    return torchvision.models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)


class HeatmapNet(nn.Module):
    """ResNet18 + FPN đơn giản, đầu ra stride 4."""

    def __init__(self, pretrained=True, ch=128):
        super().__init__()
        r = _resnet18(pretrained)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.lat = nn.ModuleList([nn.Conv2d(c, ch, 1) for c in (64, 128, 256, 512)])
        self.smooth = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                                    nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(ch, len(HEAT_CHANNELS), 1)
        nn.init.constant_(self.head.bias, -2.19)

    def forward(self, x):
        c1 = self.l1(self.stem(x))
        c2 = self.l2(c1)
        c3 = self.l3(c2)
        c4 = self.l4(c3)
        p = self.lat[3](c4)
        for lat, c in ((self.lat[2], c3), (self.lat[1], c2), (self.lat[0], c1)):
            p = F.interpolate(p, size=c.shape[-2:], mode="nearest") + lat(c)
        return self.head(self.smooth(p))


CROP_HEADS = {
    "node": len(NODE_CLASSES), "edge_look": len(EDGE_LOOK), "edge_stairs": 2, "edge_oneway": len(ONEWAY),
    "legend_kind": len(LEGEND_KINDS), "legend_look": len(LOOKS), "weather": len(WEATHER),
}
TASK_HEADS = {"node": ["node"], "edge": ["edge_look", "edge_stairs", "edge_oneway"],
              "legend": ["legend_kind", "legend_look"], "weather": ["weather"]}


class CropNet(nn.Module):
    """Một thân ResNet18 dùng chung cho mọi loại mảnh; mỗi loại mảnh có đầu riêng."""

    def __init__(self, pretrained=True):
        super().__init__()
        r = _resnet18(pretrained)
        self.trunk = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        self.task_emb = nn.ParameterDict({t: nn.Parameter(torch.zeros(512)) for t in TASK_HEADS})
        self.heads = nn.ModuleDict({h: nn.Sequential(nn.Dropout(0.2), nn.Linear(512, n)) for h, n in CROP_HEADS.items()})

    def forward(self, x, task):
        f = self.trunk(x)
        f = F.adaptive_avg_pool2d(f, 1).flatten(1) + self.task_emb[task]
        return {h: self.heads[h](f) for h in TASK_HEADS[task]}


def focal_loss(logits, target, alpha=2, beta=4):
    """Focal loss kiểu CenterNet cho heatmap Gaussian."""
    p = torch.sigmoid(logits.float()).clamp(1e-4, 1 - 1e-4)
    pos = target.eq(1).float()
    pos = torch.maximum(pos, (target > 0.99).float())
    neg = 1 - pos
    pos_loss = torch.log(p) * (1 - p) ** alpha * pos
    neg_loss = torch.log(1 - p) * p ** alpha * (1 - target) ** beta * neg
    n = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / n
