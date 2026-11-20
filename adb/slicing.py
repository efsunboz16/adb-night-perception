"""Ufuk bandı dilimleme (SAHI tarzı) ile çıkarım.

Uzak araçlar görüntüde dar bir yatay bantta (ufuk çevresi) toplanır. Tam kare çıkarıma ek olarak bu
bant örtüşen parçalara bölünür, `scale` kat büyütülüp modele verilir; tahminler orijinal koordinatlara
taşınıp tam kare tahminleriyle birleştirilir ve tekrarlar NMS ile ayıklanır.

Bant ve parça ayarları configs/slicing.yaml'dadır (scripts/horizon_band.py üretir).
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_slicing(path: str | Path = ROOT / "configs" / "slicing.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def tile_xs(width: int, tile_w: int, overlap: int) -> list[int]:
    """Bandı soldan sağa örtüşen parçalara böler; son parça sağ kenara hizalanır."""
    if tile_w >= width:
        return [0]
    xs = list(range(0, width - tile_w + 1, tile_w - overlap))
    if xs[-1] + tile_w < width:
        xs.append(width - tile_w)
    return xs


def map_tile_boxes(xyxy: np.ndarray, x0: int, y0: int, scale: float) -> np.ndarray:
    """Büyütülmüş parça koordinatlarındaki kutuları orijinal görüntü koordinatlarına taşır."""
    out = xyxy / scale
    out[:, [0, 2]] += x0
    out[:, [1, 3]] += y0
    return out


def drop_cut_boxes(xyxy: np.ndarray, x0: int, tile_w: int, img_w: int, margin: float = 2.0) -> np.ndarray:
    """Parçanın İÇ kenarına değen kutular (nesne kesilmiş olabilir) atılır; görüntü kenarı hariç."""
    keep = np.ones(len(xyxy), bool)
    if x0 > 0:
        keep &= xyxy[:, 0] > x0 + margin
    if x0 + tile_w < img_w:
        keep &= xyxy[:, 2] < x0 + tile_w - margin
    return keep


def _boxes(r):
    b = r.boxes
    return b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy()


def predict_full(model, img, conf, imgsz=1280, half=True):
    return _boxes(model.predict(img, imgsz=imgsz, conf=conf, half=half, verbose=False)[0])


def predict_band(model, img, cfg: dict, conf, half=True):
    """Sadece bant parçalarındaki tahminler (orijinal koordinatlarda)."""
    h, w = img.shape[:2]
    y0, y1 = (int(round(v * h)) for v in cfg["band_frac"])
    tile_w, overlap, scale = cfg["tile_w"], cfg["overlap"], cfg["scale"]
    xs = tile_xs(w, tile_w, overlap)
    crops = [cv2.resize(img[y0:y1, x:x + tile_w], None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
             for x in xs]
    imgsz = int(math.ceil(max(crops[0].shape[:2]) / 32) * 32)
    rs = model.predict(crops, imgsz=imgsz, conf=conf, half=half, batch=len(crops), verbose=False)
    xyxy, cls, cf = [], [], []
    for r, x in zip(rs, xs):
        b, c, s = _boxes(r)
        if len(b) == 0:
            continue
        b = map_tile_boxes(b, x, y0, scale)
        k = drop_cut_boxes(b, x, tile_w, w)
        xyxy.append(b[k]), cls.append(c[k]), cf.append(s[k])
    if not xyxy:
        return np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)
    return np.concatenate(xyxy), np.concatenate(cls), np.concatenate(cf)


def merge(a, b, iou_thr=0.5):
    """İki tahmin kümesini birleştirip sınıf bazında NMS uygular."""
    import torch
    from torchvision.ops import batched_nms

    xyxy = np.concatenate([a[0], b[0]]).astype(np.float32)
    cls = np.concatenate([a[1], b[1]])
    conf = np.concatenate([a[2], b[2]]).astype(np.float32)
    if len(xyxy) == 0:
        return xyxy.reshape(0, 4), cls, conf
    keep = batched_nms(torch.from_numpy(xyxy), torch.from_numpy(conf), torch.from_numpy(cls), iou_thr).numpy()
    return xyxy[keep], cls[keep], conf[keep]