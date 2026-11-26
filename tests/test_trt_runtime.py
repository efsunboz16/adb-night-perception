"""GPU hattının ön/son işlemesi, CPU'da: Ultralytics / cv2 yoluyla aynı sonucu vermeli (ADR-021).

TensorRT ve CUDA gerektiren kısım (TrtEngine) burada test edilmez; doğruluğu eval_distance.py
--backend trt ile, Ultralytics motor sonuçlarıyla karşılaştırılarak ölçülür.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy"))
from adb.slicing import band_crops, load_slicing, tile_xs  # noqa: E402
from adb.trt_runtime import (decode, letterbox_params, preprocess_band,  # noqa: E402
                             preprocess_full, unletterbox)
from build_engine import letterbox_chw  # noqa: E402

CFG = load_slicing()
RNG = np.random.default_rng(0)
IMG = RNG.integers(0, 256, (720, 1280, 3), dtype=np.uint8)


def test_letterbox_params_720p():
    assert letterbox_params(720, 1280, 736, 1280) == (1.0, 720, 1280, 8, 0)


def test_full_preprocess_identical_to_cpu_path():
    lb = letterbox_params(720, 1280, 736, 1280)
    out = torch.empty((1, 3, 736, 1280))
    preprocess_full(torch.from_numpy(IMG), out, lb)
    ref = letterbox_chw(IMG, (736, 1280))                     # Ultralytics LetterBox + BGR->RGB + /255
    assert np.allclose(out[0].numpy(), ref, atol=1e-6)


def test_band_preprocess_matches_cv2_resize():
    """cv2.INTER_LINEAR 2x büyütme ile en fazla 1 gri seviye fark. cv2 uint8'de sabit noktalı katsayı
    kullanır, yuvarlama sınırdaki değerlerde ayrışır. Rastgele gürültü en kötü durum (%8 pikselde 1 seviye)."""
    crops, xs, y0 = band_crops(IMG, CFG)
    y1 = int(round(CFG["band_frac"][1] * 720))
    out = torch.empty((len(xs), 3, 704, 768))
    preprocess_band(torch.from_numpy(IMG), out, y0, y1, xs, CFG["tile_w"], CFG["scale"])
    ref = np.stack([letterbox_chw(c, (704, 768)) for c in crops])
    diff = np.abs(out.numpy() - ref) * 255
    assert diff.max() <= 1.0 + 1e-4 and diff.mean() < 0.15


def _pred(rows, nc=10, n=50):
    """Sahte YOLO çıktısı (1, 4+nc, n): rows = [(cx, cy, w, h, sınıf, olasılık), ...], gerisi sıfır."""
    p = torch.zeros((1, 4 + nc, n))
    for i, (cx, cy, w, h, c, s) in enumerate(rows):
        p[0, :4, i] = torch.tensor([cx, cy, w, h])
        p[0, 4 + c, i] = s
    return p


def test_decode_threshold_nms_and_classes():
    pred = _pred([(100, 100, 20, 10, 0, 0.9),     # araç
                  (101, 100, 20, 10, 0, 0.6),     # aynı araç, tekrar -> NMS atar
                  (101, 100, 20, 10, 6, 0.5),     # aynı yerde far: farklı sınıf, kalır
                  (500, 300, 4, 4, 7, 0.05)])     # eşik altı
    (xyxy, cls, conf), = decode(pred, conf=0.1)
    assert cls.tolist() == [0, 6] and np.allclose(conf.numpy(), [0.9, 0.5])
    assert np.allclose(xyxy[0].numpy(), [90, 95, 110, 105])


def test_decode_empty():
    (xyxy, cls, conf), = decode(_pred([]), conf=0.1)
    assert xyxy.shape == (0, 4) and len(cls) == 0


def test_unletterbox_removes_padding_and_clips():
    lb = letterbox_params(720, 1280, 736, 1280)
    b = unletterbox(torch.tensor([[10., 18., 50., 60.], [1270., 700., 1290., 740.]]), lb, 720, 1280)
    assert np.allclose(b.numpy(), [[10, 10, 50, 52], [1270, 692, 1280, 720]])


def test_band_tiles_consistent_with_slicing():
    assert tile_xs(1280, CFG["tile_w"], CFG["overlap"]) == band_crops(IMG, CFG)[1]
