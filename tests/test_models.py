"""Faz 5: motor meta verisi, giriş boyutları ve sabit boyutlu motorla çıkarım kontrolleri."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))
from adb.models import (Detector, band_input_shape, engine_meta, full_input_shape,  # noqa: E402
                        run_dir_for, write_engine_file)
from adb.slicing import load_slicing, predict_band, predict_full  # noqa: E402
from build_engine import letterbox_chw, meta_for_json  # noqa: E402

CFG = load_slicing()


# ---------------------------------------------------------------- sahte Ultralytics modeli
class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy, self.cls, self.conf = _T(np.asarray(xyxy, float).reshape(-1, 4)), _T(cls), _T(conf)


class _R:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeModel:
    """predict çağrılarını kaydeder; her girişte (10,10)-(20,20) car kutusu döndürür."""

    def __init__(self):
        self.calls = []

    def predict(self, src, imgsz=None, batch=None, **kw):
        n = len(src) if isinstance(src, list) else 1
        self.calls.append({"imgsz": imgsz, "batch": batch, "n": n,
                           "shape": (src[0] if isinstance(src, list) else src).shape})
        return [_R(_Boxes([[10, 10, 20, 20]], [0], [0.9])) for _ in range(n)]


IMG = np.zeros((720, 1280, 3), np.uint8)


# ---------------------------------------------------------------- boyutlar
def test_input_shapes_for_720p():
    assert full_input_shape(720, 1280) == (736, 1280)          # kare 1280×1280 yerine dikdörtgen
    assert band_input_shape(720, 1280, CFG) == ((704, 768), 4)


# ---------------------------------------------------------------- meta veri
def test_engine_meta_roundtrip(tmp_path):
    meta = {"imgsz": [704, 768], "batch": 4, "names": "{0: 'car'}"}
    p = write_engine_file(tmp_path / "band_fp16.engine", b"\x00TRTENGINE" * 10, meta)
    assert engine_meta(p) == meta
    assert p.read_bytes().endswith(b"\x00TRTENGINE" * 10)      # motor baytları meta verinin hemen ardından


def test_raw_engine_has_no_meta(tmp_path):
    p = write_engine_file(tmp_path / "full_fp16_raw.engine", b"ftrt" + b"\x07" * 100, None)
    assert engine_meta(p) is None
    assert engine_meta(tmp_path / "yok.engine") is None
    assert engine_meta(tmp_path / "best.pt") is None


def test_onnx_meta_strings_become_numbers():
    m = meta_for_json({"imgsz": "[736, 1280]", "batch": "1", "names": "{0: 'car'}"})
    assert m["imgsz"] == [736, 1280] and m["batch"] == 1 and m["names"] == "{0: 'car'}"


def test_run_dir_engine_does_not_overwrite_pt_results(tmp_path):
    assert run_dir_for(tmp_path / "runs/baseline/weights/best.pt") == tmp_path / "runs/baseline"
    d = run_dir_for(tmp_path / "deploy/models/full_fp16.engine", root=tmp_path)
    assert d == tmp_path / "runs/trt/full_fp16" and d.is_dir()


# ---------------------------------------------------------------- çıkarım
def test_full_uses_engine_imgsz():
    m = FakeModel()
    predict_full(Detector(m, "full_fp16.engine", (736, 1280), 1, True), IMG, 0.1, imgsz=1280)
    assert m.calls[0]["imgsz"] == (736, 1280)


def test_full_pt_keeps_cli_imgsz():
    m = FakeModel()
    predict_full(m, IMG, 0.1, 1280)
    assert m.calls[0]["imgsz"] == 1280


def test_band_pt_same_input_as_before():
    """.pt'de bant girişi Faz 3–4 ile aynı (704×768 parçalar, 4 adet, tek batch)."""
    m = FakeModel()
    xyxy, cls, conf = predict_band(m, IMG, CFG, 0.1)
    c = m.calls[0]
    assert c["imgsz"] == (704, 768) and c["n"] == 4 and c["batch"] == 4 and c["shape"][:2] == (704, 768)
    assert len(xyxy) == 4                                      # her parçadan bir kutu, orijinal koordinatlarda


def test_band_engine_ok():
    m = FakeModel()
    predict_band(Detector(m, "band_fp16.engine", (704, 768), 4, True), IMG, CFG, 0.1)
    assert m.calls[0]["imgsz"] == (704, 768)


def test_band_engine_wrong_shape_or_batch_raises():
    with pytest.raises(ValueError, match="giriş boyutu"):
        predict_band(Detector(FakeModel(), "x.engine", (768, 768), 4, True), IMG, CFG, 0.1)
    with pytest.raises(ValueError, match="batch"):
        predict_band(Detector(FakeModel(), "x.engine", (704, 768), 1, True), IMG, CFG, 0.1)


# ---------------------------------------------------------------- INT8 kalibrasyon ön işlemesi
def test_letterbox_matches_ultralytics_padding():
    img = np.full((720, 1280, 3), 200, np.uint8)
    img[..., 0] = 10                                           # B kanalı farklı: RGB sırası kontrolü
    x = letterbox_chw(img, (736, 1280))
    assert x.shape == (3, 736, 1280) and x.dtype == np.float32
    assert np.allclose(x[:, :8], 114 / 255) and np.allclose(x[:, -8:], 114 / 255)   # üst/alt 8 px dolgu
    assert np.isclose(x[2, 100, 100], 10 / 255) and np.isclose(x[0, 100, 100], 200 / 255)  # B -> son kanal
    crop = np.zeros((704, 768, 3), np.uint8)
    assert np.allclose(letterbox_chw(crop, (704, 768)), 0)    # bant parçası tam oturur, dolgu yok
