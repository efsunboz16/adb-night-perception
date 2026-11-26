"""Algılama modellerini yükleme: PyTorch (.pt), ONNX veya TensorRT (.engine).

Faz 5'te tam kare ve ufuk bandı iki ayrı, sabit boyutlu TensorRT motoruna dönüştürüldü (ADR-019):
  tam kare : 736×1280, batch 1  (720p dikdörtgen; 1280×1280 kare girişte piksellerin %42'si dolgu)
  bant     : 704×768, batch 4   (352×384'lük 4 parça, 2 kat büyütülmüş; configs/slicing.yaml)

Sabit boyutlu bir motora farklı boyutta görüntü verilemez. Bu yüzden her modelin giriş boyutu ve
batch'i burada bir kez belirlenir ve çıkarım fonksiyonlarına (adb/slicing.py) aktarılır:
  - .engine : Ultralytics'in dosya başına yazdığı meta veriden (4 bayt uzunluk + JSON)
  - .pt     : komut satırındaki --imgsz (bant için parça boyutundan hesaplanır)

trtexec ile derlenen motorlarda bu meta veri yoktur; değerlendirmede Ultralytics'in ürettiği
.engine dosyaları kullanılmalı. trtexec motorları yalnızca gecikme ölçümü içindir.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

ENGINE_SUFFIXES = (".engine", ".plan")


def is_engine(path: str | Path) -> bool:
    return Path(path).suffix.lower() in ENGINE_SUFFIXES


def engine_meta(path: str | Path) -> dict | None:
    """Ultralytics .engine dosyasının başındaki meta veriyi okur. Meta veri yoksa None."""
    p = Path(path)
    if not is_engine(p) or not p.exists():
        return None
    with p.open("rb") as f:
        head = f.read(4)
        if len(head) < 4:
            return None
        n = int.from_bytes(head, byteorder="little", signed=True)
        if not 0 < n < 1_000_000:                 # trtexec motoru: ilk baytlar TensorRT başlığı
            return None
        try:
            return json.loads(f.read(n).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None


def ceil32(v: int) -> int:
    return int(math.ceil(v / 32) * 32)


def band_input_shape(img_h: int, img_w: int, cfg: dict) -> tuple[tuple[int, int], int]:
    """Ufuk bandı parçalarının model giriş boyutu (h, w) ve parça sayısı.
    1280×720 ve varsayılan slicing.yaml için ((704, 768), 4)."""
    from adb.slicing import tile_xs

    y0, y1 = (int(round(v * img_h)) for v in cfg["band_frac"])
    h = int(round((y1 - y0) * cfg["scale"]))
    w = int(round(min(cfg["tile_w"], img_w) * cfg["scale"]))
    return (ceil32(h), ceil32(w)), len(tile_xs(img_w, cfg["tile_w"], cfg["overlap"]))


def full_input_shape(img_h: int, img_w: int, imgsz: int = 1280) -> tuple[int, int]:
    """Tam kare dikdörtgen giriş: uzun kenar imgsz, kısa kenar orana göre, 32'nin katı.
    1280×720 için (736, 1280)."""
    r = imgsz / max(img_h, img_w)
    return ceil32(int(round(img_h * r))), ceil32(int(round(img_w * r)))


@dataclass
class Detector:
    """Bir model ve onun sabit giriş özellikleri."""
    model: object
    path: str
    imgsz: int | tuple[int, int]          # Ultralytics predict'e verilecek değer
    batch: int | None = None              # motorun sabit batch'i (.pt için None: serbest)
    engine: bool = False

    def describe(self) -> str:
        kind = "TensorRT" if self.engine else Path(self.path).suffix.lstrip(".") or "model"
        sz = "×".join(map(str, self.imgsz)) if isinstance(self.imgsz, (tuple, list)) else str(self.imgsz)
        return f"{Path(self.path).name} ({kind}, {sz}" + (f", batch {self.batch})" if self.batch else ")")


def _load(path: str, imgsz_default):
    from ultralytics import YOLO

    if is_engine(path):
        meta = engine_meta(path)
        if meta is None or "imgsz" not in meta:
            raise SystemExit(f"{path}: Ultralytics meta verisi yok (trtexec motoru mu?). Değerlendirme için "
                             f"deploy/export.py ile üretilen .engine dosyasını kullan.")
        return Detector(YOLO(path, task="detect"), str(path), tuple(int(v) for v in meta["imgsz"]),
                        int(meta.get("batch", 1)), engine=True)
    return Detector(YOLO(path), str(path), imgsz_default)


def load_detectors(weights: str, band_weights: str | None = None, imgsz: int = 1280,
                   need_band: bool = True) -> tuple[Detector, Detector | None]:
    """Tam kare ve bant modellerini yükler.

    band_weights verilmezse:
      - .pt / .onnx: bant da aynı modelle çalışır (Faz 3–4 davranışı)
      - .engine   : sabit boyutlu motor bantta kullanılamaz; need_band ise hata
    """
    full = _load(weights, imgsz)
    if band_weights:
        band = _load(band_weights, None)
    elif not full.engine:
        band = Detector(full.model, full.path, None)       # imgsz parça boyutundan hesaplanır
    elif need_band:
        raise SystemExit("Tam kare için TensorRT motoru verildi; bant için de --band-weights gerekli "
                         "(ör. deploy/models/band_fp16.engine).")
    else:
        band = None
    return full, band


def run_dir_for(weights: str | Path, root: Path | None = None) -> Path:
    """Sonuçların yazılacağı klasör.
    .pt  : runs/<ad>/weights/best.pt -> runs/<ad>   (Faz 1–4 ile aynı)
    motor: deploy/models/full_fp16.engine -> runs/trt/full_fp16   (PyTorch sonuçlarının üzerine yazmaz)"""
    p = Path(weights).resolve()
    if is_engine(p) or p.suffix.lower() == ".onnx":
        root = root or Path(__file__).resolve().parents[1]
        d = root / "runs" / "trt" / p.stem
        d.mkdir(parents=True, exist_ok=True)
        return d
    return p.parents[1]


def write_engine_file(path: str | Path, engine_bytes: bytes, meta: dict | None) -> Path:
    """Motoru Ultralytics'in okuyabileceği biçimde yazar: 4 bayt uzunluk + JSON meta + motor.
    meta None ise ham motor (trtexec --loadEngine ile açılabilir, Ultralytics sınıf adlarını bilmez)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        if meta is not None:
            m = json.dumps(meta).encode("utf-8")
            f.write(len(m).to_bytes(4, byteorder="little", signed=True))
            f.write(m)
        f.write(engine_bytes)
    return p
