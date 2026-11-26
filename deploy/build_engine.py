"""ONNX'ten TensorRT motoru derler (FP16 veya INT8), Ultralytics'in okuyabileceği meta veriyle.

Neden Ultralytics'in `format=engine` dışa aktarması yerine kendi derleyicimiz:
  - INT8 kalibrasyonu doğru veriyle yapılmalı: tam kare motoru tam karelerle, bant motoru BANT
    PARÇALARIYLA (2x büyütülmüş 352×384 kesitler). Ultralytics kalibrasyonda kare letterbox kullanır,
    dikdörtgen sabit girişli bir motorla uyuşmaz.
  - Hangi katmanların FP16'da kalacağını seçebilmek (--fp16-layers): uzak ışıkların zayıf
    aktivasyonları INT8'de ilk kaybolan şey; tespit kafasını FP16 tutmak bir seçenek.
  - Motor tek GPU'ya bağlıdır. Derleme hangi makinede yapılırsa ölçüm de orada yapılır.

Meta veri ONNX dosyasından (Ultralytics'in yazdığı metadata_props) kopyalanır; böylece motor
YOLO("x.engine") ile açılır ve değerlendirme scriptleri değişmeden çalışır.

Kullanım (deploy/export.py bunları sırayla çağırır):
  python deploy/build_engine.py --onnx deploy/models/full.onnx --precision fp16
  python deploy/build_engine.py --onnx deploy/models/band.onnx --precision int8 --kind band
  python deploy/build_engine.py --onnx deploy/models/full.onnx --precision fp16 --raw   # trtexec için
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adb.models import write_engine_file  # noqa: E402

PAD = 114  # Ultralytics LetterBox dolgu rengi


# ---------------------------------------------------------------- ön işleme (Ultralytics ile aynı)
def letterbox_chw(img_bgr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Ultralytics çıkarım ön işlemesinin aynısı: oranı koruyarak sığdır, ortala, 114 ile doldur,
    BGR->RGB, /255, CHW float32."""
    import cv2

    h, w = img_bgr.shape[:2]
    H, W = shape
    r = min(H / h, W / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        img_bgr = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (H - nh) // 2, (W - nw) // 2
    out = np.full((H, W, 3), PAD, np.uint8)
    out[top:top + nh, left:left + nw] = img_bgr
    return np.ascontiguousarray(out[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0


def calib_batches(paths: list[str], kind: str, shape: tuple[int, int], slicing_cfg: dict | None):
    """Kalibrasyon girdileri: tam kare için görüntü başına 1, bant için görüntü başına N parça (batch)."""
    import cv2
    from adb.slicing import band_crops

    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        if kind == "band":
            crops, _, _ = band_crops(img, slicing_cfg)
            yield np.stack([letterbox_chw(c, shape) for c in crops])
        else:
            yield letterbox_chw(img, shape)[None]


def pick_calib_images(data: Path, n: int, conds=("night", "night_adverse"), seed: int = 0) -> list[str]:
    """Train bölümünden (test DEĞİL) rastgele gece kareleri."""
    names = []
    for c in conds:
        f = data / "splits" / "train" / f"{c}.txt"
        if f.exists():
            names += f.read_text(encoding="utf-8").split()
    if not names:
        raise SystemExit(f"{data}/splits/train/ altında gece listesi yok.")
    random.Random(seed).shuffle(names)
    return [str(data / "images" / "train" / x) for x in names[:n]]


# ---------------------------------------------------------------- ONNX meta verisi
def onnx_meta(onnx_path: Path) -> dict:
    import onnx

    m = onnx.load(str(onnx_path), load_external_data=False)
    meta = {p.key: p.value for p in m.metadata_props}
    shp = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]   # [batch, 3, h, w]
    meta["batch"] = str(shp[0])
    meta["imgsz"] = str([shp[2], shp[3]])
    return meta


def meta_for_json(meta: dict) -> dict:
    """Ultralytics engine meta verisi JSON'dur; ONNX'teki metin değerleri (ör. names) Ultralytics
    tarafından ayrıştırılır. imgsz ve batch'i sayı olarak yazıyoruz ki adb.models.engine_meta okuyabilsin."""
    out = dict(meta)
    out["imgsz"] = json.loads(meta["imgsz"])
    out["batch"] = int(meta["batch"])
    return out


# ---------------------------------------------------------------- TensorRT
def build(onnx_path: Path, out_path: Path, precision: str, kind: str, calib_paths: list[str] | None,
          slicing_cfg: dict | None, workspace_gb: float, fp16_layers: list[str], raw: bool,
          cache_path: Path | None) -> Path:
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise SystemExit(f"ONNX okunamadı: {onnx_path}\n{errs}")
    inp = network.get_input(0)
    if inp.dtype != trt.float32:
        raise SystemExit(f"{onnx_path.name}: giriş {inp.dtype}, FP32 bekleniyor. Dosyayı silip "
                         f"deploy/export.py ile yeniden üret.")
    b, _, h, w = inp.shape
    print(f"{onnx_path.name}: giriş {tuple(inp.shape)}, {network.num_layers} katman")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    config.set_flag(trt.BuilderFlag.FP16)

    calibrator = None
    if precision == "int8":
        if not calib_paths:
            raise SystemExit("INT8 için kalibrasyon görüntüleri gerekli.")
        config.set_flag(trt.BuilderFlag.INT8)

        class Calibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self):
                super().__init__()
                self.it = calib_batches(calib_paths, kind, (h, w), slicing_cfg)
                self.buf = torch.empty((b, 3, h, w), dtype=torch.float32, device="cuda")
                self.n = 0

            def get_batch_size(self):
                return b

            def get_batch(self, names):
                try:
                    x = next(self.it)
                except StopIteration:
                    return None
                if x.shape[0] != b:
                    raise RuntimeError(f"kalibrasyon batch {x.shape[0]}, motor batch {b}")
                self.buf.copy_(torch.from_numpy(x))
                self.n += 1
                if self.n % 50 == 0:
                    print(f"\r  kalibrasyon: {self.n}/{len(calib_paths)}", end="", flush=True)
                return [int(self.buf.data_ptr())]

            def read_calibration_cache(self):
                if cache_path and cache_path.exists():
                    print(f"  kalibrasyon önbelleği kullanılıyor: {cache_path}")
                    return cache_path.read_bytes()
                return None

            def write_calibration_cache(self, cache):
                if cache_path:
                    cache_path.write_bytes(bytes(cache))

        calibrator = Calibrator()
        config.int8_calibrator = calibrator

        if fp16_layers:
            n = 0
            for i in range(network.num_layers):
                L = network.get_layer(i)
                if any(s in L.name for s in fp16_layers):
                    L.precision = trt.float16
                    n += 1
            config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            print(f"  FP16'da tutulan katman: {n} ({', '.join(fp16_layers)})")

    t0 = time.time()
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise SystemExit("Motor derlenemedi (ayrıntı yukarıdaki TensorRT uyarılarında).")
    if calibrator is not None:
        print()
    meta = None if raw else meta_for_json(onnx_meta(onnx_path))
    if meta is not None:
        meta["precision"] = precision
        meta["tensorrt"] = trt.__version__
        meta["gpu"] = torch.cuda.get_device_name(0)
    write_engine_file(out_path, bytes(ser), meta)
    print(f"  -> {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f} sn)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--precision", choices=["fp16", "int8"], default="fp16")
    ap.add_argument("--kind", choices=["full", "band", "val"], help="boş: ONNX adından (full/band/val)")
    ap.add_argument("--out", type=Path, help="boş: <onnx klasörü>/<kind>_<precision>.engine")
    ap.add_argument("--data", type=Path, default=ROOT / "data/bdd_adb")
    ap.add_argument("--calib-n", type=int, default=500)
    ap.add_argument("--fp16-layers", nargs="*", default=[],
                    help="INT8'de FP16 tutulacak katman adı parçaları, ör. /model.23/ (tespit kafası)")
    ap.add_argument("--workspace", type=float, default=4.0, help="GB")
    ap.add_argument("--raw", action="store_true", help="meta verisiz (trtexec --loadEngine için)")
    a = ap.parse_args()

    kind = a.kind or a.onnx.stem.split("_")[0]
    if kind not in ("full", "band", "val"):
        raise SystemExit("--kind belirt (full / band / val)")
    suffix = "_raw" if a.raw else ""
    out = a.out or a.onnx.parent / f"{kind}_{a.precision}{suffix}.engine"
    calib = slicing = cache = None
    if a.precision == "int8":
        calib = pick_calib_images(a.data, a.calib_n)
        print(f"Kalibrasyon: train bölümünden {len(calib)} gece karesi"
              + (" (bant parçaları)" if kind == "band" else ""))
        if kind == "band":
            from adb.slicing import load_slicing
            slicing = load_slicing()
        cache = a.onnx.parent / f"{kind}_int8_calib.cache"
    build(a.onnx, out, a.precision, kind, calib, slicing, a.workspace, a.fp16_layers, a.raw, cache)


if __name__ == "__main__":
    main()
