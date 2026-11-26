"""Faz 5 · Dışa aktarma: PyTorch modeli -> ONNX -> TensorRT motorları (ADR-019).

Üretilenler (deploy/models/, repoya girmez):
  full.onnx, band.onnx              taşınabilir; başka GPU'da (ör. Jetson) motor buradan derlenir
  full_<p>.engine                   tam kare, 736×1280, batch 1
  band_<p>.engine                   ufuk bandı, 704×768, batch 4 (bant parçası sayısı)
  val.onnx, val_<p>.engine          (--val) kare 1280×1280; yalnızca mAP için (eval_conditions.py)
  *_raw.engine                      (--raw) meta verisiz kopya, trtexec --loadEngine için
  manifest.json                     boyutlar, sürümler, GPU
<p>: fp16 veya int8. Boyutlar configs/system.yaml (kamera) ve configs/slicing.yaml'dan hesaplanır.

Kullanım:
  python deploy/export.py                                  # FP16, tam kare + bant
  python deploy/export.py --precision fp16 int8 --val      # FP16 ve INT8, mAP için kare motorlar da
  python deploy/export.py --precision int8 --fp16-layers /model.23/   # INT8, tespit kafası FP16
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy"))
from adb.geometry import load_config  # noqa: E402
from adb.models import band_input_shape, full_input_shape  # noqa: E402
from adb.slicing import load_slicing  # noqa: E402


def onnx_matches(path: Path, imgsz: tuple[int, int], batch: int) -> tuple[bool, str]:
    """Var olan ONNX doğru boyutta, FP32 girişli ve Ultralytics meta verili mi?
    (Ultralytics'in format=engine dışa aktarmasından kalan ONNX'ler farklı olabilir.)"""
    import onnx

    m = onnx.load(str(path), load_external_data=False)
    t = m.graph.input[0].type.tensor_type
    shp = [d.dim_value for d in t.shape.dim]
    if shp != [batch, 3, *imgsz]:
        return False, f"boyut {shp}"
    if t.elem_type != onnx.TensorProto.FLOAT:
        return False, "giriş FP32 değil"
    if not any(p.key == "names" for p in m.metadata_props):
        return False, "meta veri yok"
    return True, ""


def export_onnx(weights: str, imgsz: tuple[int, int], batch: int, out: Path) -> Path:
    from ultralytics import YOLO

    if out.exists():
        ok, why = onnx_matches(out, imgsz, batch)
        if ok:
            print(f"{out.name} zaten var ve uyumlu, yeniden üretilmiyor")
            return out
        print(f"{out.name} var ama uyumsuz ({why}), yeniden üretiliyor")
        out.unlink()
    f = YOLO(weights).export(format="onnx", imgsz=list(imgsz), batch=batch, dynamic=False, simplify=True,
                             half=False, device=0, verbose=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(f), out)
    print(f"  -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "runs/baseline/weights/best.pt"))
    ap.add_argument("--out", type=Path, default=ROOT / "deploy/models")
    ap.add_argument("--precision", nargs="+", choices=["fp16", "int8"], default=["fp16"])
    ap.add_argument("--val", action="store_true", help="mAP için kare 1280×1280 motor da üret")
    ap.add_argument("--raw", action="store_true", help="trtexec için meta verisiz kopyalar da üret")
    ap.add_argument("--data", type=Path, default=ROOT / "data/bdd_adb")
    ap.add_argument("--calib-n", type=int, default=500)
    ap.add_argument("--fp16-layers", nargs="*", default=[])
    ap.add_argument("--workspace", type=float, default=4.0)
    a = ap.parse_args()

    from build_engine import build, pick_calib_images

    cam = load_config()["camera"]
    H, W = cam["height_px"], cam["width_px"]
    slicing = load_slicing()
    full_shape = full_input_shape(H, W)
    band_shape, n_tiles = band_input_shape(H, W, slicing)
    kinds = {"full": (full_shape, 1), "band": (band_shape, n_tiles)}
    if a.val:
        kinds["val"] = ((1280, 1280), 1)
    print("Giriş boyutları: " + ", ".join(f"{k} {s[0]}×{s[1]} batch {b}" for k, (s, b) in kinds.items()))

    t0 = time.time()
    onnx = {}
    for k, (shape, b) in kinds.items():
        print(f"\n[ONNX] {k}")
        onnx[k] = export_onnx(a.weights, shape, b, a.out / f"{k}.onnx")

    calib = pick_calib_images(a.data, a.calib_n) if "int8" in a.precision else None
    built = []
    for p in a.precision:
        for k in kinds:
            print(f"\n[TensorRT] {k} {p}")
            cache = a.out / f"{k}_int8_calib.cache" if p == "int8" else None
            built.append(build(onnx[k], a.out / f"{k}_{p}.engine", p, "band" if k == "band" else "full",
                               calib, slicing if k == "band" else None, a.workspace, a.fp16_layers, False, cache))
            if a.raw:
                build(onnx[k], a.out / f"{k}_{p}_raw.engine", p, "band" if k == "band" else "full",
                      calib, slicing if k == "band" else None, a.workspace, a.fp16_layers, True, cache)

    import tensorrt as trt
    import torch
    import ultralytics

    man_p = a.out / "manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8")) if man_p.exists() else {}
    man.update({
        "weights": str(Path(a.weights)), "gpu": torch.cuda.get_device_name(0),
        "tensorrt": trt.__version__, "torch": torch.__version__, "ultralytics": ultralytics.__version__,
        "shapes": {k: {"imgsz": list(s), "batch": b} for k, (s, b) in kinds.items()},
        "fp16_layers_in_int8": a.fp16_layers, "calib_n": a.calib_n if calib else None,
        "engines": sorted({*man.get("engines", []), *(e.name for e in built)}),
    })
    man_p.write_text(json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nBitti ({time.time() - t0:.0f} sn). {man_p}")
    print("Sonraki adım:\n  python eval/eval_distance.py --weights deploy/models/full_fp16.engine "
          "--band-weights deploy/models/band_fp16.engine --conds night --slice")


if __name__ == "__main__":
    main()
