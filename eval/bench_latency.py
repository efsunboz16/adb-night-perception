"""Faz 5 · Uçtan uca gecikme ölçümü, aşama aşama (REQ-P-05, REQ-P-06).

trtexec yalnızca motorun kendisini ölçer. Bu script ADB hattının Python'daki tamamını ölçer:

  tam kare   ön işleme + TensorRT çıkarım + NMS (Ultralytics predict)
  bant       4 parça kesme/büyütme + çıkarım + koordinat dönüşümü
  birleştir  tam kare + bant, sınıf bazında NMS
  takip      adb/tracking.py
  karar      adb/decision.py (paralaks, güvenlik payı, histerezis)
  izleyici   adb/safety.py

Kareler önceden belleğe okunur: disk okuma süresi ölçüme karışmaz (gerçek sistemde kare kameradan
gelir). İlk --warmup kare ısınmadır, sayılmaz. İki senaryo ölçülür: bant her karede (1) ve iki
karede bir (2). Ortalama değil yüzdelikler raporlanır: gecikme bütçesini kötü kareler aşar.

Ölçümden önce: bilgisayar prizde, Windows güç modu "En iyi performans", GPU kullanan diğer
uygulamalar kapalı.

Kullanım:
  python eval/bench_latency.py --weights deploy/models/full_fp16.engine --band-weights deploy/models/band_fp16.engine
  python eval/bench_latency.py --weights runs/baseline/weights/best.pt            # PyTorch karşılaştırması
  python eval/bench_latency.py --weights deploy/models/full_fp16.engine --band-weights deploy/models/band_fp16.engine --backend trt   # GPU hattı
Çıktı: ekrana tablo + runs/trt/<motor>/latency.json (veya runs/<ad>/latency.json)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adb.decision import BeamController  # noqa: E402
from adb.geometry import load_config  # noqa: E402
from adb.safety import SafetyMonitor  # noqa: E402
from adb.tracking import Tracker  # noqa: E402

STAGES = ["tam kare", "bant", "birleştir", "takip", "karar", "izleyici"]


def pct(x) -> dict:
    x = np.asarray(x, float) * 1000.0
    if len(x) == 0:
        return {}
    return {"mean": round(float(x.mean()), 2), "p50": round(float(np.percentile(x, 50)), 2),
            "p95": round(float(np.percentile(x, 95)), 2), "p99": round(float(np.percentile(x, 99)), 2),
            "max": round(float(x.max()), 2)}


def load_frames(a) -> tuple[list[np.ndarray], str]:
    """Video karelerinden ardışık bir dizi (takip gerçekçi çalışsın); yoksa gece test görüntüleri."""
    import cv2

    paths: list[Path] = []
    src = ""
    if a.videos.exists():
        vids = json.loads(a.videos.read_text(encoding="utf-8"))
        for name, v in vids.items():
            if v.get("cond") not in ("night", "night_adverse"):
                continue
            d = Path(v["frames_dir"].replace("\\", "/"))
            d = d if d.is_absolute() else ROOT / d
            paths += sorted(d.glob(f"{name}-*.jpg"))
            if len(paths) >= a.n:
                break
        src = f"gece videoları ({a.videos.name})"
    if not paths:
        lst = a.data / "splits" / "test" / "night.txt"
        paths = [a.data / "images" / "test" / n for n in lst.read_text(encoding="utf-8").split()]
        src = "gece test görüntüleri (bağımsız kareler)"
    frames = []
    for p in paths[:a.n]:
        img = cv2.imread(str(p))
        if img is not None:
            frames.append(img)
    if not frames:
        sys.exit("Kare bulunamadı (--videos / --data).")
    return frames, src


def run(frames, full, band, cfg, sysc, band_every: int, fps: float, warmup: int):
    from adb.slicing import merge, predict_band, predict_full

    conf = 0.10
    tr = Tracker()
    bc = BeamController(sysc, sysc["decision"]["min_confidence"])
    sm = SafetyMonitor(sysc)
    t = {s: [] for s in STAGES}
    total = []
    empty = (np.zeros((0, 4)), np.zeros(0, int), np.zeros(0))
    for i, img in enumerate(frames):
        ts = i / fps
        c0 = time.perf_counter()
        f = predict_full(full, img, conf)
        c1 = time.perf_counter()
        b = predict_band(band, img, cfg, conf) if i % band_every == 0 else empty
        c2 = time.perf_counter()
        m = merge(f, b)
        c3 = time.perf_counter()
        out = tr.update(*m)
        c4 = time.perf_counter()
        mask = bc.step(*out[:3], ts)
        c5 = time.perf_counter()
        ok = sm.frame(ts, mean01=float(img[::4, ::4].mean()) / 255.0)   # 1/16 örnekleme; tam kare ~3 ms
        SafetyMonitor.apply(mask, ok)
        c6 = time.perf_counter()
        if i < warmup:
            continue
        for s, d in zip(STAGES, (c1 - c0, c2 - c1, c3 - c2, c4 - c3, c5 - c4, c6 - c5)):
            t[s].append(d)
        total.append(c6 - c0)
    res = {s: pct(v) for s, v in t.items()}
    if band_every > 1:                                 # bant yalnızca işlendiği karelerde anlamlı
        bt = np.asarray(t["bant"])
        res["bant (işlenen kareler)"] = pct(bt[bt > 1e-4])
    res["TOPLAM"] = pct(total)
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help=".pt veya tam kare .engine")
    ap.add_argument("--band-weights", help="bant motoru (.engine); .pt için gerekmez")
    ap.add_argument("--backend", choices=["ultralytics", "trt"], default="ultralytics",
                    help="trt: motorlar doğrudan, ön/son işleme GPU'da (adb/trt_runtime.py)")
    ap.add_argument("--no-cuda-graph", action="store_true", help="--backend trt'de CUDA Graph kapalı (karşılaştırma)")
    ap.add_argument("--videos", type=Path, default=ROOT / "data/bdd_video/videos.json")
    ap.add_argument("--data", type=Path, default=ROOT / "data/bdd_adb")
    ap.add_argument("--n", type=int, default=600, help="ölçülecek kare sayısı (ısınma dahil)")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--band-every", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()

    import torch

    from adb.models import load_detectors, run_dir_for
    from adb.slicing import load_slicing

    sysc = load_config()
    fps = float(sysc["camera"]["fps"])
    budget = float(sysc["targets"]["end_to_end_latency_ms_max"])
    if a.backend == "trt" and a.no_cuda_graph:
        from adb import trt_runtime
        trt_runtime.DEFAULT_CUDA_GRAPH = False
    full, band = load_detectors(a.weights, a.band_weights, a.imgsz, backend=a.backend)
    cfg = load_slicing()
    frames, src = load_frames(a)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Tam kare: {full.describe()} | bant: {band.describe()}")
    print(f"{len(frames)} kare, {src} | {gpu} | ısınma {a.warmup} kare")

    results = {}
    for k in a.band_every:
        print(f"\nBant {'her karede' if k == 1 else f'{k} karede bir'} ...", flush=True)
        results[str(k)] = run(frames, full, band, cfg, sysc, k, fps, a.warmup)

    for k, res in results.items():
        print(f"\n### Bant {'her karede' if k == '1' else f'{k} karede bir'} (ms)\n")
        print("| Aşama | ortalama | p50 | p95 | p99 | en kötü |\n|---|---|---|---|---|---|")
        for s, v in res.items():
            if v:
                b = "**" if s == "TOPLAM" else ""
                print(f"| {b}{s}{b} | {v['mean']} | {v['p50']} | {b}{v['p95']}{b} | {v['p99']} | {v['max']} |")
        p95 = res["TOPLAM"]["p95"]
        print(f"\nREQ-P-06 (p95 ≤ {budget:.0f} ms): {'GEÇTİ' if p95 <= budget else 'KALDI'} ({p95} ms) · "
              f"{fps:.0f} fps kare aralığı {1000 / fps:.1f} ms: {'sığıyor' if p95 <= 1000 / fps else 'SIĞMIYOR'}")

    import platform
    out = {"weights": a.weights, "band_weights": a.band_weights or a.weights, "backend": a.backend,
           "cuda_graph": a.backend == "trt" and not a.no_cuda_graph, "gpu": gpu,
           "frames": len(frames), "source": src, "warmup": a.warmup, "torch": torch.__version__,
           "python": platform.python_version(), "budget_ms": budget,
           "note": "Python hattı; kamera yakalama ve disk okuma hariç. ultralytics: ön/son işleme CPU'da; "
            "trt: ön/son işleme GPU'da (adb/trt_runtime.py)",
           "results": results}
    try:
        import tensorrt as trt
        out["tensorrt"] = trt.__version__
    except ImportError:
        pass
    tag = "_nograph" if a.backend == "trt" and a.no_cuda_graph else ""
    p = run_dir_for(a.weights, backend=a.backend) / f"latency{tag}.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nKaydedildi: {p}")


if __name__ == "__main__":
    main()
