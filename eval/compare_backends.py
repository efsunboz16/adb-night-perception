"""GPU hattı (--backend trt) ile Ultralytics motor yolu neden farklı sonuç veriyor? Adım adım ayırır.

Aynı tam kare motoru, aynı görüntüler. Her adımda yalnızca bir şey değişir:
  1. ön işleme  : GPU hattının giriş tensörü ile CPU referansı (Ultralytics LetterBox'ın aynısı)
  2. CUDA Graph : aynı giriş, motor CUDA Graph ile ve olmadan -> ham çıktı farkı
  3. son işleme : aynı ham çıktı, bizim decode() ile Ultralytics non_max_suppression()
  4. uçtan uca  : GpuPipeline.predict_full ile Ultralytics predict (kutu sayısı, eşleşme, skor farkı)

Kullanım:
  python eval/compare_backends.py --weights deploy/models/full_fp16.engine --n 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy"))


def match(a, b, thr=0.9):
    """İki kutu kümesi: aynı sınıf ve IoU ≥ thr ile açgözlü eşleşme. (eşleşen çiftler, a'da kalan, b'de kalan)"""
    from adb.tracking import greedy_match, iou_matrix

    if len(a[0]) == 0 or len(b[0]) == 0:
        return [], list(range(len(a[0]))), list(range(len(b[0])))
    sim = iou_matrix(a[0], b[0]) * (a[1][:, None] == b[1][None, :])
    pairs = greedy_match(sim, thr)
    ia, ib = {p[0] for p in pairs}, {p[1] for p in pairs}
    return pairs, [i for i in range(len(a[0])) if i not in ia], [j for j in range(len(b[0])) if j not in ib]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "deploy/models/full_fp16.engine"))
    ap.add_argument("--data", type=Path, default=ROOT / "data/bdd_adb")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--conf", type=float, default=0.10)
    a = ap.parse_args()

    import cv2
    import torch
    try:
        from ultralytics.utils.nms import non_max_suppression          # Ultralytics ≥ 8.3.x
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression

    from adb.models import load_detectors
    from adb.slicing import predict_full
    from adb.trt_runtime import NMS_IOU, TrtEngine, decode, letterbox_params, preprocess_full
    from build_engine import letterbox_chw

    names = (a.data / "splits/test/night.txt").read_text(encoding="utf-8").split()[:a.n]
    imgs = [cv2.imread(str(a.data / "images/test" / n)) for n in names]
    eg = TrtEngine(a.weights, cuda_graph=True)
    en = TrtEngine(a.weights, cuda_graph=False)
    _, H, W = eg.shape
    lb = letterbox_params(720, 1280, H, W)
    ult, _ = load_detectors(a.weights, None, need_band=False)
    gpu, _ = load_detectors(a.weights, None, need_band=False, backend="trt")
    predict_full(ult, imgs[0], a.conf)
    ua = ult.model.predictor.args
    print(f"{len(imgs)} gece test görüntüsü, güven ≥ {a.conf} | {a.weights}")
    print(f"NMS IoU: GPU hattı {NMS_IOU}, Ultralytics predict {ua.iou} (sınıftan bağımsız: {ua.agnostic_nms})")
    print(f"CUDA Graph: {'var' if eg.graph is not None else 'KURULAMADI'}")

    d_in, d_graph, n_dec = [], [], [0, 0, 0, 0]
    e2e = {"ult": 0, "gpu": 0, "eş": 0, "yalnız_ult": [], "yalnız_gpu": [], "skor": []}
    for img in imgs:
        # 1. ön işleme
        ref = torch.from_numpy(letterbox_chw(img, (H, W)))[None].cuda()
        t = torch.empty_like(ref)
        preprocess_full(torch.from_numpy(img).cuda(), t, lb)
        d_in.append(float((t - ref).abs().max()) * 255)
        # 2. CUDA Graph: aynı giriş
        eg.inp.copy_(ref)
        en.inp.copy_(ref)
        og, on = eg().clone(), en().clone()
        d_graph.append(float((og - on).abs().max()))
        # 3. son işleme: aynı ham çıktı (CUDA Graph'sız)
        (bx, cl, sc), = decode(on, a.conf)
        u = non_max_suppression(on, a.conf, NMS_IOU, max_det=300)[0]
        mine = (bx.cpu().numpy(), cl.cpu().numpy(), sc.cpu().numpy())
        theirs = (u[:, :4].cpu().numpy(), u[:, 5].cpu().numpy().astype(int), u[:, 4].cpu().numpy())
        p, ra, rb = match(mine, theirs, 0.99)
        n_dec[0] += len(mine[0]); n_dec[1] += len(theirs[0]); n_dec[2] += len(ra); n_dec[3] += len(rb)
        # 4. uçtan uca
        A, B = predict_full(ult, img, a.conf), predict_full(gpu, img, a.conf)
        p, ra, rb = match(A, B, 0.9)
        e2e["ult"] += len(A[0]); e2e["gpu"] += len(B[0]); e2e["eş"] += len(p)
        e2e["yalnız_ult"] += [float(A[2][i]) for i in ra]
        e2e["yalnız_gpu"] += [float(B[2][j]) for j in rb]
        e2e["skor"] += [float(B[2][j] - A[2][i]) for i, j in p]

    def q(x):
        return f"ort {np.mean(x):.3g}, en çok {np.max(x):.3g}" if len(x) else "-"

    print("\n1. Ön işleme farkı (gri seviye):", q(d_in))
    print("2. CUDA Graph ile / olmadan ham çıktı farkı:", q(d_graph))
    print(f"3. Son işleme, aynı çıktıdan: bizim {n_dec[0]} kutu, Ultralytics {n_dec[1]}; "
          f"yalnız bizde {n_dec[2]}, yalnız onda {n_dec[3]}")
    print(f"4. Uçtan uca: Ultralytics {e2e['ult']} kutu, GPU hattı {e2e['gpu']}, eşleşen {e2e['eş']}")
    print(f"   yalnız Ultralytics'te: {len(e2e['yalnız_ult'])} (skor {q(e2e['yalnız_ult'])})")
    print(f"   yalnız GPU hattında : {len(e2e['yalnız_gpu'])} (skor {q(e2e['yalnız_gpu'])})")
    print(f"   eşleşenlerde skor farkı (GPU − Ultralytics): {q(np.abs(e2e['skor']))}, "
          f"ortalama işaretli {np.mean(e2e['skor']) if e2e['skor'] else 0:+.4f}")


if __name__ == "__main__":
    main()
