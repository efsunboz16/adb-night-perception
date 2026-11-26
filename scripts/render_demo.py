"""Faz 6 · Demo videosu: ADB hattının çıktısını görüntünün üzerine çizer.

Her karede: tespitler (araç yeşil, far beyaz, arka lamba kırmızı), iz numaraları, ufuk bandı,
üstte 16 segmentlik far maskesi (açık = sarı, kısık = koyu), sağ altta kuşbakışı far yelpazesi,
solda durum (gecikme, FPS, güvenlik izleyici). Güvenlik izleyici uzun farı kapattığında
"KISA FAR" uyarısı ve sebebi gösterilir.

Girdi: video dosyası (.mp4 ...), kare klasörü (*.jpg) veya videos.json'daki bir video adı.
Kare 1280×720'ye ölçeklenir (configs/system.yaml kamerası). Kendi çekimin farklı görüş açısındaysa
açılar yaklaşık olur; yatay görüş açısını --hfov ile ver.

Kullanım:
  python scripts/render_demo.py --source gece_yolu.mp4 --out runs/demo/demo.mp4
  python scripts/render_demo.py --source data/bdd100k/images/track/val/b1c9c847-3bda4659 --fps 5
  python scripts/render_demo.py --source b1c9c847-3bda4659            # videos.json'daki ad
Varsayılan: TensorRT GPU hattı (deploy/models/full_fp16.engine + band_fp16.engine).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adb.decision import BeamController  # noqa: E402
from adb.geometry import load_config  # noqa: E402
from adb.safety import SafetyMonitor  # noqa: E402
from adb.tracking import Tracker  # noqa: E402

W, H = 1280, 720
COL = {0: (80, 220, 80), 1: (80, 220, 80), 2: (80, 220, 80), 3: (80, 220, 80),
       4: (220, 180, 60), 5: (220, 180, 60), 6: (255, 255, 255), 7: (60, 60, 255)}
NAMES = {0: "car", 1: "truck", 2: "bus", 3: "moto", 4: "bike", 5: "person", 6: "far", 7: "stop"}
LIT, DIM = (0, 215, 255), (60, 60, 60)


def frames_from(source: str, fps_arg: float | None):
    """(kare üreteci, fps, ad)"""
    p = Path(source)
    if p.is_file():
        cap = cv2.VideoCapture(str(p))
        fps = fps_arg or cap.get(cv2.CAP_PROP_FPS) or 30.0

        def gen():
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield f
            cap.release()
        return gen(), fps, p.stem
    if not p.is_dir():                                     # videos.json'daki ad
        vids = json.loads((ROOT / "data/bdd_video/videos.json").read_text(encoding="utf-8"))
        if source not in vids:
            raise SystemExit(f"{source}: dosya, klasör veya videos.json'da video adı değil")
        d = Path(vids[source]["frames_dir"].replace("\\", "/"))
        p = d if d.is_absolute() else ROOT / d
        files = sorted(p.glob(f"{source}-*.jpg"))
    else:
        files = sorted(p.glob("*.jpg")) + sorted(p.glob("*.png"))
    if not files:
        raise SystemExit(f"{p}: kare yok")
    return (cv2.imread(str(f)) for f in files), fps_arg or 5.0, Path(source).name


def draw_beam_bar(img, mask, edges, ok):
    """Üstte 16 segmentlik far maskesi, açıya göre yerleştirilmiş."""
    x0, x1, y0, y1 = 240, W - 240, 8, 34
    lo, hi = edges[0][0], edges[-1][1]
    for (a, b), d in zip(edges, mask):
        u0 = int(x0 + (a - lo) / (hi - lo) * (x1 - x0))
        u1 = int(x0 + (b - lo) / (hi - lo) * (x1 - x0))
        cv2.rectangle(img, (u0 + 1, y0), (u1 - 1, y1), DIM if (d or not ok) else LIT, -1)
    cv2.rectangle(img, (x0, y0), (x1, y1), (200, 200, 200), 1)
    for ang in (-15, -10, -5, 0, 5, 10, 15):
        u = int(x0 + (ang - lo) / (hi - lo) * (x1 - x0))
        cv2.line(img, (u, y1), (u, y1 + 5), (200, 200, 200), 1)
        cv2.putText(img, f"{ang:+d}" if ang else "0", (u - 10, y1 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)


def draw_fan(img, mask, edges, ok, tracks, cam):
    """Sağ altta kuşbakışı: araç (alt orta), 16 dilimlik far yelpazesi, araçlar. Açılar okunaklı olsun
    diye K kat genişletilir (±16° -> ±40°); mesafe doğrusal, 250 m'ye kadar."""
    cx, cy, R, K = W - 160, H - 20, 175, 2.5
    over = img.copy()
    cv2.rectangle(over, (W - 320, H - 210), (W - 5, H - 5), (20, 20, 20), -1)
    cv2.addWeighted(over, 0.65, img, 0.35, 0, img)
    for (a, b), d in zip(edges, mask):
        pts = [(cx, cy)]
        for t in np.linspace(a, b, 6):
            r = math.radians(K * t)
            pts.append((int(cx + R * math.sin(r)), int(cy - R * math.cos(r))))
        cv2.fillPoly(img, [np.array(pts, np.int32)], (45, 45, 45) if (d or not ok) else (0, 160, 200))
        cv2.polylines(img, [np.array(pts, np.int32)], True, (90, 90, 90), 1)
    for b, c in tracks:
        if c > 3:
            continue
        az = cam.px_to_azimuth((b[0] + b[2]) / 2)
        dist = cam.fx_px * 1.8 / max(b[2] - b[0], 1)
        r = min(dist / 250.0, 1.0) * R
        p = (int(cx + r * math.sin(math.radians(K * az))), int(cy - r * math.cos(math.radians(K * az))))
        cv2.circle(img, p, 4, (80, 220, 80), -1)
    cv2.rectangle(img, (cx - 6, cy - 10), (cx + 6, cy + 4), (230, 230, 230), -1)
    cv2.putText(img, "250 m", (cx - 20, cy - R - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
    cv2.putText(img, "kusbakisi (aci x2,5)", (W - 312, H - 192), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", type=Path, help="boş: runs/demo/<ad>.mp4")
    ap.add_argument("--weights", default=str(ROOT / "deploy/models/full_fp16.engine"))
    ap.add_argument("--band-weights", default=str(ROOT / "deploy/models/band_fp16.engine"))
    ap.add_argument("--backend", choices=["ultralytics", "trt"], default="trt")
    ap.add_argument("--fps", type=float, help="kare klasörü için (BDD MOT: 5)")
    ap.add_argument("--hfov", type=float, help="kendi kameranın yatay görüş açısı (derece)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=0,
                    help="her N karede bir işle; boş: 30 fps'ten yüksek videolar 30 fps'e indirilir "
                         "(takip parametreleri kare cinsinden, FM-08)")
    ap.add_argument("--start", type=float, default=0.0, help="başlangıç saniyesi")
    ap.add_argument("--no-band-overlay", action="store_true")
    a = ap.parse_args()

    from adb.models import load_detectors
    from adb.slicing import load_slicing, merge, predict_band, predict_full

    cfg = load_config()
    if a.hfov:
        cfg["camera"]["hfov_deg"] = a.hfov
        cfg["camera"]["fx_px"] = None
    sl = load_slicing()
    full, band = load_detectors(a.weights, a.band_weights, backend=a.backend)
    gen, fps, name = frames_from(a.source, a.fps)
    stride = a.stride or max(1, round(fps / 30.0))
    fps_in, fps = fps, fps / stride
    out = a.out or ROOT / "runs/demo" / f"{name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    tr, sm = Tracker(), SafetyMonitor(cfg)
    bc = BeamController(cfg, cfg["decision"]["min_confidence"])
    y0, y1 = (int(round(v * H)) for v in sl["band_frac"])
    lat, n = [], 0
    print(f"{name}: {fps_in:.0f} fps -> {fps:.0f} fps işleniyor, {full.describe()} -> {out}")
    i = -1
    for k, img in enumerate(gen):
        if img is None or k % stride or k / fps_in < a.start:
            continue
        i += 1
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        t = i / fps
        t0 = time.perf_counter()
        dets = merge(predict_full(full, img, 0.10), predict_band(band, img, sl, 0.10))
        xyxy, cls, conf, ids = tr.update(*dets)
        mask = bc.step(xyxy, cls, conf, t)
        ok = sm.frame(t, mean01=float(img[::4, ::4].mean()) / 255.0)
        mask = SafetyMonitor.apply(mask, ok)
        lat.append(1000 * (time.perf_counter() - t0))

        vis = img.copy()
        if not a.no_band_overlay:
            cv2.rectangle(vis, (0, y0), (W - 1, y1), (120, 120, 60), 1)
        for b, c, s, k in zip(xyxy, cls, conf, ids):
            if s < bc.conf_thr:
                continue
            c = int(c)
            p1, p2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
            cv2.rectangle(vis, p1, p2, COL.get(c, (200, 200, 200)), 1 if c >= 6 else 2)
            if c < 6:
                cv2.putText(vis, f"{NAMES.get(c, c)} #{int(k)} {s:.2f}", (p1[0], max(p1[1] - 4, 50)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL.get(c, (200, 200, 200)), 1)
        draw_beam_bar(vis, mask, bc.edges, ok)
        draw_fan(vis, mask, bc.edges, ok, [(b, int(c)) for b, c, s in zip(xyxy, cls, conf) if s >= bc.conf_thr], bc.cam)
        p50 = float(np.median(lat[-30:]))
        lines = ["ADB Night Perception", f"gecikme {lat[-1]:.1f} ms (p50 {p50:.1f})",
                 f"uzun far: {int((~mask).sum())}/16 segment", f"t = {t:.1f} s"]
        for j, s in enumerate(lines):
            cv2.putText(vis, s, (12, 66 + 20 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        if not ok:
            why = ", ".join(sm.reasons) or "toparlanma"
            cv2.putText(vis, f"KISA FAR: {why}", (W // 2 - 170, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 255), 2, cv2.LINE_AA)
        vw.write(vis)
        n += 1
        if a.max_frames and n >= a.max_frames:
            break
        if n % 100 == 0:
            print(f"\r  {n} kare", end="", flush=True)
    vw.release()
    if lat:
        print(f"\n{n} kare yazıldı: {out} | hat gecikmesi p50 {np.median(lat):.1f} ms, "
              f"p95 {np.percentile(lat, 95):.1f} ms (çizim hariç)")


if __name__ == "__main__":
    main()
