"""Kamaşma ihlali teşhisi: far kararının kamaştırdığı her gerçek araç için sebep + görsel.

eval_video.py ile aynı videolar ve aynı işlem hattı (tespit -> [takip] -> far kararı). Etiketli karede
açık bir segmentte kalan her gerçek araç (far kapsamı içinde) şu gruplardan birine ayrılır:

  görülmedi        : etiketli karede bu araca ait hiçbir tespit yok (araç kutusu ya da içinde ışık, güven >= 0,10)
  düşük güven      : tespit var ama güveni far kararı eşiğinin altında
  takip onayı      : ham tespit eşiği geçiyor ama takipçi henüz onaylamamış / çıktıya vermemiş
  yalnız ışık      : karar girdisinde araç kutusu yok, yalnızca ışığı var; ışığın açı aralığı aracın
                     tamamını kapsamıyor (ör. yalnız bir arka lamba)
  geometri         : araç kutusu eşiği geçen bir tespitle bulunmuş, yine de segmenti açık
                     (açı/paralaks hatası, kutu kayması, güvenlik payı yetersiz)

Her ihlal için görsel: etiketli kare, ihlal edilen araç (kırmızı), diğer araçlar (yeşil), karar
girdisi tespitler (sarı), üstte 16 segmentlik far maskesi (açık = sarı, kısık = koyu).

Kullanım:
  python eval/diagnose_glare.py --weights runs/baseline/weights/best.pt --scenes highway residential
  python eval/diagnose_glare.py --weights runs/baseline/weights/best.pt            # tüm gece videoları
Çıktı: runs/<ad>/glare_diag/*.jpg + ekranda tablo
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from adb.decision import LIGHTS, VEH, BeamController, glare_and_utilization  # noqa: E402
from adb.geometry import load_config  # noqa: E402
from adb.tracking import Tracker, iou_matrix  # noqa: E402
from eval_distance import read_gt  # noqa: E402
from eval_video import resolve  # noqa: E402

CATS = ["görülmedi", "düşük güven", "takip onayı", "yalnız ışık", "geometri"]
DIST = [("< 62 m", 32, 1e9), ("62–125 m", 16, 32), ("125 m+", 0, 16)]


def support(g, xyxy, cls, conf):
    """g kutusuna ait tespitler: araç kutusu (IoU >= 0,3) veya merkezi kutunun içinde olan ışık.
    Döner: (en yüksek araç güveni, en yüksek ışık güveni); yoksa 0."""
    if len(xyxy) == 0:
        return 0.0, 0.0
    veh = np.isin(cls, VEH)
    cv = 0.0
    if veh.any():
        iou = iou_matrix(g[None], xyxy[veh])[0]
        cv = float(conf[veh][iou >= 0.3].max()) if (iou >= 0.3).any() else 0.0
    lt = np.isin(cls, LIGHTS)
    cl = 0.0
    if lt.any():
        cx, cy = (xyxy[lt, 0] + xyxy[lt, 2]) / 2, (xyxy[lt, 1] + xyxy[lt, 3]) / 2
        ins = (cx >= g[0]) & (cx <= g[2]) & (cy >= g[1]) & (cy <= g[3])
        cl = float(conf[lt][ins].max()) if ins.any() else 0.0
    return cv, cl


def lit_flags(img, gt, bright=200, pad=2):
    """Her gerçek araç için: lambası yanıyor mu? Kutunun alt %75'inde parlak nokta (gri >= bright) var mı.
    Otomatik lamba etiketleyicisiyle aynı kural (scripts/autolabel_lights.py). Test etiketlerinde lamba
    sınıfı olmadığı için görüntüden ölçülür. Yansıma da parlak görünebilir; görsellerle kontrol edilmeli."""
    import cv2
    if len(gt) == 0:
        return np.zeros(0, bool)
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    h, w = g.shape
    out = []
    for b in gt:
        x0, x1 = max(int(b[0]) - pad, 0), min(int(np.ceil(b[2])) + pad, w)
        y0 = max(int(b[1] + 0.25 * (b[3] - b[1])) - pad, 0)
        y1 = min(int(np.ceil(b[3])) + pad, h)
        roi = g[y0:y1, x0:x1]
        out.append(bool(roi.size) and int(roi.max()) >= bright)
    return np.array(out, bool)


def classify(g, raw, dec, thr, tracked):
    rv, rl = support(g, *raw)
    if max(rv, rl) < 0.10:
        return "görülmedi"
    if max(rv, rl) < thr:
        return "düşük güven"
    dv, dl = support(g, *dec)
    if max(dv, dl) < thr:
        return "takip onayı" if tracked else "düşük güven"
    return "geometri" if dv >= thr else "yalnız ışık"


def draw(img, gt, vi, dec, thr, mask, cam, edges, title):
    import cv2
    im = img.copy()
    for k, b in enumerate(gt):
        col, t = ((0, 0, 255), 3) if k == vi else ((0, 200, 0), 1)
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, t)
    xyxy, cls, conf = dec
    for b, c, s in zip(xyxy, cls, conf):
        if s >= thr and int(c) in VEH + LIGHTS:
            cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 220, 255), 1)
    h, w = im.shape[:2]
    for (a0, a1), dim in zip(edges, mask):                     # segmentleri görüntü x eksenine yerleştir
        x0 = int(np.clip(cam.cx_px + cam.fx_px * math.tan(math.radians(a0)), 0, w - 1))
        x1 = int(np.clip(cam.cx_px + cam.fx_px * math.tan(math.radians(a1)), 0, w - 1))
        cv2.rectangle(im, (x0, 0), (x1, 14), (40, 40, 40) if dim else (0, 220, 255), -1)
        cv2.rectangle(im, (x0, 0), (x1, 14), (90, 90, 90), 1)
    b = gt[vi]
    pad, z = 60, 3                                             # ihlal edilen aracın 3x büyütülmüş yakın görünümü
    x0, y0 = max(int(b[0]) - pad, 0), max(int(b[1]) - pad, 0)
    x1, y1 = min(int(b[2]) + pad, w), min(int(b[3]) + pad, h)
    crop = cv2.resize(img[y0:y1, x0:x1], None, fx=z, fy=z, interpolation=cv2.INTER_NEAREST)   # ham kareden
    cv2.rectangle(crop, (int((b[0] - x0) * z) - 2, int((b[1] - y0) * z) - 2),
                  (int((b[2] - x0) * z) + 2, int((b[3] - y0) * z) + 2), (0, 0, 255), 1)
    for d, c, s in zip(xyxy, cls, conf):                       # karar girdisi tespitler, yakın görünümde
        if s >= thr and int(c) in VEH + LIGHTS and d[2] > x0 and d[0] < x1 and d[3] > y0 and d[1] < y1:
            cv2.rectangle(crop, (int((d[0] - x0) * z), int((d[1] - y0) * z)),
                          (int((d[2] - x0) * z), int((d[3] - y0) * z)), (0, 220, 255), 1)
    ch, cw = crop.shape[:2]
    if ch < h - 30 and cw < w // 2:
        im[h - ch:h, w - cw:w] = crop
        cv2.rectangle(im, (w - cw, h - ch), (w - 1, h - 1), (255, 255, 255), 1)
    ascii_title = title.translate(str.maketrans("çğıöşüÇĞİÖŞÜ–", "cgiosuCGIOSU-"))   # OpenCV yazısı Türkçe harf çizemez
    cv2.putText(im, ascii_title, (8, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".pt veya tam kare .engine")
    ap.add_argument("--band-weights", help="bant motoru (.engine); yalnızca bantlı modlarda ve motorla gerekir")
    ap.add_argument("--backend", choices=["ultralytics", "trt"], default="ultralytics",
                    help="trt: motorlar doğrudan, ön/son işleme GPU'da (adb/trt_runtime.py)")
    ap.add_argument("--videos", type=Path, default=ROOT / "data/bdd_video/videos.json")
    ap.add_argument("--data", type=Path, default=ROOT / "data/bdd_adb")
    ap.add_argument("--conds", nargs="+", default=["night", "night_adverse"])
    ap.add_argument("--scenes", nargs="+")
    ap.add_argument("--mode", default="takip·tam", choices=["tam", "takip·tam", "tam+bant", "takip·tam+bant"])
    ap.add_argument("--conf", type=float, help="far kararı güven eşiği (boş: system.yaml)")
    ap.add_argument("--release", type=float, help="histerezis (boş: system.yaml)")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--imgsz", type=int, default=1280)
    a = ap.parse_args()

    import cv2
    from adb.models import load_detectors, run_dir_for
    from adb.slicing import load_slicing, merge, predict_band, predict_full

    sysc = load_config()
    thr = a.conf if a.conf is not None else sysc["decision"]["min_confidence"]
    vids = {k: v for k, v in json.loads(a.videos.read_text(encoding="utf-8")).items() if v["cond"] in a.conds}
    if a.scenes:
        sc = json.loads((a.data / "scenes.json").read_text(encoding="utf-8"))
        vids = {k: v for k, v in vids.items() if sc.get(k) in a.scenes}
    use_band, tracked = "bant" in a.mode, a.mode.startswith("takip")
    cfg = load_slicing(ROOT / "configs/slicing.yaml") if use_band else None
    model, band_model = load_detectors(a.weights, a.band_weights, a.imgsz, need_band=use_band, backend=a.backend)
    out_dir = run_dir_for(a.weights, backend=a.backend) / "glare_diag"
    out_dir.mkdir(exist_ok=True)
    for f in out_dir.glob("*.jpg"):
        f.unlink()
    ref = BeamController(sysc, thr, a.release)
    print(f"{len(vids)} video | mod {a.mode} | güven ≥ {thr} | histerezis {ref.release} sn")

    cats = {(d[0], lit): Counter() for d in DIST for lit in (True, False)}
    n_in = Counter()
    rows = []
    for vi_, (name, v) in enumerate(vids.items(), 1):
        frames = sorted(resolve(v["frames_dir"]).glob(f"{name}-*.jpg"))
        key = v["key_index"]
        seq = frames[max(0, key - a.warmup):key + 1]
        tr = Tracker() if tracked else None
        bc = BeamController(sysc, thr, a.release)
        img = None
        for i, f in enumerate(seq):
            img = cv2.imread(str(f))
            raw = predict_full(model, img, 0.10, a.imgsz)
            if use_band:
                raw = merge(raw, predict_band(band_model, img, cfg, 0.10))
            dec = tr.update(*raw)[:3] if tracked else raw
            mask = bc.step(*dec, i / a.fps)
        lbl = a.data / "labels" / v.get("split", "test") / f"{name}.txt"
        gt, gt_all = read_gt(lbl), read_gt(lbl, range(6))
        lit = lit_flags(img, gt)
        viol, inb, _, _ = glare_and_utilization(mask, gt, gt_all, bc.cam, bc.edges, bc.lamps)
        w = gt[:, 2] - gt[:, 0] if len(gt) else np.zeros(0)
        for k in range(len(gt)):
            if not inb[k]:
                continue
            dname = next(d for d, lo, hi in DIST if lo <= w[k] < hi)
            n_in[(dname, bool(lit[k]))] += 1
            if not viol[k]:
                continue
            cat = classify(gt[k], raw, dec, thr, tracked)
            cats[(dname, bool(lit[k]))][cat] += 1
            dist_m = bc.cam.fx_px * 1.8 / max(w[k], 1)
            ltxt = "isikli" if lit[k] else "isiksiz"
            title = f"{cat} | {ltxt} | ~{dist_m:.0f} m | {name}"
            fn = out_dir / f"{ltxt}_{cat.replace(' ', '_')}_{name}_{k}.jpg"
            cv2.imwrite(str(fn), draw(img, gt, k, dec, thr, mask, bc.cam, bc.edges, title))
            rows.append({"video": name, "gt_index": k, "cat": cat, "lit": bool(lit[k]), "dist_m": round(dist_m),
                         "image": fn.name})
        print(f"\r  {vi_}/{len(vids)}", end="", flush=True)
    print()

    print(f"\n### Kamaşma ihlali sebepleri ({a.mode}, güven ≥ {thr}, histerezis {ref.release} sn)\n")
    print("Işıklı araç: kutusunun alt %75'inde parlak nokta (lamba) olan araç; içinde sürücü olan, ADB'nin")
    print("korumakla yükümlü olduğu araç. Işıksız: park etmiş / lambası görünmeyen. Yansımalar da parlak")
    print("görünebilir, görsellerle kontrol et.\n")
    print("| Araç | Mesafe | Far kapsamındaki | İhlal | Oran | " + " | ".join(CATS) + " |")
    print("|---|---|---|---|---|" + "---|" * len(CATS))
    for lt in (True, False):
        tot, ntot = Counter(), 0
        for d, _, _ in DIST:
            c, n = cats[(d, lt)], n_in[(d, lt)]
            tot.update(c)
            ntot += n
            print(f"| {'ışıklı' if lt else 'ışıksız'} | {d} | {n} | {sum(c.values())} | "
                  f"{sum(c.values()) / max(n, 1):.3f} | " + " | ".join(str(c[k]) for k in CATS) + " |")
        print(f"| **{'ışıklı' if lt else 'ışıksız'}** | **toplam** | {ntot} | {sum(tot.values())} | "
              f"**{sum(tot.values()) / max(ntot, 1):.3f}** | " + " | ".join(str(tot[k]) for k in CATS) + " |")
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nGörseller: {out_dir}  ({len(rows)} adet)")


if __name__ == "__main__":
    main()