"""Mesafe bazlı recall (REQ-F-01/02): araçlar hangi mesafeye kadar yakalanıyor?

Her test görüntüsünde modelin tahminleri alınır, her gerçek (BDD) araç kutusu genişliğine göre
mesafe grubuna konur ve yakalanıp yakalanmadığı sayılır. İki recall raporlanır:
  - kutu        : araç kutusu, sınıftan bağımsız bir araç tahminiyle IoU >= eşik
  - kutu+ışık   : yukarıdaki VEYA kutunun içinde tahmin edilmiş bir head/tail_light merkezi var
                  (ADB için yeterli: aracın ışığı görüldüyse o açı kısılabilir)
Aradaki fark, ışık sınıflarının uzak mesafede katkısını gösterir.

Mesafe ≈ fx * 1,8 m / kutu genişliği (configs/system.yaml kamerası; BDD kamerası farklı, yaklaşık).

Kullanım:
  python eval/eval_distance.py --weights runs/baseline/weights/best.pt
  python eval/eval_distance.py --weights ... --conds night --conf 0.1 0.25
  # TensorRT (Faz 5): tam kare ve bant ayrı motorlar
  python eval/eval_distance.py --weights deploy/models/full_fp16.engine --band-weights deploy/models/band_fp16.engine --conds night --slice
Çıktı: ekrana tablo + runs/<ad>/eval_distance[_<imgsz>].json + .png (imgsz 1280 değilse dosya adına eklenir).
Motorlarda: runs/trt/<motor adı>/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adb.geometry import Camera, load_config  # noqa: E402

VEH = (0, 1, 2, 3)          # car, truck, bus, motorcycle
LIGHTS = (6, 7)             # head_light, tail_light
BINS = [(4, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 64), (64, 10_000)]
CAR_W_M = 1.8
IMG_W, IMG_H = 1280, 720


def read_gt(lbl: Path, classes=VEH):
    if not lbl.exists():
        return np.zeros((0, 4))
    rows = []
    for ln in lbl.read_text(encoding="utf-8").splitlines():
        c, cx, cy, w, h = ln.split()
        if int(c) in classes:
            cx, cy, w, h = float(cx) * IMG_W, float(cy) * IMG_H, float(w) * IMG_W, float(h) * IMG_H
            rows.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return np.array(rows).reshape(-1, 4)


def iou(a, b):
    """a: (N,4), b: (M,4) -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match(gt, veh_pred, light_pred, iou_thr):
    """Her GT için (kutu yakalandı mı, kutu veya ışık yakalandı mı)."""
    n = len(gt)
    box_hit = np.zeros(n, bool)
    if n and len(veh_pred):
        m = iou(gt, veh_pred)
        used = set()
        for g in np.argsort(-(gt[:, 2] - gt[:, 0])):          # büyükten küçüğe, açgözlü eşleşme
            for p in np.argsort(-m[g]):
                if m[g, p] < iou_thr:
                    break
                if p not in used:
                    used.add(p)
                    box_hit[g] = True
                    break
    light_hit = np.zeros(n, bool)
    if n and len(light_pred):
        cx = (light_pred[:, 0] + light_pred[:, 2]) / 2
        cy = (light_pred[:, 1] + light_pred[:, 3]) / 2
        inside = ((cx[None] >= gt[:, 0:1] - 2) & (cx[None] <= gt[:, 2:3] + 2) &
                  (cy[None] >= gt[:, 1:2] - 2) & (cy[None] <= gt[:, 3:4] + 2))
        light_hit = inside.any(axis=1)
    return box_hit, box_hit | light_hit


def false_alarms(xyxy, cls, gt_all, margin=4.0):
    """Hiçbir gerçek nesne kutusuna (araç, bisiklet, yaya; payla) düşmeyen araç/ışık tahmini sayısı.
    ADB'de bunların her biri gereksiz kısılan bir segment demektir. Not: BDD'nin etiketlemediği
    uzak araçlar da burada yanlış alarm sayılır (ADR-007), yani değer gerçeğinden kötü görünür."""
    sel = np.isin(cls, VEH + LIGHTS)
    if not sel.any():
        return 0
    b = xyxy[sel]
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    if len(gt_all) == 0:
        return int(sel.sum())
    inside = ((cx[:, None] >= gt_all[None, :, 0] - margin) & (cx[:, None] <= gt_all[None, :, 2] + margin) &
              (cy[:, None] >= gt_all[None, :, 1] - margin) & (cy[:, None] <= gt_all[None, :, 3] + margin))
    return int((~inside.any(axis=1)).sum())


def accumulate(st, gt, xyxy, cls, conf, confs, iou_thr, gt_all=None):
    if gt_all is not None:
        for c in confs:
            k = conf >= c
            st[c]["fp"][0] += false_alarms(xyxy[k], cls[k], gt_all)
            st[c]["img"][0] += 1
    if len(gt) == 0:
        return
    widths = gt[:, 2] - gt[:, 0]
    bin_idx = np.full(len(gt), -1)
    for b, (lo, hi) in enumerate(BINS):
        bin_idx[(widths >= lo) & (widths < hi)] = b
    for c in confs:
        keep = conf >= c
        box_hit, any_hit = match(gt, xyxy[keep & np.isin(cls, VEH)], xyxy[keep & np.isin(cls, LIGHTS)], iou_thr)
        for b in range(len(BINS)):
            sel = bin_idx == b
            st[c]["gt"][b] += sel.sum()
            st[c]["box"][b] += box_hit[sel].sum()
            st[c]["any"][b] += any_hit[sel].sum()


def new_stats(confs):
    return {c: {**{k: np.zeros(len(BINS), int) for k in ("gt", "box", "any")},
                "fp": np.zeros(1, int), "img": np.zeros(1, int)} for c in confs}


def run_sliced(model, band_model, paths, data, a, min_conf, cond):
    """Her görüntüde: tam kare bir kez, bant parçaları ek olarak. İki mod aynı görüntülerde ölçülür."""
    import cv2
    from adb.slicing import load_slicing, merge, predict_band, predict_full

    cfg_path = ROOT / "configs" / "slicing.yaml"
    if not cfg_path.exists():
        sys.exit("configs/slicing.yaml yok. Önce çalıştır: python scripts/horizon_band.py")
    cfg = load_slicing(cfg_path)
    st_full, st_band = new_stats(a.conf), new_stats(a.conf)
    t_full, t_band, t0 = [], [], time.time()
    for i, p in enumerate(paths, 1):
        img = cv2.imread(p)
        if img is None:
            continue
        s0 = time.perf_counter()
        full = predict_full(model, img, min_conf, a.imgsz)
        s1 = time.perf_counter()
        band = predict_band(band_model, img, cfg, min_conf)
        merged = merge(full, band)
        s2 = time.perf_counter()
        if i > 10:                                   # ilk kareler ısınma, süreye katılmaz
            t_full.append(s1 - s0)
            t_band.append(s2 - s1)
        lbl = data / "labels" / "test" / f"{Path(p).stem}.txt"
        gt, gt_all = read_gt(lbl), read_gt(lbl, range(6))
        accumulate(st_full, gt, *full, a.conf, a.iou, gt_all=gt_all)
        accumulate(st_band, gt, *merged, a.conf, a.iou, gt_all=gt_all)
        if i % 250 == 0 or i == len(paths):
            print(f"\r  [{cond}] {i}/{len(paths)}  ({time.time() - t0:.0f} sn)", end="", flush=True)
    print()
    if t_full:
        tf, tb = 1000 * np.mean(t_full), 1000 * np.mean(t_band)
        print(f"  [{cond}] süre/kare: tam kare {tf:.1f} ms, bant ek {tb:.1f} ms -> toplam {tf + tb:.1f} ms "
              f"(okuma + ön/son işlem dahil)")
    return {f"{cond}·tam": st_full, f"{cond}·bant": st_band}


def iter_predictions(model, src, paths, conf, batch):
    """(yol, (xyxy, cls, conf)) üretir. Ultralytics: metin dosyasından akış; GPU hattı: kare kare."""
    if getattr(model, "adb_gpu_pipeline", False):
        import cv2
        for p in paths:
            img = cv2.imread(p)
            if img is not None:
                yield p, model.predict_full(img, conf)
        return
    for r in model.model.predict(str(src), imgsz=model.imgsz, conf=conf, half=True,
                                 batch=model.batch or batch, stream=True, verbose=False):
        b = r.boxes
        yield r.path, (b.xyxy.cpu().numpy(), b.cls.cpu().numpy().astype(int), b.conf.cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".pt veya tam kare .engine")
    ap.add_argument("--band-weights", help="bant motoru (.engine); .pt için gerekmez")
    ap.add_argument("--backend", choices=["ultralytics", "trt"], default="ultralytics",
                    help="trt: motorlar doğrudan, ön/son işleme GPU'da (adb/trt_runtime.py)")
    ap.add_argument("--data", default=str(ROOT / "data/bdd_adb"))
    ap.add_argument("--conds", nargs="+", default=["day", "night", "night_adverse"])
    ap.add_argument("--conf", type=float, nargs="+", default=[0.1, 0.25],
                    help="raporlanacak güven eşikleri (ADB için düşük eşik daha güvenli)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="koşul başına en fazla görüntü (0 = hepsi)")
    ap.add_argument("--slice", action="store_true",
                    help="tam kare ile tam kare + ufuk bandı dilimlemeyi aynı görüntülerde karşılaştır "
                         "(configs/slicing.yaml gerekir: scripts/horizon_band.py)")
    a = ap.parse_args()

    from adb.models import load_detectors, run_dir_for
    model, band_model = load_detectors(a.weights, a.band_weights, a.imgsz, need_band=a.slice, backend=a.backend)
    print(f"Tam kare: {model.describe()}" + (f" | bant: {band_model.describe()}" if a.slice else ""))
    fx = Camera.from_config(load_config()).fx_px
    data = Path(a.data)
    run_dir = run_dir_for(a.weights, backend=a.backend)
    tag = "" if a.imgsz == 1280 else f"_{a.imgsz}"          # önceki sonuçların üzerine yazmasın
    min_conf = min(a.conf)

    # stats[cond][conf] = {"gt": [bin], "box": [bin], "any": [bin]}
    stats = {}
    for cond in a.conds:
        lst = data / "splits" / "test" / f"{cond}.txt"
        if not lst.exists():
            print(f"atlandı: {lst} yok")
            continue
        names = lst.read_text(encoding="utf-8").split()
        if a.limit:
            names = names[:a.limit]
        paths = [str((data / "images" / "test" / n).resolve()) for n in names]
        # Yolları liste olarak vermek Ultralytics'in TÜM görüntüleri önce belleğe yüklemesine yol açar
        # (MemoryError). Metin dosyası verilince görüntüler tek tek okunur.
        src = run_dir / f"eval_distance_{cond}.txt"
        src.write_text("\n".join(paths), encoding="utf-8")
        if a.slice:
            stats.update(run_sliced(model, band_model, paths, data, a, min_conf, cond))
            continue
        st = new_stats(a.conf)
        t0 = time.time()
        for i, (pth, dets) in enumerate(iter_predictions(model, src, paths, min_conf, a.batch), 1):
            lbl = data / "labels" / "test" / f"{Path(pth).stem}.txt"
            accumulate(st, read_gt(lbl), *dets, a.conf, a.iou, gt_all=read_gt(lbl, range(6)))
            if i % 500 == 0 or i == len(paths):
                print(f"\r  [{cond}] {i}/{len(paths)}  ({time.time() - t0:.0f} sn)", end="", flush=True)
        print()
        stats[cond] = st

    # ---------------- tablo
    def dist_label(lo, hi):
        far, near = fx * CAR_W_M / lo, fx * CAR_W_M / hi
        return f"< {far:.0f}" if hi > 1000 else f"{near:.0f}–{far:.0f}"

    out = {"iou": a.iou, "fx": fx, "bins_px": BINS, "results": {}}
    for c in a.conf:
        print(f"\n### Recall, güven eşiği {c}, IoU {a.iou}  (kutu / kutu+ışık)\n")
        hdr = "| Genişlik (px) | ~Mesafe (m) | " + " | ".join(stats) + " |"
        print(hdr + "\n" + "|---" * (2 + len(stats)) + "|")
        for b, (lo, hi) in enumerate(BINS):
            cells = []
            for cond, st in stats.items():
                g = st[c]["gt"][b]
                if g == 0:
                    cells.append("-")
                    continue
                cells.append(f"{st[c]['box'][b] / g:.2f} / {st[c]['any'][b] / g:.2f} (n={g})")
            lbl = f"≥ {lo}" if hi > 1000 else f"{lo}–{hi}"
            print(f"| {lbl} | {dist_label(lo, hi)} | " + " | ".join(cells) + " |")
        fps = [f"{st[c]['fp'][0] / max(st[c]['img'][0], 1):.2f}" for st in stats.values()]
        print("| **Yanlış alarm / kare** | | " + " | ".join(fps) + " |")
        out["results"][str(c)] = {cond: {k: v.tolist() for k, v in st[c].items()} for cond, st in stats.items()}

    (run_dir / f"eval_distance{tag}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # ---------------- grafik
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        c = a.conf[0]
        centers = [fx * CAR_W_M / np.sqrt(lo * min(hi, 128)) for lo, hi in BINS]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for cond, st in stats.items():
            g = st[c]["gt"].astype(float)
            ok = g >= 20                                   # az örnekli grupları çizme (gürültülü)
            g[~ok] = np.nan
            line, = ax.plot(centers, st[c]["any"] / g, "o-", label=f"{cond} · kutu+ışık")
            ax.plot(centers, st[c]["box"] / g, "o--", color=line.get_color(), alpha=0.6, label=f"{cond} · kutu")
        ax.axvline(220, color="gray", ls=":", lw=1)
        ax.text(222, 0.03, "220 m hedefi", color="gray", fontsize=9)
        ax.axhline(0.9, color="gray", ls=":", lw=1)
        ax.set_xlabel("Yaklaşık mesafe (m)")
        ax.set_ylabel(f"Recall (güven ≥ {c}, IoU ≥ {a.iou})")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(run_dir / f"eval_distance{tag}.png", dpi=130)
        print(f"\nGrafik: {run_dir / f'eval_distance{tag}.png'}")
    except Exception as e:
        print(f"Grafik çizilemedi: {e}")
    print(f"Sonuçlar: {run_dir / f'eval_distance{tag}.json'}")


if __name__ == "__main__":
    main()