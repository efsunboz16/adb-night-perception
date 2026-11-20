"""Sözde etiketleme (ADR-012): temel modelin uzakta emin olduğu tespitleri eğitim etiketlerine ekler.

Sorun (ADR-007): BDD'nin gece uzak araç etiketleri eksik; otomatik ışık etiketleri de sadece BDD
kutularının içinden üretildiği için o araçların ışıkları da etiketsiz. Model bu yüzden en çok
ihtiyacımız olan uzak ışıkları "arka plan" olarak bastırmayı öğreniyor.

Öğretmen: temel model + ufuk bandı dilimleme (adb/slicing.py). Sadece train/val gece ve alacakaranlık
karelerine uygulanır; test setine dokunulmaz.

Eklenen etiketler (hepsi mevcut etiketlerle çakışmıyorsa):
  A. BDD araç kutusunun alt %75'indeki head/tail_light tahmini (güven >= --light-conf)
  B. Hiçbir kutuya ait olmayan ışık: ufuk bandında, küçük ve AYNI SINIFTAN bir eşi yanında
     (aynı yükseklik, yakın yatay mesafe) -> far/arka lamba çifti. Tek ışık eklenmez (sokak lambası olabilir).
  C. Uzak araç (genişlik < --far-veh-px), güven >= --veh-conf, hiçbir BDD kutusuyla çakışmıyor.

Kullanım:
  python scripts/pseudo_label.py --preview 12        # önce gözle kontrol (data/bdd_adb_lights_pl/preview)
  python scripts/pseudo_label.py                     # tüm train/val (~30-60 dk)
Çıktı: data/bdd_adb_lights_pl/{labels,splits}  (görüntüler değişmez, eski veri setindekiler kullanılır)
"""

import argparse
import json
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VEH = (0, 1, 2, 3)
HEAD, TAIL = 6, 7
LIGHTS = (HEAD, TAIL)
GT_ALL = (0, 1, 2, 3, 4, 5)
CONDS = ("night", "night_adverse", "dusk", "dusk_adverse")
IMG_W, IMG_H = 1280, 720


# ------------------------------------------------------------------ saf seçim mantığı (test edilebilir)

def read_labels(lbl: Path):
    """-> (cls[N], xyxy[N,4]) piksel."""
    if not lbl.exists():
        return np.zeros(0, int), np.zeros((0, 4))
    c, b = [], []
    for ln in lbl.read_text(encoding="utf-8").splitlines():
        k, cx, cy, w, h = ln.split()
        cx, cy, w, h = float(cx) * IMG_W, float(cy) * IMG_H, float(w) * IMG_W, float(h) * IMG_H
        c.append(int(k))
        b.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return np.array(c, int), np.array(b).reshape(-1, 4)


def _centers(b):
    return (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2


def _inside(px, py, boxes, margin=0.0):
    """(P,) noktaların her biri için hangi kutuların içinde: (P, B) bool."""
    if len(boxes) == 0:
        return np.zeros((len(px), 0), bool)
    return ((px[:, None] >= boxes[None, :, 0] - margin) & (px[:, None] <= boxes[None, :, 2] + margin) &
            (py[:, None] >= boxes[None, :, 1] - margin) & (py[:, None] <= boxes[None, :, 3] + margin))


def _iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def select_pseudo(pred_xyxy, pred_cls, pred_conf, gt_cls, gt_xyxy, band_y, p, pred_peak=None):
    """Eklenecek sözde etiketleri seçer. -> list[(cls, xyxy, kural)]
    pred_peak: her tahmin kutusundaki en parlak piksel (RGB kanallarının en büyüğü, 0-255).
    Verilmezse parlaklık şartı uygulanmaz."""
    if pred_peak is None:
        pred_peak = np.full(len(pred_xyxy), 255.0)
    out = []
    veh_gt = gt_xyxy[np.isin(gt_cls, VEH)]
    any_gt = gt_xyxy[np.isin(gt_cls, GT_ALL)]
    light_gt = gt_xyxy[np.isin(gt_cls, LIGHTS)]
    y0, y1 = band_y

    # ---- ışıklar
    lm = np.isin(pred_cls, LIGHTS) & (pred_conf >= p["light_conf"])
    lb, lc, lpk = pred_xyxy[lm], pred_cls[lm], pred_peak[lm]
    if len(lb):
        cx, cy = _centers(lb)
        # mevcut ışık etiketiyle çakışan -> zaten var
        dup = _inside(cx, cy, light_gt, margin=p["dup_px"]).any(axis=1)
        # A: BDD araç kutusunun alt %75'inde
        lower = veh_gt.copy()
        if len(lower):
            lower[:, 1] = veh_gt[:, 1] + 0.25 * (veh_gt[:, 3] - veh_gt[:, 1])
        in_veh = _inside(cx, cy, lower, margin=2).any(axis=1)
        in_any = _inside(cx, cy, any_gt, margin=2).any(axis=1)
        w = lb[:, 2] - lb[:, 0]
        for i in range(len(lb)):
            if dup[i]:
                continue
            if in_veh[i]:
                # A varsayılan kapalı: park halindeki araçların lamba yansımalarını "ışık" diye
                # etiketliyordu; bu araçlar BDD'de zaten etiketli olduğu için recall katkısı da küçük.
                if p.get("rule_a", False):
                    out.append((int(lc[i]), lb[i], "A"))
                continue
            if in_any[i] or not (y0 <= cy[i] <= y1) or w[i] > p["light_max_px"]:
                continue
            if lpk[i] < p.get("min_peak", 0):          # sönük: reflektör / yol çivisi olabilir
                continue
            # B: aynı sınıftan bir eş var mı? (aynı yükseklik, yakın yatay mesafe, kendisi değil)
            same = (lc == lc[i]) & (np.arange(len(lb)) != i)
            dx, dy = np.abs(cx - cx[i]), np.abs(cy - cy[i])
            pair = same & (dx >= p["pair_dx"][0]) & (dx <= p["pair_dx"][1]) & (dy <= p["pair_dy"])
            pair &= ~_inside(cx, cy, any_gt, margin=2).any(axis=1)
            pair &= lpk >= p.get("min_peak", 0)
            if pair.any():
                out.append((int(lc[i]), lb[i], "B"))

    # ---- C: etiketsiz uzak araçlar
    vm = np.isin(pred_cls, VEH) & (pred_conf >= p["veh_conf"])
    vb, vc = pred_xyxy[vm], pred_cls[vm]
    if len(vb):
        w = vb[:, 2] - vb[:, 0]
        cx, cy = _centers(vb)
        ov = _iou(vb, any_gt).max(axis=1) if len(any_gt) else np.zeros(len(vb))
        inside_any = _inside(cx, cy, any_gt).any(axis=1)
        for i in range(len(vb)):
            if w[i] < p["far_veh_px"] and ov[i] < 0.1 and not inside_any[i]:
                out.append((int(vc[i]), vb[i], "C"))
    return out


def peak_brightness(img, xyxy):
    """Her kutudaki en parlak piksel değeri (kanalların en büyüğü; kırmızı lamba da 255'e çıkar)."""
    h, w = img.shape[:2]
    out = np.zeros(len(xyxy))
    for i, (x0, y0, x1, y1) in enumerate(xyxy):
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(np.ceil(x1)) + 1), min(h, int(np.ceil(y1)) + 1)
        if x1 > x0 and y1 > y0:
            out[i] = img[y0:y1, x0:x1].max()
    return out


def to_yolo_line(c, b):
    x0, y0, x1, y1 = np.clip(b, [0, 0, 0, 0], [IMG_W, IMG_H, IMG_W, IMG_H])
    return f"{c} {(x0 + x1) / 2 / IMG_W:.6f} {(y0 + y1) / 2 / IMG_H:.6f} {(x1 - x0) / IMG_W:.6f} {(y1 - y0) / IMG_H:.6f}"


# ------------------------------------------------------------------ ana akış

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/baseline/weights/best.pt"))
    ap.add_argument("--data", type=Path, default=ROOT / "data/bdd_adb_lights")
    ap.add_argument("--out", type=Path, default=ROOT / "data/bdd_adb_lights_pl")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--light-conf", type=float, default=0.45)
    ap.add_argument("--veh-conf", type=float, default=0.5)
    ap.add_argument("--far-veh-px", type=float, default=32, help="bu genişliğin altındaki araç 'uzak'")
    ap.add_argument("--light-max-px", type=float, default=16, help="kutusuz ışık en fazla bu genişlikte")
    ap.add_argument("--min-peak", type=float, default=200,
                    help="kutusuz ışığın en parlak pikseli en az bu olmalı (gerçek lambalar doygun)")
    ap.add_argument("--rule-a", action="store_true",
                    help="BDD kutusu içindeki ek ışıkları da ekle (varsayılan kapalı: park halindeki "
                         "araçlardaki yansımaları da etiketliyor)")
    ap.add_argument("--preview", type=int, default=0)
    a = ap.parse_args()
    p = {"light_conf": a.light_conf, "veh_conf": a.veh_conf, "far_veh_px": a.far_veh_px,
         "light_max_px": a.light_max_px, "dup_px": 3.0, "pair_dx": (3.0, 60.0), "pair_dy": 4.0,
         "min_peak": a.min_peak, "rule_a": a.rule_a}

    import cv2
    from ultralytics import YOLO
    from adb.slicing import load_slicing, merge, predict_band, predict_full

    cfg_path = ROOT / "configs" / "slicing.yaml"
    if not cfg_path.exists():
        sys.exit("configs/slicing.yaml yok. Önce: python scripts/horizon_band.py")
    scfg = load_slicing(cfg_path)
    band_y = (scfg["band_frac"][0] * IMG_H, scfg["band_frac"][1] * IMG_H)
    model = YOLO(a.weights)

    def teacher(img):
        full = predict_full(model, img, min(a.light_conf, a.veh_conf))
        band = predict_band(model, img, scfg, min(a.light_conf, a.veh_conf))
        xyxy, cls, conf = merge(full, band)
        return xyxy, cls, conf, peak_brightness(img, xyxy)

    def select(img_pred, gc, gb):
        xyxy, cls, conf, peak = img_pred
        return select_pseudo(xyxy, cls, conf, gc, gb, band_y, p, pred_peak=peak)

    names = {s: [n for c in CONDS for n in
                 ((a.data / "splits" / s / f"{c}.txt").read_text(encoding="utf-8").split()
                  if (a.data / "splits" / s / f"{c}.txt").exists() else [])] for s in a.splits}

    # ---------------- önizleme
    if a.preview:
        pdir = a.out / "preview"
        pdir.mkdir(parents=True, exist_ok=True)
        random.seed(3)
        colors = {"A": (255, 0, 255), "B": (0, 255, 255), "C": (0, 128, 255)}
        for n in random.sample(names["train"], min(a.preview, len(names["train"]))):
            img = cv2.imread(str(a.data / "images" / "train" / n))
            gc, gb = read_labels(a.data / "labels" / "train" / f"{Path(n).stem}.txt")
            add = select(teacher(img), gc, gb)
            vis = img.copy()
            for b in gb[np.isin(gc, LIGHTS)]:                    # mevcut ışık etiketleri
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 200, 0), 1)
            for b in gb[np.isin(gc, VEH)]:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (160, 160, 160), 1)
            for c, b, rule in add:
                cv2.rectangle(vis, (int(b[0]) - 3, int(b[1]) - 3), (int(b[2]) + 3, int(b[3]) + 3), colors[rule], 1)
            cv2.line(vis, (0, int(band_y[0])), (IMG_W, int(band_y[0])), (90, 90, 90), 1)
            cv2.line(vis, (0, int(band_y[1])), (IMG_W, int(band_y[1])), (90, 90, 90), 1)
            cv2.imwrite(str(pdir / n), vis)
            print(f"  {n}: +{len(add)}  {dict(Counter(r for _, _, r in add))}")
        print(f"Önizleme: {pdir}\n  gri=BDD araç, ince yeşil=mevcut ışık etiketi, sarı=B (kutusuz ışık çifti), "
              f"turuncu=C (etiketsiz uzak araç), mor=A (sadece --rule-a ile), gri çizgiler=ufuk bandı")
        return

    # ---------------- tam çalıştırma
    print("Etiketler kopyalanıyor...")
    for s in a.splits:
        dst = a.out / "labels" / s
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(a.data / "labels" / s, dst, ignore=shutil.ignore_patterns("*.cache"))
    if (a.out / "splits").exists():
        shutil.rmtree(a.out / "splits")
    shutil.copytree(a.data / "splits", a.out / "splits")

    stats = Counter()
    for s in a.splits:
        t0 = time.time()
        for i, n in enumerate(names[s], 1):
            img = cv2.imread(str(a.data / "images" / s / n))
            if img is None:
                continue
            lbl = a.out / "labels" / s / f"{Path(n).stem}.txt"
            gc, gb = read_labels(lbl)
            add = select(teacher(img), gc, gb)
            if add:
                lines = lbl.read_text(encoding="utf-8").splitlines() if lbl.exists() else []
                lines += [to_yolo_line(c, b) for c, b, _ in add]
                lbl.write_text("\n".join(lines), encoding="utf-8")
            for c, _, rule in add:
                stats[f"{s}/{rule}"] += 1
                stats[f"{s}/cls{c}"] += 1
            stats[f"{s}/frames"] += 1
            stats[f"{s}/frames_changed"] += bool(add)
            if i % 250 == 0 or i == len(names[s]):
                eta = (time.time() - t0) / i * (len(names[s]) - i)
                print(f"\r  [{s}] {i}/{len(names[s])}  kalan ~{eta / 60:.0f} dk   ", end="", flush=True)
        print()

    meta = {"params": {k: list(v) if isinstance(v, tuple) else v for k, v in p.items()},
            "teacher": a.weights, "band_y": band_y, "stats": dict(stats)}
    (a.out / "pseudo_label_stats.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    for s in a.splits:
        fr = max(stats[f"{s}/frames"], 1)
        print(f"[{s}] {stats[f'{s}/frames']} kare, {stats[f'{s}/frames_changed']} karede ekleme | "
              f"A (kutu içi ışık) {stats[f'{s}/A']}, B (kutusuz ışık çifti) {stats[f'{s}/B']}, "
              f"C (uzak araç) {stats[f'{s}/C']} | kare başına {(stats[f'{s}/A'] + stats[f'{s}/B'] + stats[f'{s}/C']) / fr:.2f}")
    print(f"Yazıldı: {a.out}/labels, splits, pseudo_label_stats.json")


if __name__ == "__main__":
    main()