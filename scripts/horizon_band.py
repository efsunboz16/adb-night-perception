"""Ufuk bandını veriden belirler ve configs/slicing.yaml dosyasına yazar.

Gece eğitim karelerinde uzak araç kutularının (genişlik < --far-px) dikey merkezlerine bakılır;
bant, bu merkezlerin %2–%98 aralığını + kutu yüksekliği payını kapsar. Bant yüksekliği,
büyütme sonrası 32'nin katı olacak şekilde yuvarlanır (YOLO adımı).

Kullanım:
  python scripts/horizon_band.py
"""

import argparse
import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
VEH = {0, 1, 2, 3}
IMG_W, IMG_H = 1280, 720


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/bdd_adb")
    ap.add_argument("--conds", nargs="+", default=["night", "night_adverse"])
    ap.add_argument("--far-px", type=float, default=16, help="bu genişliğin altındaki araç 'uzak' (~125 m+)")
    ap.add_argument("--tile-w", type=int, default=384)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--min-height", type=float, default=160,
                    help="en az bant yüksekliği (px); çok ince şerit modele bağlam bırakmaz")
    ap.add_argument("--out", type=Path, default=ROOT / "configs/slicing.yaml")
    a = ap.parse_args()

    ys, hs = [], []
    for cond in a.conds:
        f = a.data / "splits" / "train" / f"{cond}.txt"
        for n in f.read_text(encoding="utf-8").split():
            lbl = a.data / "labels" / "train" / f"{Path(n).stem}.txt"
            if not lbl.exists():
                continue
            for ln in lbl.read_text(encoding="utf-8").splitlines():
                c, _, cy, w, h = ln.split()
                if int(c) in VEH and float(w) * IMG_W < a.far_px:
                    ys.append(float(cy) * IMG_H)
                    hs.append(float(h) * IMG_H)
    ys, hs = np.array(ys), np.array(hs)
    if len(ys) < 100:
        raise SystemExit(f"Yeterli uzak araç bulunamadı ({len(ys)}).")
    lo, hi = np.percentile(ys, [2, 98])
    pad = float(np.percentile(hs, 95))
    y0, y1 = max(0.0, lo - pad), min(IMG_H, hi + pad)
    # büyütülmüş yükseklik 32'nin katı olsun
    step = 32 / a.scale
    height = math.ceil(max(y1 - y0, a.min_height) / step) * step
    c = (y0 + y1) / 2
    y0, y1 = max(0.0, c - height / 2), min(float(IMG_H), c + height / 2)
    inside = np.mean((ys >= y0) & (ys <= y1))

    cfg = {
        "band_frac": [round(float(y0) / IMG_H, 4), round(float(y1) / IMG_H, 4)],
        "tile_w": a.tile_w, "overlap": a.overlap, "scale": a.scale,
        "source": f"scripts/horizon_band.py, {len(ys)} uzak araç (genişlik < {a.far_px} px, {', '.join(a.conds)})",
    }
    a.out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    n_tiles = len(range(0, IMG_W - a.tile_w + 1, a.tile_w - a.overlap)) + 1
    print(f"Uzak araç sayısı: {len(ys)} | dikey merkez medyanı: {np.median(ys):.0f} px")
    print(f"Bant: y = {y0:.0f}–{y1:.0f} px (yükseklik {y1 - y0:.0f} px), uzak araçların %{100 * inside:.1f}'i içinde")
    print(f"Parçalar: ~{n_tiles} × {a.tile_w}×{y1 - y0:.0f} px, {a.scale}x büyütme -> "
          f"{int(a.tile_w * a.scale)}×{int((y1 - y0) * a.scale)} px")
    print(f"Yazıldı: {a.out}")


if __name__ == "__main__":
    main()