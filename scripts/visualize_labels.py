"""YOLO etiketlerini görüntülerin üzerine çizip ızgara halinde kaydeder (gözle kontrol).

Kullanım:
  python scripts/visualize_labels.py --data data/bdd_adb --split train --cond night --n 12
Çıktı: data/bdd_adb/preview_<split>_<cond>.jpg
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
COLORS = [(0, 200, 255), (255, 160, 0), (255, 0, 200), (0, 255, 0), (255, 255, 0), (0, 0, 255)]


def draw(img_path: Path, lbl_path: Path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if lbl_path.exists():
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            c, cx, cy, bw, bh = line.split()
            c = int(c)
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            p1 = (int(cx - bw / 2), int(cy - bh / 2))
            p2 = (int(cx + bw / 2), int(cy + bh / 2))
            small = bw * bh < 32 * 32
            cv2.rectangle(img, p1, p2, COLORS[c % len(COLORS)], 1 if small else 2)
            if not small:
                cv2.putText(img, NAMES[c], (p1[0], max(12, p1[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, COLORS[c % len(COLORS)], 1, cv2.LINE_AA)
    cv2.putText(img, img_path.stem, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.resize(img, (640, 360))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/bdd_adb"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--cond", default="night")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = (args.data / "splits" / args.split / f"{args.cond}.txt").read_text(encoding="utf-8").split()
    random.seed(args.seed)
    picks = random.sample(names, min(args.n, len(names)))
    tiles = [t for t in (draw(args.data / "images" / args.split / n,
                              args.data / "labels" / args.split / f"{Path(n).stem}.txt") for n in picks)
             if t is not None]
    if not tiles:
        raise SystemExit("Görüntü okunamadı.")
    cols = 3
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    out = args.data / f"preview_{args.split}_{args.cond}.jpg"
    cv2.imwrite(str(out), grid)
    print(f"Kaydedildi: {out}  ({len(picks)} görüntü)")


if __name__ == "__main__":
    main()
