"""Araç kutularının genişlik dağılımı ve yaklaşık mesafe karşılığı (koşul bazında).

Soru: BDD100K gece etiketleri ne kadar uzaktaki araçları kapsıyor?
Mesafe ≈ fx * 1,8 m / kutu genişliği. fx configs/system.yaml'dan gelir; BDD kameraları
farklı olduğu için mesafeler yaklaşıktır, karşılaştırma için kullan.

Kullanım:
  python scripts/box_size_stats.py --data data/bdd_adb --split train
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adb.geometry import Camera, load_config  # noqa: E402

BINS = [(0, 8), (8, 12), (12, 16), (16, 24), (24, 32), (32, 64), (64, 10_000)]
CAR_W_M = 1.8
VEHICLES = {0, 1, 2}  # car, truck, bus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/bdd_adb"))
    ap.add_argument("--split", default="train")
    ap.add_argument("--conds", default="day,night")
    args = ap.parse_args()
    fx = Camera.from_config(load_config()).fx_px

    print(f"Araç kutusu genişlikleri ({args.split}), fx = {fx:.0f} px, araç genişliği {CAR_W_M} m\n")
    header = "| Genişlik (px) | ~Mesafe (m) | " + " | ".join(args.conds.split(",")) + " |"
    print(header + "\n" + "|---" * (2 + len(args.conds.split(","))) + "|")

    counts = {}
    for cond in args.conds.split(","):
        names = (args.data / "splits" / args.split / f"{cond}.txt").read_text(encoding="utf-8").split()
        c = [0] * len(BINS)
        for n in names:
            lbl = args.data / "labels" / args.split / f"{Path(n).stem}.txt"
            if not lbl.exists():
                continue
            for line in lbl.read_text(encoding="utf-8").splitlines():
                cls, _, _, w, _ = line.split()
                if int(cls) not in VEHICLES:
                    continue
                wpx = float(w) * 1280
                for i, (lo, hi) in enumerate(BINS):
                    if lo <= wpx < hi:
                        c[i] += 1
                        break
        counts[cond] = (c, len(names))

    for i, (lo, hi) in enumerate(BINS):
        far = fx * CAR_W_M / max(lo, 1)
        near = fx * CAR_W_M / hi
        dist = f"> {near:.0f}" if lo == 0 else (f"< {far:.0f}" if hi > 1000 else f"{near:.0f}–{far:.0f}")
        label = f"< {hi}" if lo == 0 else (f"≥ {lo}" if hi > 1000 else f"{lo}–{hi}")
        cells = []
        for cond in args.conds.split(","):
            c, n_img = counts[cond]
            tot = sum(c) or 1
            cells.append(f"{c[i]} (%{100 * c[i] / tot:.1f}) · {c[i] / max(n_img, 1):.2f}/kare")
        print(f"| {label} | {dist} | " + " | ".join(cells) + " |")
    print("\nNot: < 4 px kutular hazırlık aşamasında atıldı (MIN_BOX_PX).")


if __name__ == "__main__":
    main()