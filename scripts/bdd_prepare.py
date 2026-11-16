"""
BDD100K -> YOLO hazırlık scripti (ADB projesi, Faz 1 / 2. hafta)

Yaptıkları:
  1. det_20 etiketlerini okur (det_train.json, det_val.json)
  2. Sadece ADB için gereken araç sınıflarını tutar
  3. Her görüntüyü timeofday + weather koşuluna göre etiketler
  4. YOLO formatında etiket dosyaları yazar
  5. Bölümler: BDD train -> train (%90) + val (%10), resmi BDD val -> test
  6. Koşul bazlı istatistik raporu (stats.json) çıkarır

Beklenen klasör yapısı (BDD100K indirildikten sonra):
  bdd100k/
    images/100k/train/*.jpg
    images/100k/val/*.jpg
    labels/det_20/det_train.json
    labels/det_20/det_val.json

Kullanım:
  python scripts/bdd_prepare.py --bdd data/raw/bdd100k --out data/bdd_adb
  python scripts/bdd_prepare.py --bdd data/raw/bdd100k --out data/bdd_adb --copy-images   # görüntüleri de kopyalar
"""

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# ADB için tutulan sınıflar -> YOLO sınıf id'si
# person isteğe bağlı: istemezsen satırı sil.
CLASS_MAP = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "motorcycle": 3,
    "bicycle": 4,
    "person": 5,
}
CLASS_NAMES = [name for name, _ in sorted(CLASS_MAP.items(), key=lambda kv: kv[1])]

IMG_W, IMG_H = 1280, 720  # BDD100K görüntü boyutu
MIN_BOX_PX = 4            # bundan küçük kutular gürültü sayılır


def condition_of(attrs: dict) -> str:
    """Görüntüyü tek bir koşul etiketine indirger: day / night / dusk + yağmur bilgisi."""
    tod = attrs.get("timeofday", "undefined")
    weather = attrs.get("weather", "undefined")
    base = {"daytime": "day", "night": "night", "dawn/dusk": "dusk"}.get(tod, "unknown")
    wet = weather in ("rainy", "snowy", "foggy")
    return f"{base}_adverse" if wet else base


def to_yolo(box: dict):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    x1, x2 = max(0.0, min(x1, x2)), min(IMG_W, max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(IMG_H, max(y1, y2))
    w, h = x2 - x1, y2 - y1
    if w < MIN_BOX_PX or h < MIN_BOX_PX:
        return None
    return ((x1 + w / 2) / IMG_W, (y1 + h / 2) / IMG_H, w / IMG_W, h / IMG_H, w * h)


def size_bucket(area_px: float) -> str:
    # COCO tarzı kovalar: küçük kutular ≈ uzak araçlar (ADB için kritik)
    if area_px < 32 * 32:
        return "small"
    if area_px < 96 * 96:
        return "medium"
    return "large"


def process(frames, split, img_dir: Path, out: Path, copy_images: bool, stats, index):
    lbl_dir = out / "labels" / split
    im_dir = out / "images" / split
    lbl_dir.mkdir(parents=True, exist_ok=True)
    im_dir.mkdir(parents=True, exist_ok=True)

    for fr in frames:
        cond = condition_of(fr.get("attributes", {}))
        lines = []
        for lab in fr.get("labels") or []:
            cls = CLASS_MAP.get(lab.get("category"))
            if cls is None or "box2d" not in lab:
                continue
            yb = to_yolo(lab["box2d"])
            if yb is None:
                continue
            cx, cy, w, h, area = yb
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats[split]["boxes_by_class"][CLASS_NAMES[cls]] += 1
            stats[split]["boxes_by_size"][size_bucket(area)] += 1

        stem = Path(fr["name"]).stem
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
        src = img_dir / fr["name"]
        if copy_images and src.exists():
            shutil.copy2(src, im_dir / fr["name"])

        stats[split]["images_by_condition"][cond] += 1
        index[split][cond].append(fr["name"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy-images", action="store_true")
    args = ap.parse_args()

    lbl = args.bdd / "labels" / "det_20"
    train_frames = json.loads((lbl / "det_train.json").read_text())
    test_frames = json.loads((lbl / "det_val.json").read_text())

    # Train'den val ayır: koşul oranları korunsun diye koşul bazında (stratified) böl
    random.seed(args.seed)
    by_cond = defaultdict(list)
    for fr in train_frames:
        by_cond[condition_of(fr.get("attributes", {}))].append(fr)
    tr, va = [], []
    for frs in by_cond.values():
        random.shuffle(frs)
        k = int(len(frs) * args.val_ratio)
        va += frs[:k]
        tr += frs[k:]

    stats = defaultdict(lambda: {
        "images_by_condition": Counter(),
        "boxes_by_class": Counter(),
        "boxes_by_size": Counter(),
    })
    index = defaultdict(lambda: defaultdict(list))

    img_root = args.bdd / "images" / "100k"
    process(tr, "train", img_root / "train", args.out, args.copy_images, stats, index)
    process(va, "val", img_root / "train", args.out, args.copy_images, stats, index)
    process(test_frames, "test", img_root / "val", args.out, args.copy_images, stats, index)

    # Koşul listeleri: ileride "sadece gece test" gibi değerlendirmeler için
    for split, conds in index.items():
        for cond, names in conds.items():
            p = args.out / "splits" / split / f"{cond}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join(names))

    # YOLO dataset yaml
    yaml = [
        f"path: {args.out.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ] + [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    (args.out / "bdd_adb.yaml").write_text("\n".join(yaml) + "\n")

    report = {s: {k: dict(v) for k, v in d.items()} for s, d in stats.items()}
    (args.out / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    for split in ("train", "val", "test"):
        c = stats[split]["images_by_condition"]
        total = sum(c.values())
        night = c["night"] + c["night_adverse"]
        print(f"[{split}] {total} görüntü | gece: {night} (%{100 * night / max(total, 1):.1f}) | "
              f"koşullar: {dict(c)}")
        print(f"         kutu boyutu: {dict(stats[split]['boxes_by_size'])}")
    print(f"\nYazıldı: {args.out}/bdd_adb.yaml, stats.json, splits/")


if __name__ == "__main__":
    main()
