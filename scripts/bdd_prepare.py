"""
BDD100K -> YOLO hazırlık scripti (ADB projesi, Faz 1 / 2. hafta)

Yaptıkları:
  1. BDD100K tespit etiketlerini okur. İki format desteklenir:
       a) det_20: tek dosya  labels/det_20/det_train.json, det_val.json
       b) 2018 sürümü: görsel başına bir JSON  labels/.../train/*.json, labels/.../val/*.json
     Etiket yeri otomatik bulunur; gerekirse --train-labels / --val-labels ile elle ver.
  2. Sadece ADB için gereken sınıfları tutar
  3. Her görüntüyü timeofday + weather koşuluna göre etiketler
  4. YOLO formatında etiket dosyaları yazar
  5. Bölümler: BDD train -> train (%90) + val (%10), resmi BDD val -> test
     (BDD test klasörünün etiketi yayınlanmadığı için kullanılmaz)
  6. Koşul bazlı istatistik raporu (stats.json) çıkarır

Kullanım:
  python scripts/bdd_prepare.py --bdd data/raw/bdd100k --out data/bdd_adb --check     # sadece etiketleri okuyup özetle
  python scripts/bdd_prepare.py --bdd data/raw/bdd100k --out data/bdd_adb --copy-images
"""

import argparse
import json
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

# ADB için tutulan sınıflar -> YOLO sınıf id'si (configs/classes.yaml ile aynı)
CLASS_MAP = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "motorcycle": 3,
    "bicycle": 4,
    "person": 5,
}
# 2018 formatındaki farklı isimler
ALIASES = {"motor": "motorcycle", "bike": "bicycle", "rider": "person", "pedestrian": "person"}
CLASS_NAMES = [name for name, _ in sorted(CLASS_MAP.items(), key=lambda kv: kv[1])]

IMG_W, IMG_H = 1280, 720  # BDD100K görüntü boyutu
MIN_BOX_PX = 4            # bundan küçük kutular gürültü sayılır


# ---------------------------------------------------------------- etiket okuma

def read_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def normalize(frame: dict) -> dict:
    """Her iki formatı da {name, attributes, labels:[{category, box2d}]} şekline getirir."""
    name = frame["name"]
    if not Path(name).suffix:
        name += ".jpg"
    if "labels" in frame:                        # det_20
        labels = frame["labels"] or []
    else:                                        # 2018: frames[0].objects
        labels = []
        for f in frame.get("frames") or []:
            labels += f.get("objects") or []
    return {"name": Path(name).name, "attributes": frame.get("attributes") or {}, "labels": labels}


def find_labels(bdd: Path, split: str) -> Path | None:
    """Etiket dosyasını/klasörünü bulur. split: 'train' veya 'val'."""
    lbl = bdd / "labels"
    for cand in (lbl / "det_20" / f"det_{split}.json",
                 lbl / f"bdd100k_labels_images_{split}.json"):
        if cand.is_file():
            return cand
    # görsel başına JSON: adı split olan ve içinde .json bulunan klasör
    for d in sorted(lbl.rglob(split)):
        if d.is_dir() and next(d.glob("*.json"), None):
            return d
    return None


def load_frames(src: Path) -> list[dict]:
    if src.is_dir():
        files = sorted(src.glob("*.json"))
        print(f"  {src} -> {len(files)} JSON dosyası (görsel başına) okunuyor...")
        frames = []
        for p in files:
            data = read_json(p)
            frames += data if isinstance(data, list) else [data]
    else:
        print(f"  {src} okunuyor...")
        data = read_json(src)
        frames = data if isinstance(data, list) else [data]
    return [normalize(f) for f in frames]


# ---------------------------------------------------------------- dönüşüm

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


def class_of(category: str | None):
    if category is None:
        return None
    return CLASS_MAP.get(ALIASES.get(category, category))


def place_image(src: Path, dst: Path) -> str:
    """Görüntüyü hedefe koyar. Önce sabit bağlantı (hard link) dener: anında biter ve ek disk
    kullanmaz (aynı sürücüde olmalı). Olmazsa kopyalar. Hedef zaten varsa atlar (kaldığı yerden devam)."""
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return "skipped"
    if dst.exists():
        dst.unlink()                      # yarım kalmış kopya
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        tmp = dst.with_suffix(dst.suffix + ".part")
        shutil.copyfile(src, tmp)
        tmp.replace(dst)                  # kesintide yarım dosya bırakmaz
        return "copied"


def process(frames, split, img_dir: Path, out: Path, copy_images: bool, write: bool, stats, index):
    lbl_dir = out / "labels" / split
    im_dir = out / "images" / split
    if write:
        lbl_dir.mkdir(parents=True, exist_ok=True)
        im_dir.mkdir(parents=True, exist_ok=True)

    n = len(frames)
    t0 = time.time()
    for i, fr in enumerate(frames, 1):
        if write and (i % 1000 == 0 or i == n):
            el = time.time() - t0
            eta = el / i * (n - i)
            print(f"\r  [{split}] {i}/{n}  ({100 * i / n:.0f}%)  kalan ~{eta / 60:.1f} dk   ", end="", flush=True)
            if i == n:
                print()
        cond = condition_of(fr["attributes"])
        lines = []
        for lab in fr["labels"]:
            cat = lab.get("category")
            cls = class_of(cat)
            if cls is None or "box2d" not in lab:
                stats[split]["skipped_categories"][str(cat)] += 1
                continue
            yb = to_yolo(lab["box2d"])
            if yb is None:
                stats[split]["skipped_categories"]["(çok küçük kutu)"] += 1
                continue
            cx, cy, w, h, area = yb
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            stats[split]["boxes_by_class"][CLASS_NAMES[cls]] += 1
            stats[split]["boxes_by_size"][size_bucket(area)] += 1

        src = img_dir / fr["name"]
        if not src.exists():
            stats[split]["missing_images"]["count"] += 1
        if write:
            (lbl_dir / f"{Path(fr['name']).stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            if copy_images and src.exists():
                stats[split]["image_placement"][place_image(src, im_dir / fr["name"])] += 1

        stats[split]["images_by_condition"][cond] += 1
        index[split][cond].append(fr["name"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--train-labels", type=Path, help="det_train.json veya görsel başına JSON klasörü")
    ap.add_argument("--val-labels", type=Path, help="det_val.json veya görsel başına JSON klasörü")
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy-images", action="store_true")
    ap.add_argument("--check", action="store_true", help="hiçbir şey yazma, sadece özetle")
    args = ap.parse_args()

    tr_src = args.train_labels or find_labels(args.bdd, "train")
    te_src = args.val_labels or find_labels(args.bdd, "val")
    if not tr_src or not te_src:
        raise SystemExit(
            f"Etiketler bulunamadı (train: {tr_src}, val: {te_src}).\n"
            f"{args.bdd / 'labels'} altındaki klasörleri kontrol et veya --train-labels / --val-labels ver.")

    print("Etiketler:")
    train_frames = load_frames(tr_src)
    test_frames = load_frames(te_src)

    # Train'den val ayır: koşul oranları korunsun diye koşul bazında (stratified) böl
    random.seed(args.seed)
    by_cond = defaultdict(list)
    for fr in train_frames:
        by_cond[condition_of(fr["attributes"])].append(fr)
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
        "skipped_categories": Counter(),
        "missing_images": Counter(),
        "image_placement": Counter(),
    })
    index = defaultdict(lambda: defaultdict(list))

    write = not args.check
    img_root = args.bdd / "images" / "100k"
    process(tr, "train", img_root / "train", args.out, args.copy_images, write, stats, index)
    process(va, "val", img_root / "train", args.out, args.copy_images, write, stats, index)
    process(test_frames, "test", img_root / "val", args.out, args.copy_images, write, stats, index)

    if write:
        for split, conds in index.items():
            for cond, names in conds.items():
                p = args.out / "splits" / split / f"{cond}.txt"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("\n".join(names), encoding="utf-8")

        yaml = [
            f"path: {args.out.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
        ] + [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
        (args.out / "bdd_adb.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

        report = {s: {k: dict(v) for k, v in d.items()} for s, d in stats.items()}
        (args.out / "stats.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    for split in ("train", "val", "test"):
        s = stats[split]
        c = s["images_by_condition"]
        total = sum(c.values())
        night = c["night"] + c["night_adverse"]
        print(f"[{split}] {total} görüntü | gece: {night} (%{100 * night / max(total, 1):.1f}) | "
              f"koşullar: {dict(c)}")
        print(f"         kutular: {dict(s['boxes_by_class'])}")
        print(f"         kutu boyutu: {dict(s['boxes_by_size'])}")
        if s["image_placement"]:
            print(f"         görüntüler: {dict(s['image_placement'])}  (linked=bağlantı, copied=kopya, skipped=zaten vardı)")
        if s["missing_images"]["count"]:
            print(f"         UYARI: {s['missing_images']['count']} görüntü dosyası bulunamadı")

    if sum(stats["train"]["boxes_by_class"].values()) == 0:
        print("\nUYARI: Hiç kutu okunamadı. Atlanan kategoriler:",
              dict(stats["train"]["skipped_categories"].most_common(10)))
    print("\n" + ("(--check: hiçbir dosya yazılmadı)" if args.check
                  else f"Yazıldı: {args.out}/bdd_adb.yaml, stats.json, splits/"))


if __name__ == "__main__":
    main()
