"""BDD araç kutularına dayanarak far/lamba etiketlerini otomatik üretir (ADR-008).

Mantık:
  - Her gece/alacakaranlık karesinde parlak lekeler bulunur (prelabel_lights.detect ile aynı yöntem).
  - Leke, BDD'de İNSANIN çizdiği bir araç kutusunun içinde ve kutunun alt %75'indeyse
    -> o aracın ışığıdır. Çevresi kırmızıysa tail_light, değilse head_light.
    (Üst kısım hariç: ön camdaki yansımalar ışık sayılmasın.)
  - Kutu dışında ve ufkun üstündeki lekeler -> street_light.
  - Kutu dışında ve ufkun altındaki lekeler etiketlenmez: etiketsiz uzak araç, reflektör ya da
    sokak lambası olabilir; otomatik karar vermek güvenli değil.
  - Test bölümüne dokunulmaz: ışık sınıfları test'te insan kontrollü küçük bir setle ölçülür.

Çıktı: BDD araç etiketleri + ışık etiketleri birlikte, yeni bir veri seti klasöründe
(görüntüler hard link). Sınıf id'leri configs/classes.yaml ile aynı.

Kullanım:
  python scripts/autolabel_lights.py --preview 12      # önce gözle kontrol
  python scripts/autolabel_lights.py                    # tüm train/val
"""

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prelabel_lights import detect  # noqa: E402

VEHICLE_CLS = {0, 1, 2, 3}          # car, truck, bus, motorcycle
HEAD, TAIL, STREET = 6, 7, 8
NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person",
         "head_light", "tail_light", "street_light", "reflector"]
LIGHT_CONDS = ("night", "night_adverse", "dusk", "dusk_adverse")
DRAW = {HEAD: (0, 212, 255), TAIL: (45, 45, 255), STREET: (255, 156, 45)}


def read_boxes(lbl: Path, w: int, h: int):
    boxes, lines = [], []
    if lbl.exists():
        for line in lbl.read_text(encoding="utf-8").splitlines():
            c, cx, cy, bw, bh = line.split()
            lines.append(line)
            if int(c) in VEHICLE_CLS:
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
    return boxes, lines


def light_box(c, w, h, min_px=4, pad=2):
    x, y, bw, bh = c["bbox"]
    x0, y0, x1, y1 = x - pad, y - pad, x + bw + pad, y + bh + pad
    if x1 - x0 < min_px:
        x0, x1 = c["x"] - min_px / 2, c["x"] + min_px / 2
    if y1 - y0 < min_px:
        y0, y1 = c["y"] - min_px / 2, c["y"] + min_px / 2
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    return ((x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h)


def label_frame(img, boxes, a):
    h, w = img.shape[:2]
    cands = detect(img, a)
    out, per_box = [], Counter()
    for c in sorted(cands, key=lambda d: -d["peak"]):
        host = None
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            top = y0 + 0.25 * (y1 - y0)
            if x0 - 2 <= c["x"] <= x1 + 2 and top <= c["y"] <= y1 + 2:
                host = i
                break
        if host is not None:
            if per_box[host] >= a.max_per_vehicle:
                continue
            per_box[host] += 1
            cls = TAIL if c["cls"] == "tail_light" else HEAD
        elif a.street and c["y"] < a.horizon * h:
            cls = STREET   # varsayılan kapalı (ADR-009): kare başına ~65 gürültülü etiket üretiyordu
        else:
            continue
        out.append((cls, light_box(c, w, h), c))
    return out


def place(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/bdd_adb"))
    ap.add_argument("--out", type=Path, default=Path("data/bdd_adb_lights"))
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--max-per-vehicle", type=int, default=4)
    ap.add_argument("--street", action="store_true", help="street_light etiketlerini de üret (ADR-009: kapalı)")
    ap.add_argument("--preview", type=int, default=0)
    # dedektör eşikleri (prelabel_lights ile aynı anlamda)
    ap.add_argument("--bright", type=int, default=200)
    ap.add_argument("--contrast", type=int, default=40)
    ap.add_argument("--tophat-k", type=int, default=31)
    ap.add_argument("--red", type=float, default=25)
    ap.add_argument("--horizon", type=float, default=0.45)
    ap.add_argument("--max-area", type=int, default=3000)
    a = ap.parse_args()
    a.max_below, a.max_above = 10**6, 10**6

    if a.preview:
        pdir = a.out / "preview"
        pdir.mkdir(parents=True, exist_ok=True)
        names = (a.data / "splits" / "train" / "night.txt").read_text(encoding="utf-8").split()
        import random
        random.seed(1)
        for n in random.sample(names, min(a.preview, len(names))):
            img = cv2.imread(str(a.data / "images" / "train" / n))
            h, w = img.shape[:2]
            boxes, _ = read_boxes(a.data / "labels" / "train" / f"{Path(n).stem}.txt", w, h)
            vis = img.copy()
            for x0, y0, x1, y1 in boxes:
                cv2.rectangle(vis, (int(x0), int(y0)), (int(x1), int(y1)), (200, 200, 200), 1)
            labs = label_frame(img, boxes, a)
            for cls, (cx, cy, bw, bh), _ in labs:
                p0 = (int((cx - bw / 2) * w) - 3, int((cy - bh / 2) * h) - 3)
                p1 = (int((cx + bw / 2) * w) + 3, int((cy + bh / 2) * h) + 3)
                cv2.rectangle(vis, p0, p1, DRAW[cls], 1)
            cv2.imwrite(str(pdir / n), vis)
            cnt = Counter(NAMES[c] for c, _, _ in labs)
            print(f"  {n}: {len(boxes)} araç kutusu -> {dict(cnt)}")
        print(f"Önizlemeler: {pdir}  (gri=BDD araç, sarı=head, kırmızı=tail, mavi=street)")
        return

    stats = Counter()
    for split in a.splits.split(","):
        names = (a.data / "images" / split).iterdir()
        names = sorted(p.name for p in names)
        light_names = set()
        for cond in LIGHT_CONDS:
            f = a.data / "splits" / split / f"{cond}.txt"
            if f.exists():
                light_names |= set(f.read_text(encoding="utf-8").split())
        (a.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for i, n in enumerate(names, 1):
            stem = Path(n).stem
            src_img = a.data / "images" / split / n
            lbl = a.data / "labels" / split / f"{stem}.txt"
            place(src_img, a.out / "images" / split / n)
            lines = lbl.read_text(encoding="utf-8").splitlines() if lbl.exists() else []
            if n in light_names:
                img = cv2.imread(str(src_img))
                if img is not None:
                    h, w = img.shape[:2]
                    boxes, _ = read_boxes(lbl, w, h)
                    for cls, (cx, cy, bw, bh), _ in label_frame(img, boxes, a):
                        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                        stats[f"{split}/{NAMES[cls]}"] += 1
                    stats[f"{split}/light_frames"] += 1
            (a.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
            if i % 500 == 0 or i == len(names):
                eta = (time.time() - t0) / i * (len(names) - i)
                print(f"\r  [{split}] {i}/{len(names)}  kalan ~{eta / 60:.1f} dk   ", end="", flush=True)
        print()

    # test: sadece araç etiketleri (ışıklar insan kontrollü setle ölçülecek)
    for sub in ("splits",):
        if not (a.out / sub).exists():
            shutil.copytree(a.data / sub, a.out / sub)
    yaml = [f"path: {a.out.resolve().as_posix()}", "train: images/train", "val: images/val",
            f"test: {(a.data / 'images' / 'test').resolve().as_posix()}", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (a.out / "bdd_adb_lights.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    (a.out / "autolabel_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    for split in a.splits.split(","):
        fr = stats[f"{split}/light_frames"] or 1
        print(f"[{split}] {stats[f'{split}/light_frames']} gece/alacakaranlık karesi | "
              f"head {stats[f'{split}/head_light']} ({stats[f'{split}/head_light'] / fr:.1f}/kare), "
              f"tail {stats[f'{split}/tail_light']} ({stats[f'{split}/tail_light'] / fr:.1f}/kare), "
              f"street {stats[f'{split}/street_light']}")
    print(f"Yazıldı: {a.out}/bdd_adb_lights.yaml")


if __name__ == "__main__":
    main()
