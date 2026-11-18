"""Kendi far/lamba etiketlerin için gece karelerini seçer (Faz 1 / p1h).

Seçim kuralları (gerekçe: docs/labeling_guide.md):
  - Sadece bizim TRAIN bölümünden, temiz gece ('night') kareleri. Val ve test'e dokunulmaz.
  - Sahne kotası: otoyol %50, yerleşim %25, şehir caddesi %25 (yetmezse diğerlerinden tamamlanır).
  - Karelerin %80'i genişliği < 16 px olan en az bir araç içerir (~125 m ötesi; BDD gece
    etiketlerinin en eksik olduğu bölge, bkz. ADR-007); %20'si rastgele
    (boş yollar: sokak lambası ve reflektör negatif örnekleri için).
  - İçlerinden 100 kare çift etiketleme (etiket uyumu ölçümü) için işaretlenir.

Kullanım:
  python scripts/select_label_frames.py --bdd data/raw/bdd100k --data data/bdd_adb --out data/own_lights
Çıktı:
  data/own_lights/images/*.jpg      (hard link)
  data/own_lights/selection.csv     (name, scene, small_vehicles, vehicles, double_label)
"""

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

SCENE_QUOTA = {"highway": 0.50, "residential": 0.25, "city street": 0.25}
FAR_W_PX = 16          # bu genişliğin altındaki araç ~125 m'den uzak
IMG_W, IMG_H = 1280, 720


def scene_of(raw_labels: Path, name: str) -> str:
    p = raw_labels / f"{Path(name).stem}.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("attributes", {}).get("scene", "undefined")
    except FileNotFoundError:
        return "undefined"


def box_stats(lbl: Path) -> tuple[int, int]:
    if not lbl.exists():
        return 0, 0
    small = total = 0
    for line in lbl.read_text(encoding="utf-8").splitlines():
        c, _, _, w, h = line.split()
        if int(c) <= 4:  # araçlar (person hariç)
            total += 1
            small += float(w) * IMG_W < FAR_W_PX
    return small, total


def link(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd", type=Path, default=Path("data/raw/bdd100k"))
    ap.add_argument("--data", type=Path, default=Path("data/bdd_adb"))
    ap.add_argument("--out", type=Path, default=Path("data/own_lights"))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-double", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    raw_labels = args.bdd / "labels" / "det_20" / "train"
    names = (args.data / "splits" / "train" / "night.txt").read_text(encoding="utf-8").split()
    print(f"{len(names)} temiz gece karesi taranıyor...")

    pool = []
    for n in names:
        small, total = box_stats(args.data / "labels" / "train" / f"{Path(n).stem}.txt")
        pool.append({"name": n, "scene": scene_of(raw_labels, n), "small_vehicles": small, "vehicles": total})
    print("Sahne dağılımı:", dict(Counter(r["scene"] for r in pool).most_common()))

    chosen, used = [], set()
    for scene, frac in SCENE_QUOTA.items():
        want = round(args.n * frac)
        cands = [r for r in pool if r["scene"] == scene]
        with_small = [r for r in cands if r["small_vehicles"] > 0]
        random.shuffle(with_small)
        random.shuffle(cands)
        take = with_small[:round(want * 0.8)]
        ids = {r["name"] for r in take}
        take += [r for r in cands if r["name"] not in ids][:want - len(take)]
        chosen += take
        used |= {r["name"] for r in take}
    rest = [r for r in pool if r["name"] not in used]
    random.shuffle(rest)
    chosen += rest[:args.n - len(chosen)]           # kota dolmadıysa tamamla

    for r in chosen:
        r["double_label"] = 0
    for r in random.sample(chosen, min(args.n_double, len(chosen))):
        r["double_label"] = 1

    img_out = args.out / "images"
    img_out.mkdir(parents=True, exist_ok=True)
    for r in chosen:
        link(args.data / "images" / "train" / r["name"], img_out / r["name"])
    with (args.out / "selection.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["name", "scene", "small_vehicles", "vehicles", "double_label"])
        w.writeheader()
        w.writerows(sorted(chosen, key=lambda r: r["name"]))

    print(f"\nSeçilen: {len(chosen)} kare -> {img_out}")
    print("  sahne:", dict(Counter(r["scene"] for r in chosen).most_common()))
    print("  uzak araç (< 16 px) içeren:", sum(r["small_vehicles"] > 0 for r in chosen))
    print("  çift etiketlenecek:", sum(r["double_label"] for r in chosen))


if __name__ == "__main__":
    main()
