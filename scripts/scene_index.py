"""BDD100K sahne türlerini (highway, city street, residential, ...) bir dizine yazar.

bdd_prepare.py sahne bilgisini saklamıyordu. ADB'nin asıl kullanıldığı yer otoyol ve kırsal yol;
şehir içinde segmentlerin çoğu zaten kısık olduğu için sistem metrikleri ayırt edici değil.
Bu dizinle değerlendirmeler sahneye göre filtrelenebilir (eval_video.py --scenes highway).

Kullanım:
  python scripts/scene_index.py --bdd data/raw/bdd100k
Çıktı: data/bdd_adb/scenes.json  {görüntü_adı: sahne}
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd", type=Path, default=ROOT / "data/raw/bdd100k")
    ap.add_argument("--data", type=Path, default=ROOT / "data/bdd_adb")
    ap.add_argument("--splits", nargs="+", default=["test"], help="hangi bölümlerin görüntüleri (test = BDD val)")
    a = ap.parse_args()

    want = set()
    for sp in a.splits:
        for f in (a.data / "splits" / sp).glob("*.txt"):
            want |= {Path(n).stem for n in f.read_text(encoding="utf-8").split()}
    print(f"Aranan görüntü: {len(want)}")

    scenes = {}
    for p in a.bdd.rglob("*.json"):
        if p.stem not in want or p.stem in scenes:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        att = d.get("attributes") if isinstance(d, dict) else None
        if att and "scene" in att:
            scenes[p.stem] = att["scene"]
    if not scenes:
        raise SystemExit(f"{a.bdd} altında görüntü başına JSON etiketi (attributes.scene) bulunamadı.")
    out = a.data / "scenes.json"
    out.write_text(json.dumps(scenes), encoding="utf-8")
    print(f"Bulunan: {len(scenes)} / {len(want)}")
    print("Sahne dağılımı:", dict(Counter(scenes.values()).most_common()))
    print(f"Yazıldı: {out}")


if __name__ == "__main__":
    main()