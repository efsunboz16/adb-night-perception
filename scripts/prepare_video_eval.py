"""Faz 4 video değerlendirme listesini hazırlar (MOT 2020 Images kareleri + elimizdeki 100K etiketleri).

Fikir: BDD100K'nın her etiketli görüntüsü kendi videosunun 10. saniyesindeki karedir.
MOT 2020 Images ise aynı videoların 5 fps'lik kareleridir (<video>/<video>-0000001.jpg ...).
Bir videonun adı 100K test setimizde (BDD val) varsa, o videonun bir karesinin etiketi elimizde.
Script bu kareyi görüntü benzerliğiyle bulur (10. saniye civarı, ~51. kare).

Sızıntı: video adı eğitim listemizdeyse (BDD train), model o sahneyi gördü -> ayrı işaretlenir,
varsayılan olarak değerlendirmeye alınmaz.

Kullanım:
  python scripts/prepare_video_eval.py --frames <MOT görüntülerinin açıldığı klasör>
  (klasörde <video>/ alt klasörleri olmalı; train/val/test alt klasörleri de olabilir)
Çıktı: data/bdd_video/videos.json (+ ekranda özet)
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def video_dirs(base: Path):
    """İçinde <ad>-<sayı>.jpg kareleri olan tüm klasörler."""
    out = {}
    for d in [base, *base.rglob("*")]:
        if not d.is_dir():
            continue
        f = next(d.glob(f"{d.name}-*.jpg"), None)
        if f is not None:
            out[d.name] = d
    return out


def load_split_index(data: Path):
    """görüntü adı (uzantısız) -> (bölüm, koşul)"""
    idx = {}
    for sp in ("train", "val", "test"):
        for f in sorted((data / "splits" / sp).glob("*.txt")):
            for n in f.read_text(encoding="utf-8").split():
                idx[Path(n).stem] = (sp, f.stem)
    return idx


def small(path):
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if g is None else cv2.resize(g, (96, 54), interpolation=cv2.INTER_AREA).astype(np.float32)


def find_keyframe(frames, ref, center=51, half=15):
    """Referans görüntüye en benzeyen kare (ortalama mutlak fark, 0-255)."""
    best = (1e9, None)
    lo, hi = max(0, center - half), min(len(frames), center + half)
    for i in range(lo, hi):
        s = small(frames[i])
        if s is not None:
            d = float(np.mean(np.abs(s - ref)))
            best = min(best, (d, i))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True, help="MOT 2020 Images'ın açıldığı klasör")
    ap.add_argument("--data", type=Path, default=ROOT / "data/bdd_adb")
    ap.add_argument("--out", type=Path, default=ROOT / "data/bdd_video/videos.json")
    ap.add_argument("--max-diff", type=float, default=12.0, help="eşleşme kabul eşiği (ortalama gri fark)")
    ap.add_argument("--keep-leaks", action="store_true")
    a = ap.parse_args()

    vids = video_dirs(a.frames)
    print(f"Video klasörü: {len(vids)}  ({a.frames})")
    if not vids:
        raise SystemExit("Kare klasörü bulunamadı. --frames, içinde <video>/<video>-0000001.jpg olan klasör olmalı.")
    idx = load_split_index(a.data)
    print(f"Etiketli görüntü listesi: {len(idx)}")

    cats = Counter()
    for v in vids:
        sp = idx.get(v, (None,))[0]
        cats[{"test": "test (kullanılabilir)", "train": "train (sızıntı)", "val": "train (sızıntı)",
              None: "etiket yok"}[sp]] += 1
    print("Video adı eşleşmesi:", dict(cats))

    out, bad, conds = {}, 0, Counter()
    use = [v for v in vids if v in idx and (idx[v][0] == "test" or a.keep_leaks)]
    for k, v in enumerate(sorted(use), 1):
        sp, cond = idx[v]
        img = a.data / "images" / sp / f"{v}.jpg"
        ref = small(img)
        if ref is None:
            continue
        frames = sorted(vids[v].glob(f"{v}-*.jpg"))
        diff, i = find_keyframe(frames, ref)
        if i is None or diff > a.max_diff:
            bad += 1
            continue
        out[v] = {"split": sp, "cond": cond, "frames_dir": str(vids[v]), "n_frames": len(frames),
                  "key_index": i, "key_frame": frames[i].name, "match_diff": round(diff, 2),
                  "label": str(a.data / "labels" / sp / f"{v}.txt")}
        conds[cond] += 1
        if k % 50 == 0:
            print(f"  {k}/{len(use)}", flush=True)

    print(f"\nEtiketli karesi bulunan video: {len(out)} | eşleşmeyen: {bad}")
    print("Koşul dağılımı:", dict(conds))
    if out:
        ks = [o["key_index"] for o in out.values()]
        ds = [o["match_diff"] for o in out.values()]
        print(f"Etiketli kare indeksi: ortanca {int(np.median(ks))} (min {min(ks)}, max {max(ks)}) | "
              f"benzerlik farkı ortanca {np.median(ds):.1f}")
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"Yazıldı: {a.out}")


if __name__ == "__main__":
    main()