"""YOLO etiket dosyalarından belirli sınıfları siler (yerinde).

ADR-009: otomatik street_light etiketleri (sınıf 8) çok gürültülü; çıkarılır,
sokak lambaları eğitimde arka plan olarak kalır.

Kullanım:
  python scripts/drop_class.py --labels data/bdd_adb_lights/labels --cls 8
"""

import argparse
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--cls", type=int, nargs="+", required=True)
    a = ap.parse_args()
    drop = {str(c) for c in a.cls}
    removed, kept, files = Counter(), Counter(), 0
    for p in a.labels.rglob("*.txt"):
        lines = p.read_text(encoding="utf-8").splitlines()
        new = []
        for ln in lines:
            c = ln.split(" ", 1)[0]
            if c in drop:
                removed[c] += 1
            else:
                new.append(ln)
                kept[c] += 1
        if len(new) != len(lines):
            p.write_text("\n".join(new), encoding="utf-8")
            files += 1
    print(f"{files} dosya güncellendi. Silinen: {dict(removed)}")
    print(f"Kalan sınıf sayıları: {dict(sorted(kept.items(), key=lambda kv: int(kv[0])))}")


if __name__ == "__main__":
    main()
