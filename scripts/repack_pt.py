"""Yanlışlıkla klasöre açılmış bir PyTorch .pt dosyasını tekrar tek dosyaya paketler.

.pt dosyaları aslında zip arşividir; indirme/ayıklama araçları bazen onları klasöre açar
(içinde data.pkl, data/, version, byteorder ... olur).

Kullanım:
  python scripts/repack_pt.py runs/baseline/weights/best
  -> runs/baseline/weights/best.pt
"""

import sys
import zipfile
from pathlib import Path


def repack(folder: Path) -> Path:
    if not (folder / "data.pkl").exists():
        # bazı araçlar bir kat daha klasör ekler: best/best/data.pkl
        inner = [d for d in folder.iterdir() if d.is_dir() and (d / "data.pkl").exists()]
        if not inner:
            sys.exit(f"{folder} içinde data.pkl yok; bu bir açılmış .pt klasörü değil.")
        folder = inner[0]
    out = folder.parent / f"{folder.name}.pt"
    if folder.parent.name == folder.name:          # best/best -> weights/best.pt
        out = folder.parent.parent / f"{folder.name}.pt"
    root = folder.name
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    # data.pkl ve sürüm dosyaları önce, tensör verileri sonra (torch'un yazdığı sıraya yakın)
    files.sort(key=lambda p: (0 if p.parent == folder else 1, p.as_posix()))
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as z:
        for p in files:
            z.write(p, f"{root}/{p.relative_to(folder).as_posix()}")
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = repack(Path(sys.argv[1]))
    print(f"Yazıldı: {out} ({out.stat().st_size / 2**20:.1f} MB)")
    try:
        from ultralytics import YOLO
        m = YOLO(str(out))
        print(f"Doğrulandı: model yüklendi, {len(m.names)} sınıf: {list(m.names.values())}")
    except Exception as e:
        print(f"UYARI: dosya yüklenemedi ({type(e).__name__}: {e}). Kaggle'dan tek dosya olarak yeniden indir.")


if __name__ == "__main__":
    main()
