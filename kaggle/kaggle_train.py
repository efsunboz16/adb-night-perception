"""Kaggle'da eğitim (ADR-010 düzeniyle: gece ağırlıklı liste, küçük doğrulama, süre sınırı).

/kaggle/input altındaki verileri kendisi bulur:
  - görüntüler: içinde train/ ve val/ klasörlerinde .jpg olan dizin
  - etiketler : içinde train/ ve val/ klasörlerinde .txt olan dizin (data/bdd_adb_lights/labels)
  - listeler  : içinde train/night.txt bulunan dizin (data/bdd_adb_lights/splits)
Yollar elle de verilebilir (--images-root, --labels-root, --splits-root).

/kaggle/input salt okunur olduğu için /tmp/adb_ds altında görüntülere sembolik bağlantı,
etiketlere kopya oluşturulur (YOLO etiket klasörüne önbellek yazar).

Kullanım (notebook hücresinde):
  !python kaggle_train.py --check          # sadece veriyi bul ve listeyi kur
  !python kaggle_train.py                  # 11 saatlik eğitim
  !python kaggle_train.py --resume /kaggle/input/<onceki-cikti>/runs/baseline/weights/last.pt
  !python kaggle_train.py --init-weights /kaggle/input/<onceki-cikti>/runs/baseline/weights/best.pt
"""

import argparse
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person",
         "head_light", "tail_light", "street_light", "reflector"]
MIX = {"night": (("night", "night_adverse"), 0.75),
       "dusk": (("dusk", "dusk_adverse"), 0.10),
       "day": (("day", "day_adverse"), 0.15)}


# ------------------------------------------------------------------ veri bulma

def has_files(d: Path, ext: str) -> bool:
    try:
        return next((p for p in os.scandir(d) if p.name.endswith(ext)), None) is not None
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return False


def find_root(base: Path, ext: str, max_depth=6):
    """train/ ve val/ alt klasörlerinde `ext` dosyaları olan ilk dizini bulur.
    splits/ (train/night.txt içerir) etiket klasörü sanılmasın diye atlanır."""
    for dirpath, dirnames, _ in os.walk(base):
        dirnames.sort()
        d = Path(dirpath)
        if len(d.relative_to(base).parts) > max_depth:
            dirnames[:] = []
            continue
        if "train" in dirnames and "val" in dirnames and has_files(d / "train", ext) and has_files(d / "val", ext):
            if (d / "train" / "night.txt").exists():      # bu splits/, etiket değil
                continue
            return d
    return None


def find_splits(base: Path):
    for p in base.rglob("night.txt"):
        if p.parent.name == "train" and (p.parent.parent / "val").exists():
            return p.parent.parent
    return None


# ------------------------------------------------------------------ veri ağacı

def build(images_root, labels_root, splits_root, out: Path, n_train, n_val, seed=42):
    rng = random.Random(seed)

    def read(split, cond):
        f = splits_root / split / f"{cond}.txt"
        return f.read_text(encoding="utf-8").split() if f.exists() else []

    train = []
    print(f"Eğitim listesi ({n_train} hedef):")
    for g, (conds, share) in MIX.items():
        pool = [n for c in conds for n in read("train", c)]
        rng.shuffle(pool)
        take = pool[:round(n_train * share)]
        train += take
        print(f"  {g:6s}: {len(take):6d} / {len(pool)}")
    val = [n for c in ("night", "night_adverse", "dusk", "day") for n in read("val", c)]
    rng.shuffle(val)
    val = val[:n_val]

    lists = {}
    for split, names in (("train", train), ("val", val)):
        di, dl = out / "images" / split, out / "labels" / split
        di.mkdir(parents=True, exist_ok=True)
        dl.mkdir(parents=True, exist_ok=True)

        def prep(n, split=split, di=di, dl=dl):
            # Bizim val bölümümüz BDD'nin TRAIN görüntülerinden ayrıldı; BDD val klasörü bizim test'imiz.
            for src in (images_root / split / n, images_root / "train" / n):
                if src.exists():
                    break
            else:
                return None
            dst = di / n
            if not dst.exists():
                os.symlink(src, dst)
            try:
                shutil.copyfile(labels_root / split / f"{Path(n).stem}.txt", dl / f"{Path(n).stem}.txt")
            except FileNotFoundError:
                pass                       # etiketsiz kare = arka plan
            return str(dst)

        # /kaggle/input ağ diski: tek tek dosya işlemleri yavaş, paralel yapılır
        paths, done, t0 = [], 0, time.time()
        with ThreadPoolExecutor(max_workers=32) as ex:
            for r in ex.map(prep, names, chunksize=64):
                done += 1
                if r:
                    paths.append(r)
                if done % 2500 == 0 or done == len(names):
                    print(f"  [{split}] hazırlanıyor {done}/{len(names)}  ({time.time() - t0:.0f} sn)", flush=True)
        miss = len(names) - len(paths)
        lists[split] = out / f"{split}_list.txt"
        lists[split].write_text("\n".join(paths), encoding="utf-8")
        print(f"  [{split}] {len(paths)} görüntü hazır" + (f", {miss} bulunamadı" if miss else ""), flush=True)
        if not paths:
            sys.exit(f"{split}: hiç görüntü eşleşmedi. Görüntü dosya adları ile listeler uyuşmuyor.")
    yaml = out / "data.yaml"
    yaml.write_text(f"path: {out}\ntrain: {lists['train'].name}\nval: {lists['val'].name}\nnames:\n"
                    + "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES)) + "\n", encoding="utf-8")
    return yaml


# ------------------------------------------------------------------ ana akış

def main():
    sys.stdout.reconfigure(line_buffering=True)   # Kaggle loglarında satırlar anında görünsün
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    ap.add_argument("--images-root", type=Path)
    ap.add_argument("--labels-root", type=Path)
    ap.add_argument("--splits-root", type=Path)
    ap.add_argument("--work", type=Path, default=Path("/kaggle/working"), help="sonuçlar (kaydedilir)")
    ap.add_argument("--ds", type=Path, default=Path("/tmp/adb_ds"),
                    help="veri ağacı; /kaggle/working DIŞINDA olmalı ki bağlantılar çıktıya kopyalanmasın")
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=8, help="16 GB GPU, 1280 px; çift T4'te toplam batch")
    ap.add_argument("--n-train", type=int, default=25000)
    ap.add_argument("--n-val", type=int, default=1500)
    ap.add_argument("--hours", type=float, default=11.0, help="12 saatlik oturum sınırının altında")
    ap.add_argument("--device", default=None, help="boş: GPU sayısına göre otomatik ('0' veya '0,1')")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--init-weights", type=Path)
    a = ap.parse_args()

    img = a.images_root or find_root(a.input, ".jpg")
    lbl = a.labels_root or find_root(a.input, ".txt")
    spl = a.splits_root or find_splits(a.input)
    print(f"Görüntüler: {img}\nEtiketler : {lbl}\nListeler  : {spl}")
    if not (img and lbl and spl):
        sys.exit("Veri bulunamadı. Notebook'a iki dataset'i de Input olarak eklediğinden emin ol, "
                 "ya da --images-root / --labels-root / --splits-root ver.")

    import torch
    n_gpu = torch.cuda.device_count()
    device = a.device or (",".join(str(i) for i in range(n_gpu)) if n_gpu > 1 else "0")
    print(f"GPU: {n_gpu} adet -> device={device}")
    if n_gpu == 0:
        print("UYARI: GPU yok. Notebook ayarlarında Accelerator = GPU seç.")

    yaml = build(img, lbl, spl, a.ds, a.n_train, a.n_val)
    if a.check:
        print("\n--check tamam, eğitim başlatılmadı.")
        return

    from ultralytics import YOLO
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        if ck.get("optimizer") is None:
            sys.exit(f"{a.resume} tamamlanmış bir eğitime ait, devam ettirilemez.\n"
                     f"Daha fazla eğitmek için: --init-weights {a.resume.parent / 'best.pt'}")
        local = a.work / "resume_last.pt"
        shutil.copyfile(a.resume, local)
        YOLO(str(local)).train(resume=True)
        return

    dev = [int(d) for d in device.split(",")] if "," in device else device
    YOLO(str(a.init_weights) if a.init_weights else a.model).train(
        data=str(yaml), imgsz=a.imgsz, batch=a.batch, epochs=100, time=a.hours,
        device=dev, workers=a.workers, project=str(a.work / "runs"), name=a.name, exist_ok=True,
        seed=42, amp=True, cos_lr=True, close_mosaic=3, patience=100, plots=True,
    )


if __name__ == "__main__":
    main()
