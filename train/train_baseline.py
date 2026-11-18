"""Faz 2 temel model eğitimi — RTX 5050 Laptop (8 GB) + Windows için ayarlı.

Tek komut:
  python train/train_baseline.py              # 8 saatlik gece eğitimi + bitince test değerlendirmesi
  python train/train_baseline.py --check      # sadece ön kontrol (bellek, veri, GPU), eğitim yok
  python train/train_baseline.py --quick      # 20 dk'lık deneme
  python train/train_baseline.py --resume     # elektrik/çökme sonrası kaldığı yerden

Neler yapıyor (önceki hataların hepsine karşı önlem):
  1. Ön kontrol: GPU, RAM, Windows disk belleği (pagefile) ölçülür; dataloader işçi sayısı buna
     göre seçilir (WinError 1455 / "DataLoader worker exited" hatalarının sebebi buydu).
  2. Eğitim listesi gece ağırlıklı kurulur (ADR-010): ~%75 gece, %10 alacakaranlık, %15 gündüz.
     Rastgele %40 alt kümede gece sadece ~10K kare olurdu; burada ~19K.
  3. Eğitim sırasında doğrulama 1.500 karelik küçük bir alt kümede yapılır (tam val 7K kare her
     epoch'ta ~5 dk yiyordu). Asıl ölçüm sonda test setinde koşul bazında yapılır.
  4. --hours ile epoch sayısı süreye sığdırılır; her epoch sonunda last.pt kaydedilir.
  5. Windows'un eğitim sırasında uykuya geçmesi engellenir (kapak kapatma ayarı hariç).
  6. Bitince eval/eval_conditions.py otomatik çalışır.

Ölçülen: 1280 px, ~19 görüntü/sn -> 25K kare ≈ 22 dk/epoch (+ ~1 dk doğrulama).
"""

import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "bdd_adb_lights"
NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person",
         "head_light", "tail_light", "street_light", "reflector"]
MIX = {  # koşul grubu -> eğitim listesindeki payı
    "night": (("night", "night_adverse"), 0.75),
    "dusk": (("dusk", "dusk_adverse"), 0.10),
    "day": (("day", "day_adverse"), 0.15),
}


# ------------------------------------------------------------------ ön kontrol

def preflight(want_workers):
    import psutil
    import torch

    print("Ön kontrol")
    ok = True
    if not torch.cuda.is_available():
        print("  [HATA] CUDA görünmüyor. 'python scripts/check_env.py' çalıştır.")
        return False, 0
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"  GPU: {p.name}, {total / 2**30:.1f} GB, boş {free / 2**30:.1f} GB")
    if free < 5.5 * 2**30:
        print("  [UYARI] GPU'da 5,5 GB'dan az boş alan var. Lively/Steam/oyunları kapat.")

    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    ram, avail, swap = vm.total / 2**30, vm.available / 2**30, sw.total / 2**30
    print(f"  RAM: {ram:.0f} GB (boş {avail:.1f} GB) | disk belleği (pagefile): {swap:.0f} GB")

    # Windows'ta her işçi PyTorch + CUDA DLL'lerini ayrıca yükler (~2-3 GB sanal bellek).
    if want_workers is not None:
        workers = want_workers
    elif swap < 8:
        workers = 0
    elif swap < 16 or avail < 6:
        workers = 1
    else:
        workers = 3
    if swap < 16:
        print("  [UYARI] Disk belleği küçük. Önerilen: Gelişmiş sistem ayarları > Performans > "
              "Sanal bellek > E: özel boyut 16384-32768 MB, sonra yeniden başlat.")
    if avail < 4:
        print("  [UYARI] Boş RAM 4 GB'ın altında. Chrome, Steam, WhatsApp vb. kapat.")
    print(f"  dataloader işçi sayısı: {workers}")

    for f in (DATA / "images" / "train", DATA / "labels" / "train", DATA / "images" / "val", DATA / "splits"):
        if not f.exists():
            print(f"  [HATA] {f} yok. Önce bdd_prepare.py ve autolabel_lights.py çalıştırılmalı.")
            ok = False
    n8 = 0
    for p_ in list((DATA / "labels" / "train").glob("*.txt"))[:2000]:
        n8 += any(ln.startswith("8 ") for ln in p_.read_text(encoding="utf-8").splitlines())
    if n8:
        print(f"  [UYARI] street_light etiketleri hâlâ var (örnekte {n8} dosya). Çalıştır:\n"
              f"          python scripts/drop_class.py --labels data/bdd_adb_lights/labels --cls 8")
    return ok, workers


# ------------------------------------------------------------------ veri listeleri

def read_split(split, cond):
    f = DATA / "splits" / split / f"{cond}.txt"
    return f.read_text(encoding="utf-8").split() if f.exists() else []


def build_lists(n_train, n_val, seed, out_dir: Path):
    rng = random.Random(seed)
    train = []
    print(f"Eğitim listesi ({n_train} kare hedef):")
    for group, (conds, share) in MIX.items():
        pool = [n for c in conds for n in read_split("train", c)]
        rng.shuffle(pool)
        take = pool[:round(n_train * share)]
        train += take
        print(f"  {group:6s}: {len(take):6d} / {len(pool)} mevcut")
    val_pool = [n for c in ("night", "night_adverse", "dusk", "day") for n in read_split("val", c)]
    rng.shuffle(val_pool)
    val = val_pool[:n_val]

    out_dir.mkdir(parents=True, exist_ok=True)
    tr = out_dir / "train_list.txt"
    va = out_dir / "val_small.txt"
    tr.write_text("\n".join(str((DATA / "images" / "train" / n).resolve()) for n in train), encoding="utf-8")
    va.write_text("\n".join(str((DATA / "images" / "val" / n).resolve()) for n in val), encoding="utf-8")
    yaml = out_dir / "data.yaml"
    yaml.write_text(f"path: {out_dir.resolve().as_posix()}\ntrain: {tr.name}\nval: {va.name}\nnames:\n"
                    + "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES)) + "\n", encoding="utf-8")
    print(f"  doğrulama alt kümesi: {len(val)} kare")
    return yaml, len(train)


# ------------------------------------------------------------------ Windows uyku engeli

def keep_awake(on: bool):
    if sys.platform != "win32":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


# ------------------------------------------------------------------ ana akış

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4, help="8 GB'ta 1280 px için 4 (8 sığmıyor)")
    ap.add_argument("--n-train", type=int, default=25000, help="eğitim listesi boyutu")
    ap.add_argument("--n-val", type=int, default=1500, help="eğitim sırası doğrulama alt kümesi")
    ap.add_argument("--hours", type=float, default=8.0, help="süre üst sınırı; epoch sayısı buna göre ayarlanır")
    ap.add_argument("--workers", type=int, default=None, help="boş bırak: ön kontrol seçsin")
    ap.add_argument("--name", default="baseline")
    ap.add_argument("--quick", action="store_true", help="1.000 kare, 0,3 saat")
    ap.add_argument("--check", action="store_true", help="sadece ön kontrol ve liste kurulumu")
    ap.add_argument("--resume", action="store_true", help="runs/<name>/weights/last.pt'den devam")
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()

    if a.quick:
        a.n_train, a.n_val, a.hours, a.name = 1000, 300, 0.3, a.name + "_quick"

    ok, workers = preflight(a.workers)
    if not ok:
        sys.exit(1)
    run_dir = ROOT / "runs" / a.name
    last = run_dir / "weights" / "last.pt"

    from ultralytics import YOLO

    keep_awake(True)
    t0 = time.time()
    try:
        if a.resume:
            if not last.exists():
                sys.exit(f"{last} yok, devam edilecek eğitim bulunamadı.")
            import torch
            if torch.load(last, map_location="cpu", weights_only=False).get("optimizer") is None:
                sys.exit("Bu eğitim zaten tamamlanmış (last.pt devam ettirilemez). "
                         f"Sonuç: {run_dir / 'weights' / 'best.pt'}")
            print(f"Devam ediliyor: {last}")
            YOLO(str(last)).train(resume=True, workers=workers)
        else:
            yaml, n = build_lists(a.n_train, a.n_val, 42, ROOT / "runs" / "lists" / a.name)
            if a.check:
                print("\n--check: ön kontrol tamam, eğitim başlatılmadı.")
                return
            est = n / 19 / 60
            print(f"\nTahmini epoch süresi: ~{est:.0f} dk | süre sınırı: {a.hours} saat\n")
            YOLO(a.model).train(
                data=str(yaml), imgsz=a.imgsz, batch=a.batch, epochs=100, time=a.hours,
                workers=workers, project=str(ROOT / "runs"), name=a.name, exist_ok=a.quick,
                seed=42, deterministic=False, amp=True, cos_lr=True, close_mosaic=3,
                patience=100, plots=True, cache=False,
            )
    finally:
        keep_awake(False)
    print(f"\nEğitim bitti: {(time.time() - t0) / 3600:.1f} saat. Ağırlıklar: {run_dir / 'weights'}")

    best = run_dir / "weights" / "best.pt"
    if not a.no_eval and best.exists():
        print("\nTest setinde koşul bazlı değerlendirme başlıyor...")
        subprocess.run([sys.executable, str(ROOT / "eval" / "eval_conditions.py"),
                        "--weights", str(best), "--batch", str(a.batch)], check=False)


if __name__ == "__main__":   # Windows'ta dataloader işçileri için zorunlu
    main()
