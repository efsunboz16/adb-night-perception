"""Koşul bazında test değerlendirmesi (REQ-F-01..02 için temel tablo).

BDD test bölümünü koşullara (gündüz, gece, yağmurlu gece, ...) ayırır, her biri için
araç sınıflarında (0-5) mAP hesaplar. Işık sınıfları test'te etiketli olmadığından dahil edilmez.

Kullanım:
  python eval/eval_conditions.py --weights runs/baseline/weights/best.pt
  python eval/eval_conditions.py --weights deploy/models/val_fp16.engine     # FP16 doğruluk eşitliği (Faz 5)
Çıktı: ekrana Markdown tablo + runs/<ad>/eval_conditions.json (motorlarda runs/trt/<motor adı>/)

Motor notu: Ultralytics'in val modu dikdörtgen sabit girişi desteklemiyor (imgsz tek sayıya indiriliyor).
mAP için kare (1280×1280) bir doğrulama motoru kullanılır: python deploy/export.py --val
Aynı ağırlıklar, aynı hassasiyet; yalnızca dolgu farklı. Uzak recall ve sistem metrikleri ise asıl
dikdörtgen motorlarla ölçülür (eval_distance.py, eval_video.py).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adb.models import engine_meta, is_engine, run_dir_for  # noqa: E402
NAMES = ["car", "truck", "bus", "motorcycle", "bicycle", "person",
         "head_light", "tail_light", "street_light", "reflector"]
CONDS = ["day", "night", "night_adverse", "dusk", "day_adverse"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/bdd_adb"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=1,
                    help="Windows'ta bellek hatası olursa 0 (her işçi PyTorch'u ayrıca yükler)")
    a = ap.parse_args()

    from ultralytics import YOLO

    data = Path(a.data).resolve()
    run_dir = run_dir_for(a.weights)
    if is_engine(a.weights):
        meta = engine_meta(a.weights) or {}
        sz = meta.get("imgsz")
        if not sz or len(set(sz)) != 1:
            sys.exit(f"{a.weights}: val için kare motor gerekli (bu motor {sz}). "
                     f"Üret: python deploy/export.py --val")
        a.imgsz, a.batch = int(sz[0]), int(meta.get("batch", 1))
        print(f"Motor: {Path(a.weights).name}, {a.imgsz}×{a.imgsz}, batch {a.batch}")
    tmp = run_dir / "eval_lists"
    tmp.mkdir(exist_ok=True)
    model = YOLO(a.weights)
    results = {}
    for cond in CONDS:
        lst = data / "splits" / "test" / f"{cond}.txt"
        if not lst.exists():
            continue
        imgs = [str(data / "images" / "test" / n) for n in lst.read_text(encoding="utf-8").split()]
        (tmp / f"{cond}.txt").write_text("\n".join(imgs), encoding="utf-8")
        yaml = tmp / f"{cond}.yaml"
        names = "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES))
        yaml.write_text(f"path: {tmp.as_posix()}\ntrain: {cond}.txt\nval: {cond}.txt\nnames:\n{names}\n",
                        encoding="utf-8")
        m = model.val(data=str(yaml), imgsz=a.imgsz, batch=a.batch, classes=list(range(6)),
                      workers=a.workers, half=True,
                      plots=False, verbose=False, project=str(run_dir), name=f"val_{cond}", exist_ok=True)
        per_cls = {NAMES[int(c)]: round(float(ap50), 3) for c, ap50 in zip(m.box.ap_class_index, m.box.ap50)}
        results[cond] = {"images": len(imgs), "mAP50": round(float(m.box.map50), 3),
                         "mAP50-95": round(float(m.box.map), 3), "AP50_per_class": per_cls}

    print("\n| Koşul | Görüntü | mAP50 | mAP50-95 | car | truck | bus | motorcycle | bicycle | person |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for cond, r in results.items():
        cls = " | ".join(str(r["AP50_per_class"].get(n, "-")) for n in NAMES[:6])
        print(f"| {cond} | {r['images']} | {r['mAP50']} | {r['mAP50-95']} | {cls} |")
    out = run_dir / "eval_conditions.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nKaydedildi: {out}")


if __name__ == "__main__":
    main()
