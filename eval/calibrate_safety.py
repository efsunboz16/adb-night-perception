"""Güvenlik izleyici kalibrasyonu: görüntü kalitesi ölçülerinin videolara göre dağılımı.

REQ-S-02: görüntü bulanıksa (kirli lens, hareket bulanıklığı) veya aşırı parlaksa tüm uzun far kapanır.
configs/system.yaml'daki eşikler tahmindi. Bu script her gece videosu için ölçüleri hesaplar ve
kamaşma teşhisinde (eval/diagnose_glare.py) ışıklı araç ihlali çıkan videoları işaretler. Eşik,
bozuk videoları yakalayıp sağlam videoları mümkün olduğunca az etkileyecek şekilde seçilmeli.

Ölçüler (gri, 640x360'a küçültülmüş, etiketli kare ve öncesindeki 5 kare ortancası):
  keskinlik : Laplace varyansı (düşük = bulanık)
  parlak_kenar: parlak (>= 200) piksellerin çevresindeki Laplace enerjisi. Gece görüntüsünde detay
                çoğunlukla ışık kaynaklarının kenarlarındadır; karanlık bölgeler genel varyansı düşürür,
                bu ölçü ışıkların ne kadar keskin olduğuna bakar.
  parlaklık : ortalama gri (0–1)

Kullanım:
  python eval/calibrate_safety.py --diag runs/baseline/glare_diag/summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from eval_video import resolve  # noqa: E402


def blob_stats(g):
    """Işık lekelerinin şekli (gri, 640x360).
    uzama : parlak (>= 200) lekelerin ana/yan eksen oranının ortancası. Durağan ışık ~1–2,
            hareket bulanıklığında ışıklar çizgi olur (yüksek).
    hale  : yarı parlak (>= 100) alanın parlak (>= 200) alana oranı, lekeler toplamında. Islak/kirli
            ön camda her ışığın çevresinde geniş hale oluşur (yüksek)."""
    import cv2
    core = (g >= 200).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(core, 8)
    el = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 6:
            continue
        ys, xs = np.nonzero(lab == i)
        c = np.cov(np.stack([xs, ys]).astype(float))
        ev = np.sort(np.linalg.eigvalsh(c))
        el.append(float(np.sqrt(ev[1] / max(ev[0], 0.25))))
    halo_area = float((g >= 100).sum())
    core_area = float(core.sum())
    halo = halo_area / core_area if core_area >= 30 else float("nan")
    return (float(np.median(el)) if len(el) >= 3 else float("nan")), halo


def quality(img):
    import cv2
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (640, 360), interpolation=cv2.INTER_AREA)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    bright = cv2.dilate((g >= 200).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    edge = float(np.mean(lap[bright] ** 2)) if bright.sum() > 50 else float("nan")
    el, halo = blob_stats(g)
    return float(lap.var()), edge, float(g.mean() / 255), el, halo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=Path, default=ROOT / "data/bdd_video/videos.json")
    ap.add_argument("--data", type=Path, default=ROOT / "data/bdd_adb")
    ap.add_argument("--conds", nargs="+", default=["night", "night_adverse"])
    ap.add_argument("--diag", type=Path, help="diagnose_glare summary.json (ihlal çıkan videoları işaretlemek için)")
    ap.add_argument("--frames", type=int, default=6)
    a = ap.parse_args()
    import cv2

    vids = {k: v for k, v in json.loads(a.videos.read_text(encoding="utf-8")).items() if v["cond"] in a.conds}
    bad = {}
    if a.diag and a.diag.exists():
        for r in json.loads(a.diag.read_text(encoding="utf-8")):
            if r.get("lit"):
                bad[r["video"]] = bad.get(r["video"], 0) + 1
    rows = []
    for name, v in vids.items():
        frames = sorted(resolve(v["frames_dir"]).glob(f"{name}-*.jpg"))
        key = v["key_index"]
        q = [quality(cv2.imread(str(f))) for f in frames[max(0, key - a.frames + 1):key + 1]]
        q = np.array(q)
        rows.append((name, *np.nanmedian(q, axis=0), bad.get(name, 0)))   # (ad, keskinlik, kenar, parlaklık, uzama, hale, ihlal)

    rows.sort(key=lambda r: r[1])
    print(f"\n### Görüntü kalitesi, {len(rows)} gece videosu (keskinliğe göre sıralı, en bulanık üstte)\n")
    print("| Video | Keskinlik | Parlak kenar | Parlaklık | Uzama | Hale | Işıklı araç ihlali |")
    print("|---|---|---|---|---|---|---|")
    for n, s, e, b, el, ha, k in rows:
        mark = f"**{k}**" if k else ""
        print(f"| {n} | {s:.1f} | {e:.0f} | {b:.2f} | {el:.2f} | {ha:.1f} | {mark} |")
    s = np.array([r[1] for r in rows])
    e = np.array([r[2] for r in rows])
    print(f"\nKeskinlik yüzdelikleri: %5 {np.percentile(s, 5):.1f} | %10 {np.percentile(s, 10):.1f} | "
          f"%25 {np.percentile(s, 25):.1f} | ortanca {np.median(s):.1f}")
    print(f"Parlak kenar yüzdelikleri: %5 {np.nanpercentile(e, 5):.0f} | %10 {np.nanpercentile(e, 10):.0f} | "
          f"%25 {np.nanpercentile(e, 25):.0f} | ortanca {np.nanmedian(e):.0f}")
    if bad:
        print("\nİhlalli videoların sıradaki yeri (bozulma yönünde; 1 = en bozuk görünen):")
        for metric, idx, rev in (("Keskinlik (düşük)", 1, False), ("Parlak kenar (düşük)", 2, False),
                                 ("Uzama (yüksek)", 4, True), ("Hale (yüksek)", 5, True)):
            vals = sorted(((r[idx], r[0] in bad) for r in rows if not np.isnan(r[idx])), reverse=rev)
            print(f"  {metric}: " + ", ".join(str(i + 1) for i, (_, b) in enumerate(vals) if b) + f" / {len(vals)}")


if __name__ == "__main__":
    main()