"""README ve dokümanlar için statik şekiller (docs/media/*.png).

Kaynak veriler repoda: docs/data/*.json (sonuç dosyalarının kopyaları), docs/results.md'deki
tablolar ve configs/*.yaml. Model veya veri seti gerekmez; şekiller her yerde yeniden üretilebilir.

Kullanım:
  python scripts/make_figures.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle, Wedge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adb.geometry import Camera, load_config  # noqa: E402

DATA, OUT = ROOT / "docs/data", ROOT / "docs/media"

# Kategorik palet (sabit sıra) ve metin tonları
C1, C2, C3, C4 = "#257de0", "#f47230", "#00b894", "#6a4ce2"
TXT, TXT2, GRID = "#191919", "#666666", "#e6e6e6"
GOOD, BAD = "#008a17", "#d03b3b"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5, "axes.edgecolor": "#bbbbbb", "axes.labelcolor": TXT,
    "xtick.color": TXT2, "ytick.color": TXT2, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 12, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
    "axes.axisbelow": True, "savefig.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": "white",
})

CFG = load_config()
CAM = Camera.from_config(CFG)


def tr(x: float, nd: int = 2) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  {name}")


# ---------------------------------------------------------------- 1. piksel bütçesi
def pixel_budget():
    d = np.linspace(10, 300, 300)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for w, lab, c in ((1.8, "araç gövdesi (1,8 m)", C1), (1.5, "far çifti arası (1,5 m)", C2), (0.2, "tek far / lamba (0,2 m)", C3)):
        ax.plot(d, CAM.fx_px * w / d, color=c, lw=2, label=lab)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 200)
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_yticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
    ax.axvline(220, color=TXT2, ls=":", lw=1)
    ax.text(217, 0.58, "220 m (FMVSS 108\ntest mesafesi)", color=TXT2, fontsize=9, va="bottom", ha="right")
    ax.annotate("220 m'de far çifti 8 px,\ntek lamba ~1 px", xy=(220, CAM.fx_px * 1.5 / 220), xytext=(120, 2.2),
                arrowprops=dict(arrowstyle="->", color=TXT2), fontsize=9.5, color=TXT)
    ax.set_xlabel("Mesafe (m)")
    ax.set_ylabel("Görüntüdeki genişlik (piksel, log)")
    ax.set_title("Uzaktaki araç gece yalnızca birkaç piksellik iki ışık noktası")
    ax.legend(loc="upper right")
    save(fig, "pixel_budget.png")


# ---------------------------------------------------------------- 2. gece etiket açığı
def label_gap():
    bins = ["< 62 m", "125–166 m", "166–249 m", "> 249 m"]
    vals = [0.855, 0.55, 0.37, 0.29]
    labs = ["%83–88", "%55", "%37", "%29"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(bins, vals, color=C1, width=0.55)
    for b, s in zip(bars, labs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, s, ha="center", color=TXT, fontsize=10)
    ax.axhline(0.855, color=TXT2, ls=":", lw=1)
    ax.text(3.35, 0.87, "gecenin doğal\ntrafik oranı", color=TXT2, fontsize=8.5, ha="right", va="bottom")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "%25", "%50", "%75", "%100"])
    ax.set_ylabel("Gece / gündüz etiket oranı")
    ax.grid(axis="x", visible=False)
    ax.set_title("BDD100K gece etiketlerinde uzak araçlar eksik (ADR-007)")
    save(fig, "label_gap.png")


# ---------------------------------------------------------------- 3. mesafe bazlı recall
def recall_distance():
    d = json.loads((DATA / "eval_distance_fp16_gpu.json").read_text(encoding="utf-8"))
    bins = d["bins_px"]
    r = d["results"]["0.1"]
    mids = [CAM.fx_px * 1.8 / math.sqrt(lo * min(hi, 128)) for lo, hi in bins]
    labels = []
    for lo, hi in bins:
        far, near = CAM.fx_px * 1.8 / lo, CAM.fx_px * 1.8 / hi
        labels.append(f"> {far:.0f}" if lo == 4 else (f"< {far:.0f}" if hi > 1000 else f"{near:.0f}–{far:.0f}"))
    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    for key, lab, c in (("night·tam", "tam kare", C1), ("night·bant", "tam kare + ufuk bandı", C2)):
        y = [a / g for a, g in zip(r[key]["any"], r[key]["gt"])]
        ax.plot(mids, y, "-o", color=c, lw=2, ms=6, label=lab)
    yt = r["night·tam"]["any"][1] / r["night·tam"]["gt"][1]
    yb = r["night·bant"]["any"][1] / r["night·bant"]["gt"][1]
    ax.annotate(f"{tr(yt)} → {tr(yb)}", xy=(mids[1], yb), xytext=(mids[1] + 40, yb + 0.07),
                arrowprops=dict(arrowstyle="->", color=TXT2), fontsize=10, color=TXT, fontweight="bold")
    ax.axhline(0.90, color=TXT2, ls=":", lw=1)
    ax.text(375, 0.893, "hedef 0,90 (REQ-F-01)", color=TXT2, fontsize=9, va="top", ha="right")
    ax.axvline(220, color=TXT2, ls=":", lw=1)
    ax.text(223, 0.52, "220 m", color=TXT2, fontsize=9)
    ax.set_xlim(0, 380)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Yaklaşık mesafe (m)")
    ax.set_ylabel("Recall (araç kutusu veya ışığı)")
    ax.set_title("Gece mesafe bazlı recall: ufuk bandı dilimleme uzak araçları yakalıyor")
    ax.legend(loc="lower left")
    save(fig, "recall_distance.png")


# ---------------------------------------------------------------- 4. Faz 3 deneyleri
def phase3():
    rows = [("1b · temel model", 0.75, "1×"), ("2 · ufuk bandı dilimleme", 0.85, "~2×"),
            ("3 · sözde etiketleme", 0.75, "1×"), ("4 · P2 tespit kafası", 0.74, "1,3×"),
            ("5 · giriş 2560 px", 0.79, "4×")]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(rows))[::-1]
    cols = [TXT2 if i == 0 else (C2 if i == 1 else C1) for i in range(len(rows))]
    ax.barh(y, [r[1] for r in rows], color=cols, height=0.55)
    for yi, (lab, v, cost) in zip(y, rows):
        ax.text(v + 0.003, yi, f"{tr(v)}   hesap {cost}", va="center", fontsize=9.5, color=TXT)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0.70, 0.92)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: tr(v)))
    ax.axvline(0.90, color=TXT2, ls=":", lw=1)
    ax.text(0.898, -0.45, "hedef 0,90", color=TXT2, fontsize=9, ha="right")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("166–249 m recall (gece)")
    ax.set_title("Faz 3: uzak araçları bulduran yöntem hangisi?")
    save(fig, "phase3_experiments.png")


# ---------------------------------------------------------------- 5. gecikme
def latency():
    runs = [("Ultralytics\n(Deney 11)", "latency_ultralytics.json"),
            ("GPU hattı\nCUDA Graph yok", "latency_gpu_nograph.json"),
            ("GPU hattı\n+ CUDA Graph (Deney 12)", "latency_gpu.json")]
    stages = [("tam kare", ["tam kare"], C1), ("ufuk bandı", ["bant"], C2),
              ("birleştir + takip + karar", ["birleştir", "takip", "karar"], C3), ("güvenlik izleyici", ["izleyici"], C4)]
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    y = np.arange(len(runs))[::-1]
    for yi, (lab, f) in zip(y, runs):
        res = json.loads((DATA / f).read_text(encoding="utf-8"))["results"]["1"]
        left = 0.0
        for name, keys, c in stages:
            w = sum(res[k]["mean"] for k in keys)
            ax.barh(yi, w, left=left, color=c, height=0.5, edgecolor="white", linewidth=2,
                    label=name if yi == y[0] else None)
            left += w
        p95 = res["TOPLAM"]["p95"]
        ax.plot([p95, p95], [yi - 0.32, yi + 0.32], color=TXT, lw=2)
        ax.text(p95 + 1.2, yi, f"p95 {tr(p95, 1)} ms", va="center", fontsize=9.5, color=TXT)
    for x, t in ((33.3, "30 fps kare aralığı"), (50, "hedef ≤ 50 ms (REQ-P-06)")):
        ax.axvline(x, color=BAD if x == 50 else TXT2, ls=":", lw=1.2)
        ax.text(x + 0.8, y[0] + (0.48 if x == 50 else -2.55), t, fontsize=8.5, color=BAD if x == 50 else TXT2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in runs])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 2.75)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Kare başına süre (ms) · çubuk: ortalama, aşamalara göre · çizgi: p95")
    ax.set_title("Faz 5: bant her karede, uçtan uca gecikme (RTX 5050, TensorRT FP16)")
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    save(fig, "latency.png")


# ---------------------------------------------------------------- 6. far geometrisi
def beam_geometry():
    from adb.decision import BeamController

    bc = BeamController(CFG, 0.35)
    fx = CAM.fx_px
    # 60 m'de, 2,5° sağda karşıdan gelen araç
    dist, az = 60.0, 2.5
    u = CAM.cx_px + fx * math.tan(math.radians(az))
    w = fx * 1.8 / dist
    box = np.array([[u - w / 2, 360, u + w / 2, 360 + w * 0.6]])
    mask = bc.step(box, [0], [0.9], 0.0)
    K, R = 2.2, 10.0
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.set_aspect("equal")
    ax.axis("off")
    for (a, b), d in zip(bc.edges, mask):
        ax.add_patch(Wedge((0, 0), R, 90 - K * b, 90 - K * a, facecolor="#3a3a3a" if d else "#ffd24d",
                           edgecolor="white", lw=1.5))
    # araç
    r = dist / 100 * R
    x, yy = r * math.sin(math.radians(K * az)), r * math.cos(math.radians(K * az))
    ax.add_patch(FancyBboxPatch((x - 0.22, yy - 0.3), 0.44, 0.6, boxstyle="round,pad=0.02", fc=C1, ec="white", lw=1.5))
    ax.add_patch(FancyBboxPatch((-0.3, -0.75), 0.6, 0.9, boxstyle="round,pad=0.03", fc="#555555", ec="white"))
    ax.text(0, -1.15, "kendi aracımız\n(kamera + 2 far)", ha="center", va="top", fontsize=9, color=TXT)
    ax.annotate("karşıdan gelen araç\n~60 m, 2,5° sağda", xy=(x + 0.25, yy), xytext=(x + 2.8, yy + 1.2),
                arrowprops=dict(arrowstyle="->", color=TXT2), fontsize=9.5, color=TXT)
    on = np.where(mask)[0]
    a0, a1 = bc.edges[on[0]][0], bc.edges[on[-1]][1]
    ax.annotate(f"kısılan {len(on)} segment ({a0:+.0f}° … {a1:+.0f}°):\naraç + 1° güvenlik payı + paralaks",
                xy=(R * 0.92 * math.sin(math.radians(K * (a0 + a1) / 2)), R * 0.92 * math.cos(math.radians(K * (a0 + a1) / 2))),
                xytext=(-10.5, 8.6), arrowprops=dict(arrowstyle="->", color=TXT2), fontsize=9.5, color=TXT)
    ax.text(-R - 0.5, 0.2, "16 segmentlik LED matrisi, ±16°\n(açılar okunaklılık için\n2,2 kat genişletildi)",
            fontsize=9, color=TXT2, va="bottom")
    ax.add_patch(Rectangle((5.9, 8.9), 0.45, 0.3, fc="#ffd24d", ec="white"))
    ax.text(6.5, 9.05, "uzun far açık", va="center", fontsize=9, color=TXT)
    ax.add_patch(Rectangle((5.9, 8.4), 0.45, 0.3, fc="#3a3a3a", ec="white"))
    ax.text(6.5, 8.55, "kısık", va="center", fontsize=9, color=TXT)
    ax.set_xlim(-R - 1, R + 1)
    ax.set_ylim(-2, R + 0.6)
    ax.set_title("Far kararı: yalnızca diğer aracın bulunduğu açı dilimi karartılır", loc="left")
    save(fig, "beam_geometry.png")


# ---------------------------------------------------------------- 7. ufuk bandı dilimleme
def slicing():
    import cv2

    from adb.slicing import load_slicing, tile_xs

    src = OUT / "pexels_15119004_t15.jpg"
    if not src.exists():
        print("  (atlandı: kare yok)")
        return
    img = cv2.cvtColor(cv2.imread(str(src)), cv2.COLOR_BGR2RGB)
    img = np.clip(img.astype(float) * 1.8, 0, 255).astype(np.uint8)          # karanlık kare, görünürlük için
    sl = load_slicing()
    H, W = img.shape[:2]
    y0, y1 = (int(round(v * H)) for v in sl["band_frac"])
    xs = tile_xs(W, sl["tile_w"], sl["overlap"])
    cols = [C1, C2, C3, C4]
    fig = plt.figure(figsize=(10, 7.4))
    ax = fig.add_axes([0, 0.36, 1, 0.58])
    ax.imshow(img)
    ax.axis("off")
    ax.add_patch(Rectangle((0, y0), W - 1, y1 - y0, fill=False, ec="white", lw=1.5, ls="--"))
    for i, (x, c) in enumerate(zip(xs, cols)):
        ax.add_patch(Rectangle((x + 2, y0 + 4 + 3 * i), sl["tile_w"] - 4, y1 - y0 - 8 - 6 * i, fill=False, ec=c, lw=2.2))
    ax.text(8, y0 - 10, "ufuk bandı: 18.589 uzak gece aracının %2–98 dilimini kapsar (veriden, scripts/horizon_band.py)",
            color="white", fontsize=9.5, va="bottom")
    ax.set_title("Ufuk bandı dilimleme: tam kareye ek olarak 4 parça, 2 kat büyütülerek ayrıca taranır",
                 loc="left", fontsize=12.5)
    for i, (x, c) in enumerate(zip(xs, cols)):
        t = cv2.resize(img[y0:y1, x:x + sl["tile_w"]], None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        a = fig.add_axes([0.005 + i * 0.25, 0.02, 0.24, 0.31])
        a.imshow(t)
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(c)
            s.set_linewidth(2.5)
            s.set_visible(True)
        a.set_title(f"parça {i + 1} · 2× → 768×704", fontsize=9, color=TXT, loc="center", pad=4)
    fig.text(0.005, -0.01, "Görüntü: Pexels (video 15119004), parlaklık artırıldı.", fontsize=8, color=TXT2)
    save(fig, "slicing.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Şekiller -> {OUT}")
    for f in (pixel_budget, label_gap, recall_distance, phase3, latency, beam_geometry, slicing):
        f()


if __name__ == "__main__":
    main()
