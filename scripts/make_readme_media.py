"""Faz 6 · README görselleri: demo videosundan GIF ve kareler, doğrulama özeti.

Girdiler (senin makinende üretilir):
  runs/demo/<ad>.mp4                         scripts/render_demo.py çıktısı
  runs/trt/full_fp16_gpu/verification.json   eval/verify_requirements.py çıktısı
Çıktılar (repoya girer):
  docs/media/demo.gif               README'nin başındaki hareketli görüntü (~10 sn)
  docs/media/demo_frames.jpg        üç anahtar kare yan yana
  docs/media/verification.png       gereksinimlerin durum özeti

Kullanım:
  python scripts/make_readme_media.py --demo runs/demo/pexels_15119004.mp4 --start 12 --duration 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/media"


def read_frames(path: Path, start: float, duration: float, fps_out: float, width: int):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / fps_out))
    frames, k = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        t = k / fps
        k += 1
        if t < start or (k - 1) % step:
            continue
        if t > start + duration:
            break
        h = int(round(f.shape[0] * width / f.shape[1]))
        frames.append(cv2.cvtColor(cv2.resize(f, (width, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps / step


def make_gif(frames, fps, out: Path):
    from PIL import Image

    ims = [Image.fromarray(f) for f in frames]
    # Ortak palet: 8 kareden örnek + bindirme renkleri (sarı segment, yeşil kutu, kırmızı uyarı...).
    # Tek kareden palet alınınca o karede olmayan renkler (ör. açık segment sarısı) bozuluyordu.
    idx = np.linspace(0, len(frames) - 1, min(8, len(frames))).astype(int)
    small = [cv2.resize(frames[i], (frames[i].shape[1] // 2, frames[i].shape[0] // 2)) for i in idx]
    sw = np.zeros((small[0].shape[0], 40, 3), np.uint8)
    for j, c in enumerate([(255, 215, 0), (60, 60, 60), (80, 220, 80), (255, 255, 255), (255, 60, 60),
                           (60, 180, 220), (200, 160, 0), (45, 45, 45), (230, 230, 230), (0, 160, 200)]):
        sw[j * sw.shape[0] // 10:(j + 1) * sw.shape[0] // 10] = c
    pal = Image.fromarray(np.hstack(small + [sw])).quantize(colors=192, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
    q[0].save(out, save_all=True, append_images=q[1:], duration=int(1000 / fps), loop=0, optimize=True)
    print(f"  {out.name}: {len(q)} kare, {out.stat().st_size / 1e6:.1f} MB")


def make_frames(path: Path, times: list[float], out: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tiles = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if ok:
            tiles.append(cv2.resize(f, (640, 360), interpolation=cv2.INTER_AREA))
    cap.release()
    if tiles:
        gap = np.full((360, 8, 3), 255, np.uint8)
        row = tiles[0]
        for t in tiles[1:]:
            row = np.hstack([row, gap, t])
        cv2.imwrite(str(out), row, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"  {out.name}: {len(tiles)} kare")


def verification_figure(vjson: Path, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    rows = json.loads(vjson.read_text(encoding="utf-8"))
    col = {"GEÇTİ": ("#008a17", "✓"), "KISMEN": ("#e08a00", "~"), "KALDI": ("#d03b3b", "✗"),
           "ÖLÇÜLMEDİ": ("#8a8a8a", "–"), "UYGULANMADI": ("#5a5a5a", "∅")}
    groups = [("F", "Fonksiyonel"), ("P", "Performans"), ("S", "Güvenlik")]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.axis("off")
    ax.set_xlim(0, 3)
    ax.set_ylim(-0.5, 8.6)
    for gi, (g, name) in enumerate(groups):
        items = [r for r in rows if r["id"].split("-")[1] == g]
        ax.text(gi + 0.05, 8.3, name, fontsize=13, fontweight="bold", color="#191919")
        for i, r in enumerate(items):
            c, sym = col[r["status"]]
            y = 7.4 - i * 0.98
            ax.add_patch(FancyBboxPatch((gi + 0.05, y - 0.38), 0.9, 0.78, boxstyle="round,pad=0.01,rounding_size=0.05",
                                        fc=c, ec="white", lw=2, alpha=0.95))
            ax.text(gi + 0.09, y + 0.12, f"{sym}  {r['id']}", fontsize=10, fontweight="bold", color="white", va="center")
            txt = r["text"] if len(r["text"]) <= 38 else r["text"][:36] + "…"
            ax.text(gi + 0.09, y - 0.2, txt, fontsize=8.6, color="white", va="center")
    from collections import Counter
    cnt = Counter(r["status"] for r in rows)
    names = {"GEÇTİ": "geçti", "KISMEN": "kısmen", "KALDI": "kaldı", "ÖLÇÜLMEDİ": "ölçülmedi", "UYGULANMADI": "uygulanmadı"}
    leg = "   ".join(f"{col[k][1]} {names[k]} {cnt[k]}" for k in col if cnt[k])
    ax.text(0.05, -0.35, leg, fontsize=10, color="#191919")
    fig.suptitle("Gereksinim doğrulama özeti (docs/verification.md)", x=0.07, ha="left", fontsize=14, fontweight="bold")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {out.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", type=Path, help="render_demo.py çıktısı (.mp4)")
    ap.add_argument("--start", type=float, default=12.0)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--gif-fps", type=float, default=10.0)
    ap.add_argument("--gif-width", type=int, default=720)
    ap.add_argument("--frames-at", type=float, nargs="+", help="anahtar kare saniyeleri (boş: GIF aralığından 3 kare)")
    ap.add_argument("--verification", type=Path, default=ROOT / "runs/trt/full_fp16_gpu/verification.json")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"-> {OUT}")
    if a.demo:
        if not a.demo.exists():
            sys.exit(f"{a.demo} yok (önce scripts/render_demo.py)")
        frames, fps = read_frames(a.demo, a.start, a.duration, a.gif_fps, a.gif_width)
        make_gif(frames, fps, OUT / "demo.gif")
        ts = a.frames_at or [a.start + a.duration * f for f in (0.15, 0.5, 0.85)]
        make_frames(a.demo, ts, OUT / "demo_frames.jpg")
    if a.verification.exists():
        verification_figure(a.verification, OUT / "verification.png")
    else:
        print(f"  ({a.verification} yok, doğrulama özeti atlandı)")


if __name__ == "__main__":
    main()
