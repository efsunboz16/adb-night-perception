"""Gereksinimlerin dayanağı: hedef mesafelerde araçlar ve ışıklar görüntüde kaç piksel?

Kullanım: python scripts/pixel_budget.py [--config configs/system.yaml]
Kamera değişirse (hfov, çözünürlük) bu tabloyu yeniden üret ve docs/requirements.md'yi güncelle.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adb.geometry import Camera, camera_to_lamp_azimuth, load_config  # noqa: E402

OBJECTS = {
    "far çifti arası (1,5 m)": 1.5,
    "binek araç genişliği (1,8 m)": 1.8,
    "tek far/lamba (0,2 m)": 0.2,
}
DISTANCES = [15, 30, 50, 100, 150, 220, 300, 400]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    cam = Camera.from_config(cfg)
    lamp = cfg["headlamp"]["lamps"]["left"]

    print(f"Kamera: {cam.width_px}x{cam.height_px}, fx = {cam.fx_px:.0f} px "
          f"(1 px ≈ {cam.px_to_azimuth(cam.cx_px + 1):.3f}°)\n")
    print("| Mesafe (m) | " + " | ".join(OBJECTS) + " |")
    print("|---" * (len(OBJECTS) + 1) + "|")
    for d in DISTANCES:
        row = [f"{cam.object_width_px(w, d):.1f}" for w in OBJECTS.values()]
        print(f"| {d} | " + " | ".join(row) + " |")

    print("\nParalaks: kamerada 5° sağda görünen araç, sol farın baktığı açı")
    print("| Mesafe (m) | Far açısı (°) | Fark (°) |\n|---|---|---|")
    for d in DISTANCES:
        a = camera_to_lamp_azimuth(5.0, d, lamp["x_m"], lamp["z_m"])
        print(f"| {d} | {a:.2f} | {a - 5.0:+.2f} |")


if __name__ == "__main__":
    main()
