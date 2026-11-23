import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
from diagnose_glare import classify, lit_flags  # noqa: E402

G = np.array([600., 340, 660, 380])
NONE = (np.zeros((0, 4)), np.zeros(0, int), np.zeros(0))


def det(*items):
    return (np.array([i[0] for i in items], float), np.array([i[1] for i in items]), np.array([i[2] for i in items]))


def test_categories():
    car_hi = det(([602, 341, 659, 379], 0, 0.8))
    car_lo = det(([602, 341, 659, 379], 0, 0.2))
    light = det(([640, 360, 644, 364], 7, 0.8))
    assert classify(G, NONE, NONE, 0.35, True) == "görülmedi"
    assert classify(G, car_lo, car_lo, 0.35, True) == "düşük güven"
    assert classify(G, car_hi, NONE, 0.35, True) == "takip onayı"
    assert classify(G, car_hi, NONE, 0.35, False) == "düşük güven"
    assert classify(G, light, light, 0.35, True) == "yalnız ışık"
    assert classify(G, car_hi, car_hi, 0.35, True) == "geometri"


def test_lit_flags():
    img = np.full((720, 1280, 3), 20, np.uint8)
    gt = np.array([[600., 340, 660, 380], [100., 300, 200, 360]])
    img[370:374, 640:646] = 255                     # birinci aracın alt kısmında lamba
    img[302:305, 150:153] = 255                     # ikinci aracın en üstünde parlaklık (ön cam yansıması)
    assert lit_flags(img, gt).tolist() == [True, False]