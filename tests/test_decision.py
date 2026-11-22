import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adb.decision import BeamController, glare_and_utilization  # noqa: E402
from adb.geometry import load_config  # noqa: E402

CFG = load_config()
CAR = np.array([[620., 340, 660, 370]])          # ortada, ~50 m


def ctrl(release=0.5):
    return BeamController(CFG, conf_thr=0.25, release_delay_s=release)


def test_detection_dims_center_segments_only():
    m = ctrl().step(CAR, [0], [0.9], 0.0)
    assert m.any() and not m.all()
    assert m[7] or m[8]                           # orta segmentler
    assert not m[0] and not m[15]


def test_low_confidence_ignored():
    assert not ctrl().step(CAR, [0], [0.1], 0.0).any()


def test_hysteresis_holds_then_releases():
    c = ctrl(release=0.5)
    on = c.step(CAR, [0], [0.9], 0.0)
    assert (c.step(np.zeros((0, 4)), [], [], 0.2) == on).all()     # 1 kare kayıp: hâlâ kısık
    assert (c.step(np.zeros((0, 4)), [], [], 0.5) == on).all()     # sınırda: hâlâ kısık
    assert not c.step(np.zeros((0, 4)), [], [], 0.7).any()         # süre doldu: açılır


def test_no_hysteresis_flickers():
    c = ctrl(release=0.0)
    c.step(CAR, [0], [0.9], 0.0)
    assert not c.step(np.zeros((0, 4)), [], [], 0.2).any()


def test_glare_violation_and_utilization():
    c = ctrl()
    cam, edges, lamps = c.cam, c.edges, c.lamps
    m = c.step(CAR, [0], [0.9], 0.0)
    v, inb, lit, free = glare_and_utilization(m, CAR, CAR, cam, edges, lamps)
    assert inb.tolist() == [True]
    assert v.tolist() == [False]                                    # araç korunuyor
    assert free == 14 and lit == 12                                 # güvenlik payı 2 boş segmenti de kısıyor
    v, _, _, _ = glare_and_utilization(np.zeros(16, bool), CAR, CAR, cam, edges, lamps)
    assert v.tolist() == [True]                                     # hiç kısma yok: ihlal


def test_near_vehicle_parallax_widens_interval():
    near = np.array([[900., 300, 1200, 600]])                       # sağda, çok yakın
    m = ctrl().step(near, [0], [0.9], 0.0)
    assert m[8:].sum() >= 3


def test_vehicle_outside_beam_not_counted():
    c = ctrl()
    side = np.array([[0., 300, 40, 340]])                           # en solda, ~-29°: far kapsamı (±16°) dışı
    v, inb, _, _ = glare_and_utilization(np.zeros(16, bool), side, side, c.cam, c.edges, c.lamps)
    assert inb.tolist() == [False] and v.tolist() == [False]