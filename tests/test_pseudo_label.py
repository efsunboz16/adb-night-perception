import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from pseudo_label import select_pseudo  # noqa: E402

P = {"light_conf": 0.45, "veh_conf": 0.5, "far_veh_px": 32, "light_max_px": 16,
     "dup_px": 3.0, "pair_dx": (3.0, 60.0), "pair_dy": 4.0, "min_peak": 200, "rule_a": True}
BAND = (300.0, 460.0)


def run(preds, gt, peaks=None, **over):
    pb = np.array([p[0] for p in preds], float).reshape(-1, 4)
    pc = np.array([p[1] for p in preds], int)
    pf = np.array([p[2] for p in preds], float)
    gc = np.array([g[1] for g in gt], int)
    gb = np.array([g[0] for g in gt], float).reshape(-1, 4)
    pk = None if peaks is None else np.array(peaks, float)
    return [(c, rule) for c, _, rule in select_pseudo(pb, pc, pf, gc, gb, BAND, {**P, **over}, pred_peak=pk)]


CAR = ([500, 350, 560, 400], 0)              # BDD araç kutusu


def test_A_light_in_lower_part_of_box_added():
    assert run([([520, 385, 524, 389], 7, 0.6)], [CAR]) == [(7, "A")]


def test_A_light_in_windshield_area_skipped():
    # kutunun üst %25'i (y < 362.5): ön cam yansıması olabilir
    assert run([([520, 352, 524, 356], 7, 0.9)], [CAR]) == []


def test_existing_light_label_not_duplicated():
    assert run([([520, 385, 524, 389], 7, 0.9)], [CAR, ([519, 384, 525, 390], 7)]) == []


def test_B_unboxed_pair_added_single_skipped():
    pair = [([800, 380, 803, 383], 6, 0.5), ([812, 380, 815, 383], 6, 0.5)]
    assert run(pair, []) == [(6, "B"), (6, "B")]
    assert run(pair[:1], []) == []                                  # tek ışık: sokak lambası olabilir


def test_B_pair_outside_band_or_mixed_class_skipped():
    assert run([([800, 200, 803, 203], 6, 0.9), ([812, 200, 815, 203], 6, 0.9)], []) == []   # bant dışı
    assert run([([800, 380, 803, 383], 6, 0.9), ([812, 380, 815, 383], 7, 0.9)], []) == []   # far + lamba


def test_low_confidence_ignored():
    assert run([([520, 385, 524, 389], 7, 0.3)], [CAR]) == []


def test_C_far_unlabeled_vehicle_added_near_or_labeled_skipped():
    assert run([([900, 370, 920, 382], 0, 0.7)], []) == [(0, "C")]                  # 20 px, etiketsiz
    assert run([([900, 370, 980, 420], 0, 0.9)], []) == []                          # 80 px: yakın, eklenmez
    assert run([([501, 351, 559, 399], 0, 0.9)], [CAR]) == []                       # zaten etiketli


def test_rule_a_off_by_default_behaviour():
    # rule_a kapalıyken kutu içi ışık eklenmez (park halindeki araç yansımaları)
    assert run([([520, 385, 524, 389], 7, 0.9)], [CAR], rule_a=False) == []


def test_B_dim_pair_rejected():
    pair = [([800, 380, 803, 383], 6, 0.9), ([812, 380, 815, 383], 6, 0.9)]
    assert run(pair, [], peaks=[255, 255]) == [(6, "B"), (6, "B")]
    assert run(pair, [], peaks=[150, 150]) == []          # sönük: reflektör / yol çivisi
    assert run(pair, [], peaks=[255, 150]) == []          # eşi sönük: çift sayılmaz