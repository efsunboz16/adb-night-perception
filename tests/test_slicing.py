import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adb.slicing import drop_cut_boxes, map_tile_boxes, merge, tile_xs  # noqa: E402


def test_tiles_cover_width_with_overlap():
    xs = tile_xs(1280, 384, 64)
    assert xs[0] == 0 and xs[-1] + 384 == 1280
    assert all(b - a <= 384 - 64 for a, b in zip(xs, xs[1:]))   # her komşu arası örtüşme >= 64


def test_tile_wider_than_image():
    assert tile_xs(300, 384, 64) == [0]


def test_map_back_to_original():
    # 2x büyütülmüş parçada (20,10)-(40,30) kutusu, parça (x0=320, y0=300) -> orijinalde (330,305)-(340,315)
    b = map_tile_boxes(np.array([[20., 10., 40., 30.]]), 320, 300, 2.0)
    assert np.allclose(b, [[330, 305, 340, 315]])


def test_cut_boxes_dropped_only_at_inner_edges():
    b = np.array([[320.5, 0, 330, 10],      # sol iç kenara değiyor -> at
                  [400, 0, 410, 10],        # ortada -> tut
                  [0.5, 0, 10, 10]])        # görüntü sol kenarı (x0=0 parçası değil) -> tut değil, test x0=320
    k = drop_cut_boxes(b, 320, 384, 1280)
    assert k.tolist() == [False, True, False]
    k0 = drop_cut_boxes(np.array([[0.5, 0, 10, 10]]), 0, 384, 1280)   # x0=0: görüntü kenarı, tutulur
    assert k0.tolist() == [True]


def test_merge_removes_duplicates_keeps_distinct():
    full = (np.array([[100., 100, 120, 110]]), np.array([0]), np.array([0.6]))
    band = (np.array([[101., 100, 121, 110], [500, 100, 508, 104]]), np.array([0, 6]), np.array([0.9, 0.4]))
    xyxy, cls, conf = merge(full, band)
    assert len(xyxy) == 2                                  # aynı araç tek kez, ayrı ışık korunur
    assert np.isclose(conf, 0.9).any() and not np.isclose(conf, 0.6).any()   # tekrarda yüksek güvenli olan kalır