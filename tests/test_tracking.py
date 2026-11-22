import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adb.tracking import Tracker  # noqa: E402


def step(tr, dets):
    b = np.array([d[0] for d in dets], float).reshape(-1, 4)
    c = np.array([d[1] for d in dets], int)
    s = np.array([d[2] for d in dets], float)
    return tr.update(b, c, s)


def test_single_frame_false_alarm_never_output():
    tr = Tracker()
    out = step(tr, [([100, 100, 110, 106], 6, 0.9)])
    assert len(out[0]) == 0                         # ilk görüşte onaysız
    out = step(tr, [])
    assert len(out[0]) == 0                         # bir daha görülmedi: hiç çıkmadı


def test_low_conf_far_light_confirmed_after_two_hits():
    tr = Tracker(high=0.35, low=0.10)
    step(tr, [([600, 300, 604, 303], 6, 0.15)])
    out = step(tr, [([601, 300, 605, 303], 6, 0.15)])
    assert len(out[0]) == 1 and out[1][0] == 6      # düşük güvenli uzak ışık da iz olur


def test_tiny_boxes_associate_despite_zero_iou():
    tr = Tracker()
    step(tr, [([600, 300, 604, 303], 6, 0.5)])
    out = step(tr, [([606, 300, 610, 303], 6, 0.5)])   # 6 px kaydı: ham IoU = 0
    assert len(out[0]) == 1 and len(tr.tracks) == 1


def test_coasting_small_box_output_one_frame():
    tr = Tracker(max_coast=2)
    for x in (100, 110):
        step(tr, [([x, 100, x + 20, 110], 0, 0.6)])
    out = step(tr, [])                               # bantsız kare: küçük (uzak) iz 1 kare çıktıda kalır
    assert len(out[0]) == 1
    assert np.isclose(out[0][0][0], 120)             # sabit hızla ilerletildi
    assert len(step(tr, [])[0]) == 0                 # 2. görünmeyen kare: çıktıda değil
    assert len(tr.tracks) == 1                       # ama iz hâlâ bellekte (max_coast)


def test_large_box_not_coasted_in_output():
    tr = Tracker()
    for _ in range(2):
        step(tr, [([100, 100, 300, 200], 0, 0.9)])
    assert len(step(tr, [])[0]) == 0                 # yakın araç: tahmini hayalet kutu verilmez


def test_output_score_not_inflated():
    tr = Tracker()
    step(tr, [([100, 100, 120, 110], 0, 0.9)])
    out = step(tr, [([100, 100, 120, 110], 0, 0.15)])
    assert np.isclose(out[2][0], 0.15)               # eski yüksek güven taşınmaz


def test_groups_do_not_mix():
    tr = Tracker()
    step(tr, [([100, 100, 140, 130], 0, 0.9)])
    step(tr, [([100, 100, 140, 130], 0, 0.9), ([110, 120, 114, 123], 6, 0.9)])   # ışık araç kutusunun içinde
    assert sorted(t.cls for t in tr.tracks) == [0, 6]   # ışık araç izine bağlanmadı, ayrı aday iz


def test_velocity_after_gap():
    tr = Tracker(max_coast=3)
    step(tr, [([0, 0, 10, 10], 0, 0.9)])
    step(tr, [([10, 0, 20, 10], 0, 0.9)])
    step(tr, [])
    out = step(tr, [([30, 0, 40, 10], 0, 0.9)])
    assert np.allclose(tr.tracks[0].vel[0], 10)      # 2 karede 20 px -> kare başına 10
    assert len(out[0]) == 1


def test_every_other_frame_detection_confirmed_and_kept():
    # bant iki karede bir: uzak araç yalnızca çift karelerde tespit ediliyor
    tr = Tracker()
    far = ([900, 350, 910, 356], 0, 0.3)
    outs = [len(step(tr, [far] if i % 2 == 0 else [])[0]) for i in range(7)]
    assert outs[:2] == [0, 0]                 # ilk görüş + boşluk: henüz onaysız
    assert all(o == 1 for o in outs[2:])      # 2. görüşte onaylandı, bantsız karelerde de çıktıda