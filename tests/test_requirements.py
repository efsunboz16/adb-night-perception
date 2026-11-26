"""Faz 6 · Gereksinim ve hata enjeksiyonu testleri (docs/verification.md, docs/fmea.md).

Tek tek modül testlerinden farklı olarak burada hat uçtan uca (takip -> far kararı -> güvenlik
izleyici) ve 30 fps zaman adımıyla çalıştırılır. Her test bir gereksinime veya bir FMEA satırına
karşılık gelir; adı ve docstring'i onu belirtir.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adb.decision import BeamController  # noqa: E402
from adb.geometry import load_config  # noqa: E402
from adb.safety import SafetyMonitor  # noqa: E402
from adb.tracking import Tracker  # noqa: E402

CFG = load_config()
FPS = float(CFG["camera"]["fps"])
DT = 1.0 / FPS
CAR = np.array([[620., 340, 660, 370]])          # ortada, ~50 m
EMPTY = (np.zeros((0, 4)), np.zeros(0, int), np.zeros(0))


class Hat:
    """Sistemin karar kısmı, sistem yapılandırmasıyla: takip -> far kararı -> güvenlik izleyici."""

    def __init__(self):
        self.tr = Tracker()
        self.bc = BeamController(CFG, CFG["decision"]["min_confidence"])
        self.sm = SafetyMonitor(CFG)

    def step(self, dets, t, mean01=0.1):
        out = self.tr.update(*dets)
        mask = self.bc.step(*out[:3], t)
        return SafetyMonitor.apply(mask, self.sm.frame(t, mean01=mean01))


def car(conf=0.9, box=CAR):
    return box, np.array([0]), np.array([conf])


# ---------------------------------------------------------------- REQ-P-03 kısma tepki süresi
def test_req_p03_new_vehicle_dimmed_on_second_frame():
    """Araç ilk görüldüğü karede aday iz; ikinci karede onaylanır ve segment kısılır.
    Tepki = (min_hits - 1) kare + hat gecikmesi. 30 fps'te 1 kare = 33 ms (hedef ≤ 200 ms)."""
    h = Hat()
    assert not h.step(EMPTY, 0.0).any()
    assert not h.step(car(), DT).any()                     # 1. görüş: tek karelik yanlış alarm süzgeci
    m = h.step(car(), 2 * DT)                              # 2. görüş: kısıldı
    assert m.any() and not m.all()
    assert h.tr.min_hits == 2


# ---------------------------------------------------------------- REQ-F-07 / FMEA: kısa tespit kaybı
def test_detection_dropout_keeps_segment_dimmed_for_release_time():
    """Araç tespiti kesilirse segment hemen açılmaz: histerezis (1,0 sn) boyunca kısık kalır."""
    h = Hat()
    t = 0.0
    for _ in range(5):
        dim = h.step(car(), t)
        t += DT
    assert dim.any()
    release = CFG["decision"]["release_delay_s"]
    while t < 4 * DT + release - 2 * DT:                   # süre dolmadan: kısık
        assert (h.step(EMPTY, t) & dim).sum() == dim.sum()
        t += DT
    for _ in range(int(0.2 * FPS)):                        # süre dolduktan sonra: açık
        t += DT
        last = h.step(EMPTY, t)
    assert not last.any()


# ---------------------------------------------------------------- REQ-S-01 / FMEA: kare kaybı
def test_req_s01_camera_freeze_dims_all_and_recovers():
    """Kamera 200 ms'den uzun kare vermezse (bekçi) tüm segmentler kısılır; kareler dönünce
    toparlanma süresi (1 sn) dolmadan uzun far açılmaz."""
    h = Hat()
    for i in range(3):
        h.step(EMPTY, i * DT)
    last = 2 * DT
    assert h.sm.watchdog(last + 0.15)                      # 150 ms: sorun yok
    assert not h.sm.watchdog(last + 0.25)                  # 250 ms: kısa fara dön
    t = last + 0.30
    assert h.step(EMPTY, t).all()                          # kare geldi ama toparlanma süresi içinde
    while t < last + 0.25 + CFG["decision"]["release_delay_s"] + DT:
        t += DT
        m = h.step(EMPTY, t)
    assert not m.any()


# ---------------------------------------------------------------- REQ-S-02 / FMEA: aşırı parlaklık
def test_req_s02_overexposure_dims_all():
    h = Hat()
    h.step(EMPTY, 0.0)
    assert h.step(EMPTY, DT, mean01=0.95).all()


# ---------------------------------------------------------------- FMEA: tek karelik yanlış alarm
def test_single_frame_false_alarm_does_not_dim():
    """Tek karelik parlak leke (yansıma) farı kısmaz: takip onayı iki görüş ister."""
    h = Hat()
    h.step(EMPTY, 0.0)
    assert not h.step(car(0.8), DT).any()
    assert not h.step(EMPTY, 2 * DT).any()
    assert not h.step(EMPTY, 3 * DT).any()


# ---------------------------------------------------------------- FMEA: girdi bozuklukları
def test_degenerate_detections_do_not_crash_or_open_beam():
    """Sıfır genişlikli, görüntü dışına taşan veya NaN olmayan ama uç değerli kutular hattı
    bozmaz; geçerli araç yine kısılır."""
    h = Hat()
    boxes = np.array([[640., 350, 640, 350], [-50, 300, 30, 340], [1250, 300, 1400, 360], [620, 340, 660, 370]])
    dets = (boxes, np.array([0, 0, 0, 0]), np.array([0.9, 0.9, 0.9, 0.9]))
    h.step(dets, 0.0)
    m = h.step(dets, DT)
    assert m[7] or m[8]


def test_low_confidence_vehicle_ignored_by_design():
    """REQ-S-03'ten bilinçli sapma (ADR-017): far kararı eşiğinin (0,35) altındaki tespit kısma
    istemez. Bu test sapmayı belgeler; eşik değişirse test de değişmeli."""
    h = Hat()
    for i in range(5):
        m = h.step(car(0.2), i * DT)
    assert not m.any()
