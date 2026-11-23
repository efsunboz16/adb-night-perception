import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adb.geometry import load_config  # noqa: E402
from adb.safety import SafetyMonitor  # noqa: E402

CFG = load_config()
DARK = np.full((72, 128), 30, np.uint8)


def mon(**over):
    cfg = copy.deepcopy(CFG)
    cfg["safety_monitor"].update(over)
    return SafetyMonitor(cfg, recover_s=1.0)


def test_normal_frames_allowed():
    m = mon()
    assert all(m.frame(i * 0.033, DARK) for i in range(30))


def test_frame_gap_triggers_and_recovers_after_delay():
    m = mon(max_frame_gap_ms=200)
    m.frame(0.0, DARK)
    assert not m.frame(0.5, DARK)                    # 500 ms boşluk
    assert "kare gecikmesi" in m.reasons
    ok = {round(t, 1): m.frame(t, DARK) for t in np.arange(0.6, 1.75, 0.1)}   # düzenli kareler
    assert not ok[0.6] and not ok[1.4]               # toparlanma süresi (1 sn) dolmadı
    assert ok[1.5] and ok[1.7]


def test_exact_gap_limit_is_ok():
    m = mon(max_frame_gap_ms=200)
    assert all(m.frame(i * 0.2, DARK) for i in range(10))     # 5 fps: tam sınırda, arıza değil


def test_watchdog_detects_missing_frames():
    m = mon(max_frame_gap_ms=200)
    m.frame(0.0, DARK)
    assert m.watchdog(0.1)
    assert not m.watchdog(0.3)                       # kare gelmedi


def test_overexposed_frame_disables_high_beam():
    m = mon(max_mean_brightness=0.85)
    assert not m.frame(0.0, np.full((72, 128), 240, np.uint8))
    assert "aşırı parlak" in m.reasons


def test_blur_check_disabled_by_default_enabled_when_set():
    flat = np.full((72, 128), 30, np.uint8)          # hiç detay yok: Laplace varyansı 0
    assert mon().frame(0.0, flat)                    # varsayılan: kapalı (ADR-018)
    assert not mon(blur_laplacian_min=5.0).frame(0.0, flat)


def test_apply_forces_all_segments_dimmed():
    mask = np.zeros(16, bool)
    assert SafetyMonitor.apply(mask, True).sum() == 0
    assert SafetyMonitor.apply(mask, False).all()