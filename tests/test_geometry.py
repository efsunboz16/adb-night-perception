import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adb.geometry import (Camera, camera_to_lamp_azimuth, load_config,  # noqa: E402
                          segment_edges, segments_to_dim)


def cam():
    return Camera.from_config(load_config())


def test_center_pixel_is_zero_degrees():
    c = cam()
    assert abs(c.px_to_azimuth(c.cx_px)) < 1e-9


def test_image_edge_is_half_fov():
    c = cam()
    assert math.isclose(c.px_to_azimuth(c.width_px), 30.0, abs_tol=1e-6)


def test_parallax_vanishes_far_away():
    near = camera_to_lamp_azimuth(5.0, 15, -0.7, 1.8) - 5.0
    far = camera_to_lamp_azimuth(5.0, 400, -0.7, 1.8) - 5.0
    assert abs(near) > 2.0 and abs(far) < 0.2


def test_segments_cover_range():
    e = segment_edges(16, (-16, 16))
    assert len(e) == 16 and e[0][0] == -16 and e[-1][1] == 16


def test_vehicle_dims_only_its_segments():
    e = segment_edges(16, (-16, 16))
    dim = segments_to_dim([(4.2, 5.0)], e, margin_deg=1.0)  # 3.2..6.0 -> segment 9 (2..4°), 10 (4..6°)
    assert [i for i, d in enumerate(dim) if d] == [9, 10]
    dim = segments_to_dim([(4.2, 5.1)], e, margin_deg=1.0)  # 6.1 sınırı aşar -> 11 de kısılır
    assert [i for i, d in enumerate(dim) if d] == [9, 10, 11]


def test_no_vehicles_all_on():
    assert not any(segments_to_dim([], segment_edges(16, (-16, 16)), 1.0))
