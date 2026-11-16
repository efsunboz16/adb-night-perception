"""Kamera ve far geometrisi: piksel <-> açı dönüşümü ve segment maskesi.

Koordinatlar: x sağa +, z ileri +. Açılar derece, sağ +.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Camera:
    width_px: int
    height_px: int
    fx_px: float
    cx_px: float

    @classmethod
    def from_config(cls, cfg: dict) -> "Camera":
        c = cfg["camera"]
        fx = c.get("fx_px") or (c["width_px"] / 2) / math.tan(math.radians(c["hfov_deg"] / 2))
        cx = c.get("cx_px") or c["width_px"] / 2
        return cls(c["width_px"], c["height_px"], float(fx), float(cx))

    def px_to_azimuth(self, u: float) -> float:
        """Görüntüdeki yatay piksel konumunu kameradan bakış açısına çevirir."""
        return math.degrees(math.atan((u - self.cx_px) / self.fx_px))

    def object_width_px(self, width_m: float, distance_m: float) -> float:
        """Belirli mesafedeki nesnenin görüntüde kaç piksel genişlikte göründüğü."""
        return self.fx_px * width_m / distance_m


def camera_to_lamp_azimuth(az_cam_deg: float, distance_m: float, lamp_x_m: float, lamp_z_m: float) -> float:
    """Kameradan görülen açıyı farın baktığı açıya çevirir (paralaks düzeltmesi).

    Far kameradan yana ve ileri kaymış olduğu için yakın nesnelerde açı farkı büyür.
    """
    x = distance_m * math.tan(math.radians(az_cam_deg))
    return math.degrees(math.atan2(x - lamp_x_m, distance_m - lamp_z_m))


def segment_edges(num_segments: int, coverage_deg: tuple[float, float]) -> list[tuple[float, float]]:
    lo, hi = coverage_deg
    step = (hi - lo) / num_segments
    return [(lo + i * step, lo + (i + 1) * step) for i in range(num_segments)]


def segments_to_dim(intervals_deg: list[tuple[float, float]], edges: list[tuple[float, float]],
                    margin_deg: float) -> list[bool]:
    """Her segment için True = kıs. Pay eklenmiş herhangi bir açı aralığıyla kesişen segment kısılır."""
    dim = [False] * len(edges)
    for a, b in intervals_deg:
        a, b = min(a, b) - margin_deg, max(a, b) + margin_deg
        for i, (s0, s1) in enumerate(edges):
            if a < s1 and b > s0:
                dim[i] = True
    return dim


def load_config(path: str | Path = Path(__file__).resolve().parents[1] / "configs" / "system.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())
