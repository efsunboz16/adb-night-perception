"""Far kararı: tespitlerden 16 segmentlik kısma maskesine.

Adımlar (her kare):
  1. Güven eşiğini geçen araç ve ışık tespitleri alınır.
  2. Kutunun sol/sağ kenarı kameradan açıya çevrilir. Araçlarda mesafe kutu genişliğinden
     (fx * 1,8 m / genişlik), ışıklarda varsayılan uzak mesafe kullanılır.
  3. Her far için paralaks düzeltmesi yapılır (far kameradan yana ve öne kaymış). İki farın
     maskeleri birleştirilir (OR): bir segment iki farın birinde bile gerekiyorsa kısılır.
  4. Histerezis: segment istek gelir gelmez kısılır, ama istek bittikten sonra ancak
     release_delay_s saniye boyunca yeni istek gelmezse açılır. Tek karelik tespit kayıpları
     farın yanıp sönmesine (titreme) yol açmaz.

Sistem metrikleri (eval_video.py):
  kamaşma ihlali   : gerçek bir araç, açık (kısılmamış) bir segmentin içinde mi
  uzun far kullanımı: hiçbir gerçek nesnenin olmadığı segmentlerin ne kadarı açık
  titreme          : dakikadaki segment durum değişimi
"""

from __future__ import annotations

import numpy as np

from adb.geometry import Camera, camera_to_lamp_azimuth, segment_edges, segments_to_dim

VEH = (0, 1, 2, 3)
LIGHTS = (6, 7)
CAR_W_M = 1.8


def box_lamp_intervals(box, cls, cam: Camera, lamps: dict, light_dist_m=150.0, min_dist_m=5.0):
    """Kutunun her far için açı aralığı listesi [(sol, sağ), ...] (derece, paralaks düzeltmeli)."""
    x1, x2 = float(box[0]), float(box[2])
    w = max(x2 - x1, 1.0)
    dist = max(cam.fx_px * CAR_W_M / w, min_dist_m) if int(cls) in VEH else light_dist_m
    a1, a2 = cam.px_to_azimuth(x1), cam.px_to_azimuth(x2)
    return [(camera_to_lamp_azimuth(a1, dist, l["x_m"], l["z_m"]),
             camera_to_lamp_azimuth(a2, dist, l["x_m"], l["z_m"])) for l in lamps.values()]


def raw_mask(xyxy, cls, cam, edges, lamps, margin_deg, classes=VEH + LIGHTS):
    iv = []
    for b, c in zip(xyxy, cls):
        if int(c) in classes:
            iv += box_lamp_intervals(b, c, cam, lamps)
    return np.array(segments_to_dim(iv, edges, margin_deg), bool)


class BeamController:
    def __init__(self, cfg: dict, conf_thr: float, release_delay_s: float | None = None):
        self.cam = Camera.from_config(cfg)
        h, d = cfg["headlamp"], cfg["decision"]
        self.edges = segment_edges(h["num_segments"], tuple(h["coverage_deg"]))
        self.lamps = h["lamps"]
        self.margin = d["margin_deg"]
        self.release = d["release_delay_s"] if release_delay_s is None else release_delay_s
        self.conf_thr = conf_thr
        self.last_req = np.full(len(self.edges), -np.inf)

    def step(self, xyxy, cls, conf, t_s: float) -> np.ndarray:
        """True = kıs. t_s: karenin zamanı (saniye)."""
        k = np.asarray(conf) >= self.conf_thr
        req = raw_mask(np.asarray(xyxy).reshape(-1, 4)[k], np.asarray(cls)[k],
                       self.cam, self.edges, self.lamps, self.margin)
        self.last_req[req] = t_s
        return (t_s - self.last_req) <= self.release + 1e-9


def glare_and_utilization(mask, gt_veh, gt_all, cam, edges, lamps):
    """Etiketli karede sistem metrikleri.
    Döner: (ihlal [bool], far kapsamında mı [bool], boş segmentlerden açık olan sayısı, boş segment sayısı).
    Far kapsamı dışındaki araç (ör. yan şeritte çok yakın, ±16° dışı) hiç kamaştırılamaz; oran
    yalnızca kapsam içindeki araçlar üzerinden hesaplanmalı."""
    viol, inb = [], []
    for b in gt_veh:
        iv = box_lamp_intervals(b, 0, cam, lamps)
        need = np.array(segments_to_dim(iv, edges, 0.0), bool)       # pay yok: aracın gerçekten bulunduğu segmentler
        inb.append(bool(need.any()))
        viol.append(bool((need & ~mask).any()))
    occupied = raw_mask(gt_all, np.zeros(len(gt_all), int), cam, edges, lamps, 0.0, classes=(0,))
    free = ~occupied
    return np.array(viol, bool), np.array(inb, bool), int((free & ~mask).sum()), int(free.sum())