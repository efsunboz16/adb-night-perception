"""ADB için hafif çok nesneli takipçi (ByteTrack mantığı, uzak/küçük nesnelere uyarlanmış).

Neden kendi takipçimiz:
  - Uzak araç ve ışık tespitleri çoğu zaman düşük güvenli (0,1–0,25). Standart ByteTrack yeni izi
    yalnızca yüksek güvenli tespitten başlatır; bu, dilimlemenin kazandırdığı uzak recall'ı siler.
    Burada her tespit (>= low) aday iz başlatabilir, ama iz ancak `min_hits` kez görülünce onaylanır.
    Tek karelik yanlış alarmlar böylece elenir.
  - 4–8 px'lik ışıklarda kareler arası küçük bir kayma bile IoU'yu sıfırlar. Eşleştirmede kutular
    en az `min_size` px'e büyütülür (merkez korunarak); küçük nesneler de örtüşebilir.
  - Bant iki karede bir işlenirse uzak araç bantsız karede görünmez. İz `max_coast` kare boyunca
    sabit hız tahminiyle sürdürülür ve çıktıda kalır (ADB'de segment kısık kalmaya devam eder).

Eşleştirme iki aşamalı (ByteTrack): önce yüksek güvenli tespitler, kalan izlere sonra düşük güvenliler.
Sınıf grupları ayrı eşleşir: araçlar araçlarla, ışıklar ışıklarla.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GROUPS = {0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2}   # araç / bisiklet+yaya / ışık


def _grow(b: np.ndarray, min_size: float) -> np.ndarray:
    """Kutuları merkez sabit, en az min_size genişlik/yükseklik olacak şekilde büyütür."""
    b = b.astype(float).copy()
    cx, cy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    w = np.maximum(b[:, 2] - b[:, 0], min_size) / 2
    h = np.maximum(b[:, 3] - b[:, 1], min_size) / 2
    return np.stack([cx - w, cy - h, cx + w, cy + h], 1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def greedy_match(sim: np.ndarray, thr: float):
    """Benzerliği yüksekten düşüğe açgözlü eşleştirme. (iz, tespit) çiftleri döner."""
    pairs = []
    if sim.size == 0:
        return pairs
    used_r, used_c = set(), set()
    for k in np.argsort(-sim, axis=None):
        r, c = divmod(int(k), sim.shape[1])
        if sim[r, c] < thr:
            break
        if r in used_r or c in used_c:
            continue
        used_r.add(r), used_c.add(c)
        pairs.append((r, c))
    return pairs


@dataclass
class Track:
    tid: int
    box: np.ndarray
    cls: int
    score: float
    hits: int = 1
    age: int = 0                 # son eşleşmeden beri geçen kare
    det_conf: float = 0.0        # son eşleşen tespitin kendi güveni (çıktı skoru budur, şişirilmez)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(4))

    def predict(self):
        self.box = self.box + self.vel
        self.age += 1

    def update(self, box, cls, score, vel_alpha):
        age = max(self.age, 1)
        last = self.box - self.vel * age         # son gerçek (eşleşmiş) konum
        v = (box - last) / age                   # birkaç kare atlandıysa kare başına yer değiştirme
        self.vel = v if self.hits == 1 else vel_alpha * v + (1 - vel_alpha) * self.vel
        self.box, self.cls, self.age = box.astype(float), cls, 0
        self.score = max(score, 0.7 * self.score + 0.3 * score)
        self.det_conf = float(score)
        self.hits += 1


class Tracker:
    def __init__(self, high=0.35, low=0.10, match_iou=0.2, min_size=16.0, min_hits=2,
                 max_coast=3, cand_max_age=1, vel_alpha=0.5, coast_out_max_w=32.0, coast_out_frames=1):
        self.high, self.low, self.match_iou = high, low, match_iou
        self.min_size, self.min_hits, self.max_coast, self.vel_alpha = min_size, min_hits, max_coast, vel_alpha
        self.cand_max_age = cand_max_age   # aday iz bu kadar kare görünmeden bekleyebilir (bant iki karede birse 1 gerekir)
        # Görünmeyen iz çıktıya yalnızca küçükse (uzak, bant kaynaklı) ve kısa süre verilir. Yakın nesne her karede
        # tam görüntüde zaten görünür; onu tahminle sürdürmek yalnızca yanlış yerde hayalet kutu üretir.
        self.coast_out_max_w, self.coast_out_frames = coast_out_max_w, coast_out_frames
        self.tracks: list[Track] = []
        self._next = 1

    def _associate(self, tracks, boxes, cls, conf):
        if not tracks or len(boxes) == 0:
            return [], list(range(len(tracks))), list(range(len(boxes)))
        tb = _grow(np.array([t.box for t in tracks]), self.min_size)
        db = _grow(boxes, self.min_size)
        sim = iou_matrix(tb, db)
        tg = np.array([GROUPS.get(t.cls, 3) for t in tracks])
        dg = np.array([GROUPS.get(int(c), 3) for c in cls])
        sim[tg[:, None] != dg[None, :]] = 0
        pairs = greedy_match(sim, self.match_iou)
        mt = {p[0] for p in pairs}
        md = {p[1] for p in pairs}
        return pairs, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(boxes)) if j not in md]

    def update(self, xyxy: np.ndarray, cls: np.ndarray, conf: np.ndarray):
        """Bir karenin tespitleriyle güncelle. Çıktı: (xyxy, cls, skor, iz_id) onaylı ve aktif izler."""
        xyxy = np.asarray(xyxy, float).reshape(-1, 4)
        cls, conf = np.asarray(cls, int), np.asarray(conf, float)
        keep = conf >= self.low
        xyxy, cls, conf = xyxy[keep], cls[keep], conf[keep]
        for t in self.tracks:
            t.predict()

        hi = np.where(conf >= self.high)[0]
        lo = np.where(conf < self.high)[0]
        pairs, t_left, d_left = self._associate(self.tracks, xyxy[hi], cls[hi], conf[hi])
        for ti, dj in pairs:
            j = hi[dj]
            self.tracks[ti].update(xyxy[j], cls[j], conf[j], self.vel_alpha)
        hi_left = [hi[d] for d in d_left]

        rest = [self.tracks[i] for i in t_left]
        pairs2, _, d_left2 = self._associate(rest, xyxy[lo], cls[lo], conf[lo])
        for ti, dj in pairs2:
            j = lo[dj]
            rest[ti].update(xyxy[j], cls[j], conf[j], self.vel_alpha)
        lo_left = [lo[d] for d in d_left2]

        for j in hi_left + lo_left:                       # eşleşmeyen her tespit aday iz başlatır
            self.tracks.append(Track(self._next, xyxy[j].copy(), int(cls[j]), float(conf[j]), det_conf=float(conf[j])))
            self._next += 1

        # ölü izler: onaylı olanlar max_coast, adaylar cand_max_age kare sonra düşer
        self.tracks = [t for t in self.tracks
                       if t.age <= (self.max_coast if t.hits >= self.min_hits else self.cand_max_age)]
        out = [t for t in self.tracks if t.hits >= self.min_hits and (
            t.age == 0 or (t.age <= self.coast_out_frames and t.box[2] - t.box[0] < self.coast_out_max_w))]
        if not out:
            return np.zeros((0, 4)), np.zeros(0, int), np.zeros(0), np.zeros(0, int)
        return (np.array([t.box for t in out]), np.array([t.cls for t in out]),
                np.array([t.det_conf * 0.9 ** t.age for t in out]), np.array([t.tid for t in out]))