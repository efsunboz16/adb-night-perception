"""Güvenlik izleyici (REQ-S-01, REQ-S-02): algılamaya güvenilemeyen durumda tüm uzun farı kapatır.

Algılama hattı ne kadar iyi olursa olsun, girdisi bozuksa çıktısı da bozuktur. İzleyici far
kararından bağımsız çalışır ve şu durumlarda kısa fara döner (tüm segmentler kısık):

  kare gecikmesi : son kareden bu yana max_frame_gap_ms'den fazla geçti (kamera/bağlantı arızası,
                   işlemci tıkanması). Gerçek sistemde yeni kare gelmediği için bu, kare beklemeden
                   bir bekçi (watchdog) ile kontrol edilir: watchdog(şimdi).
  aşırı parlak   : ortalama parlaklık max_mean_brightness'ten büyük (pozlama bozuk, kamera
                   karşıdan gelen ışıkla doymuş, gündüz/aydınlık alan).
  bulanık        : Laplace varyansı blur_laplacian_min'den küçük. Varsayılan KAPALI (null): basit
                   ölçülerle güvenilir eşik bulunamadı (Deney 8, ADR-018).

Arıza geçtikten sonra uzun far hemen dönmez: recover_s saniye boyunca arızasız kare gelmesi gerekir.
Arıza sınırında gidip gelen bir durum farın yanıp sönmesine yol açmasın diye.
"""

from __future__ import annotations

import numpy as np


class SafetyMonitor:
    def __init__(self, cfg: dict, recover_s: float | None = None):
        s = cfg["safety_monitor"]
        self.max_gap_s = s["max_frame_gap_ms"] / 1000.0
        self.max_bright = s.get("max_mean_brightness")
        self.blur_min = s.get("blur_laplacian_min")
        self.recover_s = cfg["decision"]["release_delay_s"] if recover_s is None else recover_s
        self.last_frame_t = None
        self.fault_until = -np.inf
        self.reasons: list[str] = []

    def _fault(self, t_s, reason):
        self.fault_until = max(self.fault_until, t_s + self.recover_s)
        if reason not in self.reasons:
            self.reasons.append(reason)

    def frame(self, t_s: float, gray: np.ndarray | None = None, mean01: float | None = None) -> bool:
        """Yeni kare geldi. gray: gri görüntü (uint8) veya yalnızca ortalama parlaklık (0–1).
        Döner: uzun fara izin var mı."""
        self.reasons = []
        if self.last_frame_t is not None and t_s - self.last_frame_t > self.max_gap_s + 1e-6:
            self._fault(t_s, "kare gecikmesi")
        self.last_frame_t = t_s
        if mean01 is None and gray is not None:
            mean01 = float(gray.mean()) / 255.0
        if self.max_bright is not None and mean01 is not None and mean01 > self.max_bright:
            self._fault(t_s, "aşırı parlak")
        if self.blur_min is not None and gray is not None:
            import cv2
            if float(cv2.Laplacian(gray, cv2.CV_32F).var()) < self.blur_min:
                self._fault(t_s, "bulanık")
        return t_s >= self.fault_until - 1e-6

    def watchdog(self, now_s: float) -> bool:
        """Kare beklemeden kontrol (gerçek zamanlı döngüde zamanlayıcıdan çağrılır)."""
        if self.last_frame_t is not None and now_s - self.last_frame_t > self.max_gap_s + 1e-6:
            self._fault(now_s, "kare gecikmesi")
        return now_s >= self.fault_until - 1e-6

    @staticmethod
    def apply(mask: np.ndarray, ok: bool) -> np.ndarray:
        """İzin yoksa tüm segmentler kısık (True = kıs)."""
        return mask if ok else np.ones_like(mask, dtype=bool)