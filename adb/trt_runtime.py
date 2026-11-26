"""GPU üzerinde ADB algılama hattı: TensorRT motorları Ultralytics'siz, ön ve son işleme GPU'da (ADR-021).

Deney 11'de Python hattında bant aşaması 36,7 ms sürdü, motorun kendisi 7,8 ms. Fark CPU'da:
parçaların kesilip büyütülmesi, Ultralytics'in her parça için yaptığı letterbox / tensör dönüşümü,
FP32 veri aktarımı ve son işleme. Bu modül aynı işi GPU'da yapar:

  1. Kare (uint8, 2,7 MB) GPU'ya bir kez gönderilir (sabitlenmiş bellek). FP32 yerine 8 bit: 4 kat az veri.
  2. Tam kare: kanal çevirme, /255, 736×1280 dolgu (Ultralytics LetterBox ile aynı yerleşim).
  3. Bant: 4 parça GPU'daki kareden kesilir, 2x büyütülür (cv2.INTER_LINEAR ile aynı formül).
  4. Motor CUDA Graph ile çalıştırılır (Deney 9: çekirdek başlatma dalgalanmasını kaldırır).
  5. Kutu çözümleme, güven eşiği ve sınıf bazında NMS GPU'da; CPU'ya yalnızca kalan kutular iner.

Çıktılar adb/slicing.py'deki predict_full / predict_band ile aynı biçimde (orijinal görüntü koordinatları):
değerlendirme scriptleri `--backend trt` ile bu hattı kullanır, geri kalan kod değişmez.

Ön ve son işleme fonksiyonları cihazdan bağımsızdır (CPU'da birim test edilir); yalnızca TrtEngine
CUDA ve TensorRT gerektirir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

PAD = 114 / 255.0
DEFAULT_CUDA_GRAPH = True
# Faz 1–5'teki tüm sonuçlar Ultralytics predict ile ölçüldü; orada NMS IoU 0,5 (predictor.args.iou,
# Ultralytics 8.4.215). 0,7 kullanılınca tam karede +%13 yakın kopya kutu çıktı (eval/compare_backends.py).
NMS_IOU = 0.5
MAX_DET = 300


# ---------------------------------------------------------------- ön işleme (cihazdan bağımsız)
def letterbox_params(h: int, w: int, H: int, W: int):
    """Ultralytics LetterBox (auto=False, center=True) yerleşimi: (oran, yeni_h, yeni_w, üst, sol)."""
    r = min(H / h, W / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    dh, dw = (H - nh) / 2, (W - nw) / 2
    return r, nh, nw, int(round(dh - 0.1)), int(round(dw - 0.1))


def preprocess_full(frame_hwc, out, letterbox):
    """frame_hwc: uint8 BGR (h, w, 3) tensör. out: (1, 3, H, W) float tensör, yerinde doldurulur."""
    import torch.nn.functional as F

    r, nh, nw, top, left = letterbox
    x = frame_hwc.permute(2, 0, 1).flip(0).float()               # BGR -> RGB, CHW
    if (nh, nw) != tuple(frame_hwc.shape[:2]):
        x = F.interpolate(x[None], size=(nh, nw), mode="bilinear", align_corners=False)[0].round()
    out.fill_(PAD)
    out[0, :, top:top + nh, left:left + nw] = x / 255.0
    return out


def preprocess_band(frame_hwc, out, y0: int, y1: int, xs: list[int], tile_w: int, scale: float):
    """Bant parçaları: (n, 3, H, W) float tensöre yerinde. cv2.resize(INTER_LINEAR) ile aynı:
    yarım piksel merkezli bilineer, sonuç uint8'e yuvarlanır (Ultralytics'e cv2 çıktısı giderdi)."""
    import torch
    import torch.nn.functional as F

    tiles = torch.stack([frame_hwc[y0:y1, x:x + tile_w] for x in xs])        # (n, h, w, 3) uint8
    t = tiles.permute(0, 3, 1, 2).float()
    t = F.interpolate(t, scale_factor=scale, mode="bilinear", align_corners=False).round()
    H, W = t.shape[2:]
    out.fill_(PAD)
    out[:, :, :H, :W] = t.flip(1) / 255.0
    return out


# ---------------------------------------------------------------- son işleme (cihazdan bağımsız)
def decode(pred, conf: float, iou: float = NMS_IOU, max_det: int = MAX_DET):
    """YOLO ONNX çıktısı (B, 4+nc, N): cx, cy, w, h, sınıf olasılıkları (sigmoid uygulanmış).
    Ultralytics non_max_suppression ile aynı mantık (tek etiket, sınıf bazında NMS).
    Döner: her batch elemanı için (xyxy, cls, conf) tensörleri, motor giriş koordinatlarında."""
    import torch
    from torchvision.ops import batched_nms

    out = []
    for p in pred:                                               # (4+nc, N)
        p = p.T
        score, cls = p[:, 4:].max(1)
        k = score > conf
        p, score, cls = p[k], score[k], cls[k]
        if len(p) == 0:
            out.append((p.new_zeros((0, 4)), cls, score))
            continue
        cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2] / 2, p[:, 3] / 2
        xyxy = torch.stack([cx - w, cy - h, cx + w, cy + h], 1)
        keep = batched_nms(xyxy.float(), score.float(), cls, iou)[:max_det]
        out.append((xyxy[keep], cls[keep], score[keep]))
    return out


def unletterbox(xyxy, letterbox, h: int, w: int):
    """Motor girişi koordinatlarından orijinal görüntüye (Ultralytics scale_boxes ile aynı)."""
    r, _, _, top, left = letterbox
    b = xyxy.clone()
    b[:, [0, 2]] = ((b[:, [0, 2]] - left) / r).clamp(0, w)
    b[:, [1, 3]] = ((b[:, [1, 3]] - top) / r).clamp(0, h)
    return b


def _np(t3):
    xyxy, cls, conf = t3
    return (xyxy.float().cpu().numpy(), cls.cpu().numpy().astype(int), conf.float().cpu().numpy())


# ---------------------------------------------------------------- TensorRT
class TrtEngine:
    """Sabit boyutlu bir TensorRT motoru, statik giriş/çıkış tamponları ve isteğe bağlı CUDA Graph."""

    def __init__(self, path: str | Path, cuda_graph: bool = True):
        import tensorrt as trt
        import torch

        from adb.models import engine_meta

        data = Path(path).read_bytes()
        meta = engine_meta(path)
        if meta is not None:                                     # Ultralytics başlığını atla
            n = int.from_bytes(data[:4], "little", signed=True)
            data = data[4 + n:]
        self.meta = meta or {}
        self.path = str(path)
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(data)
        if self.engine is None:
            raise SystemExit(f"{path}: motor açılamadı (farklı TensorRT sürümü veya farklı GPU'da mı derlendi?)")
        self.ctx = self.engine.create_execution_context()
        dt = {trt.float32: torch.float32, trt.float16: torch.float16}
        self.inp = self.out = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            t = torch.empty(shape, dtype=dt[self.engine.get_tensor_dtype(name)], device="cuda")
            self.ctx.set_tensor_address(name, t.data_ptr())
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inp = t
            else:
                self.out = t
        self.stream = torch.cuda.Stream()
        self.graph = None
        self._run()                                              # ilk çalıştırma: TensorRT hazırlığı
        torch.cuda.synchronize()
        if cuda_graph:
            try:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=self.stream):
                    self.ctx.execute_async_v3(self.stream.cuda_stream)
                self.graph = g
            except Exception as e:                               # CUDA Graph desteklenmezse normal çalış
                print(f"  uyarı: {Path(path).name} için CUDA Graph kurulamadı ({e}); normal çalıştırılıyor")
                self.graph = None
        torch.cuda.synchronize()

    @property
    def shape(self):
        b, _, h, w = self.inp.shape
        return b, h, w

    def _run(self):
        import torch

        self.stream.wait_stream(torch.cuda.current_stream())
        if self.graph is not None:
            with torch.cuda.stream(self.stream):
                self.graph.replay()
        else:
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        torch.cuda.current_stream().wait_stream(self.stream)

    def __call__(self):
        """self.inp doldurulduktan sonra çağrılır; self.out'u döner."""
        self._run()
        return self.out


class GpuPipeline:
    """Tam kare + bant motorları, ortak GPU karesiyle. adb/slicing.py predict_full / predict_band
    bu nesneyi tanır (adb_gpu_pipeline özniteliği) ve çıkarımı buraya devreder."""

    adb_gpu_pipeline = True

    def __init__(self, full_path: str, band_path: str | None, slicing_cfg: dict, img_hw=(720, 1280),
                 cuda_graph: bool | None = None):
        import torch

        from adb.slicing import tile_xs

        cuda_graph = DEFAULT_CUDA_GRAPH if cuda_graph is None else cuda_graph
        self.full = TrtEngine(full_path, cuda_graph)
        self.band = TrtEngine(band_path, cuda_graph) if band_path else None
        self.cfg = slicing_cfg
        self.h, self.w = img_hw
        _, H, W = self.full.shape
        self.lb = letterbox_params(self.h, self.w, H, W)
        self.y0, self.y1 = (int(round(v * self.h)) for v in slicing_cfg["band_frac"])
        self.xs = tile_xs(self.w, slicing_cfg["tile_w"], slicing_cfg["overlap"])
        if self.band is not None:
            from adb.models import band_input_shape

            (need_h, need_w), n = band_input_shape(self.h, self.w, slicing_cfg)
            b, bh, bw = self.band.shape
            if (b, bh, bw) != (n, need_h, need_w):
                raise SystemExit(f"Bant motoru {b}×{bh}×{bw}, gereken {n}×{need_h}×{need_w}. "
                                 f"deploy/export.py ile yeniden üret.")
        self.host = torch.empty((self.h, self.w, 3), dtype=torch.uint8).pin_memory()
        self.dev = torch.empty((self.h, self.w, 3), dtype=torch.uint8, device="cuda")
        self._img = None
        # Ultralytics Detector ile aynı arayüz (describe, imgsz) — scriptler yazdırırken kullanır
        self.path, self.engine, self.batch = full_path, True, 1
        self.imgsz = (H, W)

    def describe(self) -> str:
        b = f" + {Path(self.band.path).name}" if self.band else ""
        g = "CUDA Graph" if self.full.graph is not None else "CUDA Graph yok"
        return f"GPU hattı: {Path(self.full.path).name}{b} ({g})"

    def upload(self, img: np.ndarray):
        """Kareyi GPU'ya bir kez gönderir. Aynı nesne tekrar verilirse yeniden göndermez.
        Referansı tutuyoruz: nesne yaşadıkça id'si başka bir kareye geçemez."""
        if img is self._img:
            return self.dev
        if img.shape[:2] != (self.h, self.w):
            raise ValueError(f"Kare {img.shape[:2]}, hat {(self.h, self.w)} için kuruldu.")
        import torch

        torch.cuda.current_stream().synchronize()               # önceki aktarım bitmeden tampon değişmesin
        np.copyto(self.host.numpy(), img)
        self.dev.copy_(self.host, non_blocking=True)
        self._img = img
        return self.dev

    def mean01(self, img: np.ndarray) -> float:
        """Güvenlik izleyici için ortalama parlaklık, GPU'daki kareden (CPU'da tam kare ortalaması ~3 ms)."""
        return float(self.upload(img).float().mean()) / 255.0

    def predict_full(self, img: np.ndarray, conf: float):
        f = self.upload(img)
        preprocess_full(f, self.full.inp, self.lb)
        xyxy, cls, sc = decode(self.full(), conf)[0]
        return _np((unletterbox(xyxy, self.lb, self.h, self.w), cls, sc))

    def predict_band(self, img: np.ndarray, conf: float):
        from adb.slicing import drop_cut_boxes, map_tile_boxes

        if self.band is None:
            raise SystemExit("Bant motoru verilmedi (--band-weights).")
        f = self.upload(img)
        c = self.cfg
        preprocess_band(f, self.band.inp, self.y0, self.y1, self.xs, c["tile_w"], c["scale"])
        xyxy, cls, cf = [], [], []
        for (b, k, s), x in zip(decode(self.band(), conf), self.xs):
            if len(b) == 0:
                continue
            b, k, s = _np((b, k, s))
            b = map_tile_boxes(b, x, self.y0, c["scale"])
            keep = drop_cut_boxes(b, x, c["tile_w"], self.w)
            xyxy.append(b[keep]), cls.append(k[keep]), cf.append(s[keep])
        if not xyxy:
            return np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)
        return np.concatenate(xyxy), np.concatenate(cls), np.concatenate(cf)
