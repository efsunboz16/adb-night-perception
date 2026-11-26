# deploy · Faz 5: TensorRT ve gerçek zamanlı çıkarım

Hedefler: REQ-P-05 (≥ 25 FPS) ve REQ-P-06 (uçtan uca ≤ 50 ms, p95). Kararlar: [ADR-019](../docs/decisions.md). Sonuçlar: [Deney 9](../docs/results.md).

```
deploy/
├── export.py         .pt -> ONNX -> TensorRT motorları (tam kare + bant, FP16 / INT8)
├── build_engine.py   ONNX -> motor; INT8 kalibrasyonu, Ultralytics meta verisi
├── models/           (repoda yok) .onnx, .engine, kalibrasyon önbelleği, manifest.json
└── cpp/              C++ çıkarım hattı (Faz 5.5)
```

## 1. Ortam (Windows, RTX 5050)

| Bileşen | Sürüm | Not |
|---|---|---|
| NVIDIA sürücüsü | ≥ 570 | `nvidia-smi` |
| CUDA Toolkit | 12.8 | C++ ve `trtexec` için; kurulumda sürücü bileşenini kaldır |
| TensorRT zip | 10.16.1, "CUDA 12.0 to 12.9" | developer.nvidia.com/tensorrt (ücretsiz hesap). 11.x veya "CUDA 13" paketi değil |
| `tensorrt-cu12` (pip) | zip ile **aynı** sürüm | `pip install "tensorrt-cu12==10.16.1.*"` |
| PyTorch | cu128 | RTX 50 serisi için gerekli (ADR-002) |

**PATH:** zip'teki DLL'ler ve `trtexec.exe` `bin` klasöründe (`lib` değil). PATH'e klasör yazılır, dosya değil. En sorunsuz yol PowerShell profili:

```powershell
$prof = $PROFILE.CurrentUserAllHosts
if (-not (Test-Path $prof)) { New-Item $prof -Force | Out-Null }
Add-Content $prof '$env:Path = "C:\TensorRT\TensorRT-10.16.1.11\bin;" + $env:Path'
```

`.venv\Scripts\Activate.ps1`'e ekleme: dosyanın sonundaki imza bloğundan sonra gelen satır betiği bozar.

Kontrol:
```powershell
trtexec --help | Select-Object -First 2
python -c "import tensorrt as trt, torch; print(trt.__version__, torch.version.cuda, torch.cuda.get_device_capability())"   # 10.16.1... 12.8 (12, 0)
```

## 2. Motorları üret

```powershell
python deploy/export.py                                   # FP16: full_fp16.engine, band_fp16.engine
python deploy/export.py --precision int8 --val            # INT8 + mAP için kare motorlar
python deploy/export.py --precision int8 --fp16-layers /model.23/   # INT8, tespit kafası FP16
```

Motor dosyası derlendiği GPU'ya bağlıdır. Başka bir cihazda (ör. Jetson) `.onnx`'ten yeniden derlenir. INT8 kalibrasyonu train bölümünden 500 gece karesiyle yapılır; sonuç `*_int8_calib.cache`'e yazılır ve sonraki derlemelerde kullanılır. Kalibrasyonu yenilemek için önbelleği sil.

## 3. Doğruluk eşitliği (FP16 / INT8, PyTorch'a göre)

```powershell
# Uzak recall (REQ-F-01), dilimlemeli
python eval/eval_distance.py --weights runs/baseline/weights/best.pt --conds night --slice
python eval/eval_distance.py --weights deploy/models/full_fp16.engine --band-weights deploy/models/band_fp16.engine --conds night --slice
# Sistem metrikleri: kamaşma, kullanım, titreme
python eval/eval_video.py --weights deploy/models/full_fp16.engine --band-weights deploy/models/band_fp16.engine --scenes highway residential
# mAP (kare doğrulama motoru)
python eval/eval_conditions.py --weights deploy/models/val_fp16.engine
```

Motor sonuçları `runs/trt/<motor adı>/` altına yazılır; PyTorch sonuçlarının üzerine yazılmaz.

Kabul kuralı (ADR-020'de kesinleşecek): FP16'da gece mAP50 ve 166–249 m recall en fazla 0,5 puan, INT8'de recall en fazla 2 puan düşebilir; kamaşma ihlali Faz 4'ten kötü olmamalı.

## 4. Gecikme

Önce: prizde, Windows güç modu "En iyi performans", GPU kullanan uygulamalar kapalı.

```powershell
# Yalnızca motor (meta verisiz kopya gerekir: export.py --raw)
trtexec --loadEngine=deploy/models/full_fp16_raw.engine --warmUp=2000 --iterations=1000 --percentile=50,95,99 --useSpinWait --useCudaGraph
# Python hattının tamamı, aşama aşama; bant her karede ve iki karede bir
python eval/bench_latency.py --weights deploy/models/full_fp16.engine --band-weights deploy/models/band_fp16.engine
```

## Sonuçlar (Deney 9, RTX 5050, FP16, yalnızca motor)

| | p50 | p95 | p99 |
|---|---|---|---|
| Tam kare 736×1280, CUDA Graph | 3,2 ms | 3,4 ms | 3,9 ms |
| Bant 4 × 704×768 | 7,8 ms | 8,5 ms | 9,3 ms |

Veri aktarımı dahil.

**Doğruluk (Deney 10):** FP16 = PyTorch (220 m recall 0,848 / 0,850, sistem metrikleri aynı). Hassasiyet FP16 (ADR-020).

**Python hattı (Deney 11):** bant her karede p95 81 ms, iki karede bir 39 ms. Darboğaz motor değil, CPU'daki ön/son işleme. Sıradaki adım C++ hattı.
