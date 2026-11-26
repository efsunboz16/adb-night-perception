# deploy/cpp

C++ çıkarım hattı: **gelecek çalışma** ([ADR-021](../../docs/decisions.md)). C++'ın kazandıracağı işler (GPU'da ön/son işleme, 8 bit aktarım, CUDA Graph) Python'da `adb/trt_runtime.py` ile yapıldı. C++ portu için gerekenler: TensorRT 10 C++ API, iki motor, takip / far kararı / güvenlik izleyici portu ve Python hattıyla maske karşılaştırması (altın test).
