"""Donanım ve yazılım ortamını kontrol eder (Faz 0 / p0c).

Kullanım: python scripts/check_env.py
Çıktıyı docs/decisions.md'deki donanım kararına ekle.
"""

import importlib
import platform
import shutil
import sys
from pathlib import Path


def ver(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "kurulu")
    except Exception:
        return None


def row(name, value, need=None):
    mark = "OK " if value else ("-- " if need is None else "EKSİK")
    print(f"  [{mark:5}] {name:<22} {value or ''}{'' if value or not need else '  -> ' + need}")


def main():
    print(f"Sistem: {platform.system()} {platform.machine()} | Python {sys.version.split()[0]}")
    tegra = Path("/etc/nv_tegra_release")
    print(f"Jetson: {'evet — ' + tegra.read_text().split(',')[0] if tegra.exists() else 'hayır'}\n")

    print("Eğitim:")
    torch_v = ver("torch")
    row("torch", torch_v, "pip install torch (CUDA sürümüyle)")
    if torch_v:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            row("CUDA GPU", f"{p.name}, {p.total_memory / 2**30:.1f} GB, CUDA {torch.version.cuda}")
        else:
            row("CUDA GPU", None, "GPU yok: eğitim için Colab/Kaggle veya araştırma merkezi")
    row("ultralytics", ver("ultralytics"), "pip install ultralytics")
    row("opencv", ver("cv2"), "pip install opencv-python")

    print("\nDağıtım:")
    row("onnx", ver("onnx"), "pip install onnx")
    row("onnxruntime", ver("onnxruntime"))
    row("tensorrt", ver("tensorrt"), "Jetson'da JetPack ile gelir; PC'de NVIDIA TensorRT kur")
    row("trtexec", shutil.which("trtexec") or (Path("/usr/src/tensorrt/bin/trtexec").exists() and "/usr/src/tensorrt/bin/trtexec"))
    row("cmake", shutil.which("cmake"), "C++ çıkarım kodu için gerekli")
    row("g++", shutil.which("g++"))

    total, used, free = shutil.disk_usage(Path.cwd())
    print(f"\nDisk: {free / 2**30:.0f} GB boş (BDD100K + çıktılar için ~40 GB önerilir)")


if __name__ == "__main__":
    main()
