"""GPU sorun tespiti: Ultralytics'ten bağımsız, adım adım CUDA / cuDNN testi.

Kullanım: python scripts/gpu_check.py
"""

import os
import traceback

import torch


def step(name, fn):
    try:
        out = fn()
        torch.cuda.synchronize()
        print(f"  [OK   ] {name}{'  -> ' + str(out) if out is not None else ''}")
        return True
    except Exception as e:
        print(f"  [HATA ] {name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
        return False


def conv_test(bs, size, half):
    m = torch.nn.Conv2d(3, 32, 3, padding=1).cuda()
    x = torch.rand(bs, 3, size, size, device="cuda")
    if half:
        with torch.autocast("cuda", dtype=torch.float16):
            return tuple(m(x).shape)
    return tuple(m(x).shape)


def main():
    print(f"torch {torch.__version__} | CUDA derlemesi {torch.version.cuda} | "
          f"cuDNN {torch.backends.cudnn.version()} | PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}, sm_{p.major}{p.minor}, {p.total_memory / 2**30:.1f} GB")
    print(f"Desteklenen mimariler: {torch.cuda.get_arch_list()}\n")

    print("1) Temel CUDA (cuDNN yok):")
    step("tensör ayırma 100 MB", lambda: torch.empty(25_000_000, device="cuda").numel())
    step("matris çarpımı", lambda: tuple((torch.rand(1024, 1024, device="cuda") @ torch.rand(1024, 1024, device="cuda")).shape))

    print("\n2) Konvolüsyon, cuDNN açık:")
    ok_fp32 = step("conv fp32, 8x256", lambda: conv_test(8, 256, False))
    ok_fp16 = step("conv fp16 (AMP), 8x256", lambda: conv_test(8, 256, True))

    print("\n3) Konvolüsyon, cuDNN kapalı:")
    torch.backends.cudnn.enabled = False
    step("conv fp32, 8x256", lambda: conv_test(8, 256, False))
    step("conv fp16 (AMP), 8x256", lambda: conv_test(8, 256, True))
    torch.backends.cudnn.enabled = True

    print("\n4) Gerçekçi yük, cuDNN açık:")
    step("conv fp16, 8x1280", lambda: conv_test(8, 1280, True))

    print("\nYorum:")
    if not (ok_fp32 and ok_fp16):
        print("  2. adım hata verdi -> sorun cuDNN/PyTorch derlemesinde. Bu çıktıyı gönder.")
    else:
        print("  Konvolüsyonlar çalışıyor -> sorun Ultralytics'in AMP kontrolüne özgü. Bu çıktıyı gönder.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
