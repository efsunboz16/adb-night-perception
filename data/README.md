# Veri

Bu klasör git'e eklenmez (bkz. `.gitignore`). Beklenen yapı:

```
data/
  raw/bdd100k/        # images/100k/{train,val}, labels/det_20/
  raw/acdc/           # sadece night + rain alt kümeleri
  raw/pvdn/
  bdd_adb/            # scripts/bdd_prepare.py çıktısı (YOLO formatı)
  own_lights/         # kendi far/lamba anahtar nokta etiketleri
```

Hazırlama: `python scripts/bdd_prepare.py --bdd data/raw/bdd100k --out data/bdd_adb --copy-images`
