# Kaggle'da eğitim

## 1. Zip'leri hazırla (bilgisayarında, proje klasöründe)
```powershell
tar -a -c -f bdd100k-images.zip -C data\raw\bdd100k\images\100k train val
tar -a -c -f adb-labels.zip -C data\bdd_adb_lights labels splits
```
`tar` Windows 10/11'de hazır gelir. PowerShell'in `Compress-Archive` komutu 2 GB üstünde hata verebildiği için onu kullanma.

## 2. Kaggle'a yükle
Kaggle → Datasets → New Dataset, iki ayrı dataset olarak. İkisi de **Private** olmalı (BDD100K lisansı yeniden dağıtıma izin vermiyor).
- `bdd100k-images.zip` (~5,3 GB)
- `adb-labels.zip` (küçük)

## 3. Notebook
Kaggle → Code → New Notebook → File → **Import Notebook** → `adb_kaggle_train.ipynb`.
İlk hücredeki talimatları uygula (GPU, Internet, iki dataset'i Input olarak ekle).

## 4. Sonuç
Output sekmesinden `runs/baseline/weights/best.pt` dosyasını indirip yerelde `runs/baseline/weights/` altına koy:
```
python eval/eval_conditions.py --weights runs/baseline/weights/best.pt --batch 2
```
