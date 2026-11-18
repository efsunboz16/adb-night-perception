# Eğitim

1. `python train/train_baseline.py --quick`: hattı uçtan uca dener (verinin %5'i, 3 epoch)
2. `python train/train_baseline.py`: tam eğitim (30 epoch, 1280 px)
3. Değerlendirme: `python eval/eval_conditions.py --weights runs/baseline/weights/best.pt`
