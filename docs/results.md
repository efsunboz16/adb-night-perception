# Sonuçlar

## Deney 1 · Temel model (baseline)

- **Model:** YOLO11s, 1280 px, 9,4 M parametre
- **Eğitim:** Kaggle 2×T4, 11 saat, batch 16, 25.000 kare (%75 gece, ADR-010), BDD araç etiketleri + otomatik ışık etiketleri (ADR-008/009)
- **Test:** BDD100K val (bizim test bölümümüz), sadece araç/insan sınıfları (0–5). Işık sınıflarının test etiketi yok.
- **Tarih:** 2026-11-18

### Koşul bazında (test)

| Koşul | Görüntü | P | R | mAP50 | mAP50-95 | car | truck | bus | motorcycle | bicycle | person |
|---|---|---|---|---|---|---|---|---|---|---|---|
| day | 4435 | 0,696 | 0,542 | 0,609 | 0,374 | 0,841 | 0,616 | 0,628 | 0,396 | 0,487 | 0,685 |
| night | 3363 | 0,658 | 0,543 | 0,593 | 0,347 | 0,786 | 0,594 | 0,560 | 0,443 | 0,529 | 0,643 |
| night_adverse | 566 | 0,634 | 0,532 | 0,594 | 0,372 | 0,786 | 0,619 | 0,692 | 0,462 | 0,383 | 0,620 |
| dusk | 648 | 0,770 | 0,528 | 0,621 | 0,381 | 0,836 | 0,618 | 0,666 | 0,412 | 0,542 | 0,652 |
| day_adverse | 823 | 0,720 | 0,530 | 0,599 | 0,371 | 0,839 | 0,651 | 0,543 | 0,435 | 0,442 | 0,683 |

Hız: RTX 5050 Laptop, FP16, 1280 px: 6,5 ms çıkarım + 2,5 ms ön/son işlem (görüntü başına, batch 4).

### Işık sınıfları (eğitim sırası doğrulama, 1.500 kare)

| Sınıf | P | R | mAP50 |
|---|---|---|---|
| head_light | 0,613 | 0,636 | 0,644 |
| tail_light | 0,650 | 0,745 | 0,728 |

Dikkat: Doğrulama etiketleri de otomatik üretildiği için bu değerler gerçek doğruluğu değil, modelin otomatik etiketleyiciyi ne kadar taklit ettiğini ölçer. Gerçek doğruluk için insan kontrollü set gerekir.

### Gözlemler

1. Gece–gündüz farkı küçük: genel mAP50'de 1,6 puan, car sınıfında 5,5 puan. Yağmurlu gece normal geceyle aynı seviyede.
2. Recall her koşulda ~0,53–0,54 ile sınırlı, precision 0,63–0,77. Ana darboğaz recall. Koşuldan bağımsız olması, sorunun küçük/uzak araçlarda olduğunu düşündürüyor (doğrulanacak: mesafe bazlı recall).
3. Motosiklet en zayıf sınıf (AP50 0,40–0,46). Eğitim verisinde %0,35.
4. Literatürdeki BDD100K sonuçlarıyla doğrudan karşılaştırılamaz (farklı sınıf seti, bölüm ve çözünürlük).

## Deney 1b · Mesafe bazlı recall (`eval/eval_distance.py`)

Aynı model, test seti. Araç sınıfları (car, truck, bus, motorcycle) sınıftan bağımsız eşleştirildi, IoU ≥ 0,5. Mesafe kutu genişliğinden türetildi: ≈ fx · 1,8 m / genişlik, fx = 1109 px (varsayılan kamera, yaklaşık).

- **kutu:** araç kutusu tespit edildi
- **kutu+ışık:** araç kutusu VEYA kutunun içinde bir head/tail_light tespiti var (ADB için yeterli: o açı kısılabilir)

### Güven eşiği 0,10 (kutu / kutu+ışık)

| Genişlik (px) | ~Mesafe (m) | Gündüz | Gece | Yağmurlu gece |
|---|---|---|---|---|
| 4–8 | 249–499 | 0,10 / 0,17 (n=324) | 0,04 / 0,53 (n=55) | 0,20 / 0,70 (n=10) |
| 8–12 | 166–249 | 0,44 / 0,48 (n=2310) | 0,28 / **0,75** (n=665) | 0,30 / 0,74 (n=73) |
| 12–16 | 125–166 | 0,66 / 0,69 (n=4286) | 0,54 / 0,83 (n=1877) | 0,48 / 0,81 (n=237) |
| 16–24 | 83–125 | 0,77 / 0,79 (n=8754) | 0,69 / 0,89 (n=5007) | 0,67 / 0,87 (n=651) |
| 24–32 | 62–83 | 0,83 / 0,85 (n=6020) | 0,75 / 0,91 (n=4113) | 0,74 / 0,90 (n=579) |
| 32–64 | 31–62 | 0,88 / 0,89 (n=13138) | 0,82 / 0,93 (n=8801) | 0,79 / 0,91 (n=1338) |
| ≥ 64 | < 31 | 0,95 / 0,96 (n=18418) | 0,93 / 0,97 (n=11584) | 0,93 / 0,97 (n=2062) |

Güven eşiği 0,25'te gece 166–249 m kutu+ışık recall'ı 0,75'ten 0,59'a düşüyor (tam tablo: `runs/baseline/eval_distance.json`).

### Gözlemler

1. **Recall mesafeyle sert düşüyor.** Gece sadece kutu: < 31 m'de 0,93, 166–249 m'de 0,28. Genel recall'ın 0,53'te kalmasının sebebi uzak araçlar.
2. **Işık sınıfları gece uzak mesafede belirleyici.** 166–249 m'de kutu+ışık recall'ı 0,28'den 0,75'e çıkıyor (+47 puan). Gündüz katkı yok (0,44 → 0,48), beklendiği gibi. Işıklar sayesinde gece uzak mesafe recall'ı gündüzü geçiyor (0,75 / 0,48).
3. **Gereksinim durumu (gece, eşik 0,10):** REQ-F-01 (220 m, ≥ 0,90): **0,75**, 15 puan eksik. REQ-F-02 (100 m, ≥ 0,90): 83–125 m'de **0,89**.
4. **ADB için düşük eşik gerekli.** Eşik 0,25'e çıkınca uzak mesafe recall'ı 16 puan düşüyor. Bedeli (gereksiz kısma) henüz ölçülmedi.
5. **Sınırlamalar:** Mesafeler yaklaşık. Recall yalnızca BDD'nin etiketlediği araçlar üzerinden (uzak gece etiketleri eksik, ADR-007). Kutu+ışık ölçütü kutu içinde herhangi bir ışık tahminini yeterli sayıyor.
6. **Faz 3 odağı:** gece 125 m ötesinde ışık tespiti. Otomatik ışık etiketleri yalnızca BDD kutularının içinden üretildiği için BDD'nin kaçırdığı uzak araçların ışıkları da etiketsiz. Sözde etiketleme ilk deney adayı.
