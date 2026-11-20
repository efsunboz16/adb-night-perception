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

## Deney 2 · Ufuk bandı dilimleme (`adb/slicing.py`, `eval_distance.py --slice`)

Aynı model, yeniden eğitim yok. Tam kare çıkarıma ek olarak, uzak araçların toplandığı yatay bant (`scripts/horizon_band.py` ile eğitim verisinden belirlendi, en az 160 px) 384 px genişliğinde, 64 px örtüşen parçalara bölünüp 2x büyütülerek modele verildi. Tahminler birleştirilip sınıf bazında NMS uygulandı. İki mod aynı görüntülerde ölçüldü.

### Gece, güven eşiği 0,10 (kutu / kutu+ışık)

| Genişlik (px) | ~Mesafe (m) | Tam kare | Tam kare + bant |
|---|---|---|---|
| 4–8 | 249–499 | 0,04 / 0,53 (n=55) | 0,29 / 0,69 |
| 8–12 | 166–249 | 0,28 / 0,75 (n=665) | 0,46 / **0,85** |
| 12–16 | 125–166 | 0,54 / 0,83 (n=1877) | 0,62 / 0,89 |
| 16–24 | 83–125 | 0,69 / 0,89 (n=5007) | 0,72 / 0,92 |
| 24–32 | 62–83 | 0,75 / 0,91 (n=4113) | 0,76 / 0,93 |
| 32–64 | 31–62 | 0,81 / 0,93 (n=8801) | 0,83 / 0,94 |
| ≥ 64 | < 31 | 0,93 / 0,97 (n=11584) | 0,93 / 0,97 |

Yağmurlu gece, 166–249 m: 0,74 → 0,82 (n=73).

### Gecikme (RTX 5050 Laptop, PyTorch FP16, okuma ve ön/son işlem dahil, optimize edilmemiş)

| | ms / kare |
|---|---|
| Tam kare | 15,9 |
| Bant (4 parça) ek | 32,6 |
| Toplam | **48,5** |

### Gözlemler

1. Dilimleme 220 m bandında kutu+ışık recall'ını 0,75'ten **0,85**'e çıkarıyor (+10 puan). Yalnızca kutu: 0,28 → 0,46. Yakın mesafede değişiklik yok.
2. REQ-F-01 (220 m, ≥ 0,90) açığı 15 puandan **5 puana** indi.
3. Gecikme bedeli yüksek: toplam 48,5 ms, 50 ms bütçenin sınırında ve bu masaüstü GPU'su. Seçenekler: bandı iki karede bir işleyip aradaki kareleri takiple doldurmak (Faz 4), TensorRT (Faz 5), tek geniş şerit.
4. **Yanlış alarm** (hiçbir gerçek nesne kutusuna düşmeyen araç/ışık tahmini, kare başına):

   | Koşul | Eşik | Tam kare | Tam kare + bant |
   |---|---|---|---|
   | Gece | 0,10 | 2,87 | 4,27 (+%49) |
   | Gece | 0,25 | 1,18 | 1,75 (+%48) |
   | Yağmurlu gece | 0,10 | 3,10 | 4,84 (+%56) |
   | Yağmurlu gece | 0,25 | 1,15 | 1,84 (+%60) |

   Çapraz karşılaştırma: bant + eşik 0,25, 220 m'de 0,72 recall ve 1,75 yanlış alarm veriyor. Tam kare + eşik 0,10 ise 0,75 recall ve 2,87 yanlış alarm. Benzer recall'da %39 daha az yanlış alarm. Doğru karşılaştırma tek nokta değil, eşikler boyunca recall–yanlış alarm eğrisi.
5. **Ölçüm sınırları:** BDD'nin etiketlemediği uzak araçlar da yanlış alarm sayılıyor (ADR-007), bu yüzden mutlak değerler kötümser. Dilimleme tam da bu araçları daha çok bulduğu için artışın bir kısmı gerçek tespit olabilir. Gecikme çalıştırmalar arasında kararsız (aynı koşulda tam kare 15,9 / 32,8 ms). Güvenilir ölçüm Faz 5'te yapılacak, bant maliyeti tam karenin ~1,7–2 katı.
6. **ADB anlamı:** Yanlış alarmların büyük kısmının tek karelik olması bekleniyor. Takip (Faz 4) birkaç kare tutarlılık şartıyla bunları eleyecek. Gerçek ADB performansı video üzerinde ölçülecek.

## Deney 3 · Sözde etiketleme (ADR-012)

Temel model + dilimleme öğretmen olarak train/val gece ve alacakaranlık karelerine etiket ekledi: train'de 29.756 karenin 2.442'sinde (%8), 5.052 kutusuz ışık (çift kuralı, B) ve 805 etiketsiz uzak araç (C). Kare başına 0,20 ek etiket, mevcut etiketlerin ~%1'i. Kutu içi ışık kuralı (A), park halindeki araçlardaki yansımaları etiketlediği için önizleme sonrası kapatıldı. Yeni model temel modelle birebir aynı ayarlarla eğitildi (Kaggle 2×T4, 11 saat, aynı 25.000 kare). Tek fark etiketler.

### Gece, test, güven eşiği 0,10 (araç kutusu veya ışığı)

| Mesafe | Temel · tam | Sözde · tam | Temel · bant | Sözde · bant |
|---|---|---|---|---|
| 249 m+ (n=55) | 0,53 | 0,53 | 0,69 | 0,69 |
| **166–249 m (n=665)** | **0,75** | **0,75** | **0,85** | **0,83** |
| 125–166 m (n=1877) | 0,83 | 0,84 | 0,89 | 0,88 |
| 83–125 m (n=5007) | 0,89 | 0,90 | 0,92 | 0,92 |
| Yanlış alarm / kare | 2,87 | 2,88 | 4,27 | 4,26 |

Eşik 0,25'te: 166–249 m tam kare 0,59 → 0,62, bant 0,72 → 0,72. Yanlış alarm (tam) 1,18 → 1,25.

Doğrulama setinde (1.500 kare, ama etiketler değiştiği için tam karşılaştırılabilir değil): mAP50 0,640 → 0,639, head_light 0,644 → 0,651, tail_light 0,728 → 0,746.

### Sonuç: etki yok

1. 166–249 m bandında recall değişmedi. n=665 için ölçüm belirsizliği ±1,5–2 puan, gözlenen farklar bunun içinde. Yanlış alarm aynı.
2. Tek tutarlı değişim yüksek eşikte uzak recall'daki +3 puan. Model uzak ışıklarda biraz daha emin, ama düşük eşikte yakaladıkları aynı.
3. **Yorum:** Uzak mesafe darboğazı etiket eksikliği değil, çözünürlük gibi görünüyor. Aynı modelin uzak bölgeye büyütülmüş bakması (Deney 2) +10 puan kazandırdı, daha fazla etiket kazandırmadı. ADR-007'deki "eksik etiketler uzak ışıkları bastırtıyor" hipotezi bu doz ölçeğinde doğrulanmadı.
4. Sınırlama: Doz küçüktü (%1). Çok daha fazla sözde etiketle farklı bir sonuç çıkması dışlanamaz, ama Deney 2'deki kanıt önceliği çözünürlüğe veriyor.