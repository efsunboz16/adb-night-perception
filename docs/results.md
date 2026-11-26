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

## Deney 4 · P2 tespit kafası (ADR-013)

`configs/models/yolo11s-p2.yaml`: YOLO11s'e stride-4 dalı eklendi, dört çıkışlı Detect (P2–P5). Model 9,58M parametre, 117,9 GFLOPs @1280 (temel: 9,46M / 88,8). Kaggle kotası yetmediği için 11 saat yerine **3 saat** eğitildi (2×T4, batch 10). Başlangıç noktası temel modelin `best.pt` dosyası. Omurga ve yukarıdan aşağı boyun aktarıldı. Aşağıdan yukarı yol ve Detect kafası katman indeksleri değiştiği için sıfırdan öğrenildi. Veri ve etiketler temel modelle aynı.

### Gece, test, güven eşiği 0,10 (araç kutusu veya ışığı)

| Mesafe | Temel · tam | P2 · tam | Temel · bant | P2 · bant |
|---|---|---|---|---|
| 249 m+ (n=55) | 0,53 | 0,60 | 0,69 | 0,73 |
| **166–249 m (n=665)** | **0,75** | **0,74** | **0,85** | **0,84** |
| 125–166 m (n=1877) | 0,83 | 0,82 | 0,89 | 0,88 |
| 83–125 m (n=5007) | 0,89 | 0,88 | 0,92 | 0,92 |
| < 31 m (n=11584) | 0,97 | 0,97 | 0,97 | 0,97 |
| Yanlış alarm / kare | 2,87 | 2,90 | 4,27 | 4,46 |

Eşik 0,25'te 166–249 m, tam kare: 0,59 → 0,56. Yanlış alarm 1,18 → 1,21. Yağmurlu gece 166–249 m, tam kare: 0,74 → 0,78 (n=73, ±5 puan).

### Sonuç: hedefe ulaşılmadı (220 m tam kare 0,74, hedef ≥ 0,85)

1. 166–249 m'de fark yok (−1 puan, standart hata ±1,7). 249 m+ bandındaki +7 puan n=55'te istatistiksel olarak anlamlı değil (±6,7, yani 4 araç).
2. P2 ile bant birlikte de kazanç getirmiyor (0,85 → 0,84). Yani model, dilimlemenin bulduğu uzak araçları kendi P2 kafasıyla bulamıyor.
3. Bedel: +%33 FLOPs. Gecikme bu çalıştırmada tam kare 43,8 ms ölçüldü. Ölçüm kararsız olduğu için (bkz. Deney 2) temel modelle doğrudan karşılaştırılmadı.
4. **Sınırlama:** 3 saatlik ince ayar, 11 saatlik temel eğitimle eşit bütçe değil. Yeni dal ve kafa sıfırdan başladı. Uzun eğitimle sonuç değişebilir, ama 3 saatte yakın mesafede temel seviyeye ulaşıp uzakta hiç geçmemesi zayıf bir işaret.
5. **Yorum:** Dilimlemenin kazancı "daha ince özellik haritası"ndan değil, **girişin büyütülmesinden** geliyor olabilir. 2x büyütmede 8 px'lik araç 16 px olur ve omurganın tüm katmanları onu daha büyük bir nesne olarak işler. P2 ise yalnızca son tespiti ince haritadan yapar. Uzak nesnenin anlamsal özellikleri, sığ stride-4 katmanlarında zayıf kalıyor.

## Deney 5 · Dilimlemenin kazancı nereden geliyor? Tam kare büyütme testi

Temel model, eğitimsiz. Test seti gece karelerinde tam kare, giriş boyutu büyütülerek değerlendirildi (Ultralytics letterbox 1280×720 görüntüyü büyütür). Bant parçaları 2x büyütüldüğü için 2560, ölçek bakımından dilimlemeyle eşit.

### Gece, 166–249 m (n=665), kutu / kutu+ışık

| Çıkarım | Ölçek | Eşik 0,10 | Yanlış alarm | Eşik 0,25 | Yanlış alarm |
|---|---|---|---|---|---|
| Tam kare 1280 | 1x | 0,28 / 0,75 | 2,87 | – / 0,59 | 1,18 |
| Tam kare 1600 | 1,25x | 0,32 / 0,73 | 2,82 | 0,13 / 0,57 | 1,15 |
| Tam kare 1920 | 1,5x | 0,32 / 0,76 | 2,92 | 0,13 / 0,56 | 1,19 |
| Tam kare 2560 | 2x | 0,38 / 0,79 | 3,03 | 0,16 / 0,60 | 1,18 |
| Tam kare 1280 + bant | 2x (bant) | 0,46 / **0,85** | 4,27 | – / **0,72** | 1,75 |

249 m+ (n=55): 1280 0,53, 2560 0,58, bant 0,69. Yakın mesafede 2560 hafif kötüleşiyor: < 31 m kutu 0,93 → 0,90, çünkü büyük araçlar eğitim ölçeğinin dışına çıkıyor.

### Sonuç

1. **Ölçek kazancın küçük bir kısmını açıklıyor.** 2x tam kare, 0,10 eşikte +4 puan veriyor (0,75 → 0,79, ~2 standart hata). Dilimlemenin +10 puanının yaklaşık %40'ı. 0,25 eşikte ölçekten kazanç yok (0,59 → 0,60).
2. **Kalan kısım parça görüşünden ve iki bakışın birleşiminden geliyor.** Eşit yanlış alarm düzeyinde karşılaştırınca: bant 0,25 eşikte 1,75 yanlış alarmla 0,72 recall veriyor. 2560 tam kare 0,10 eşikte 3,03 yanlış alarmla 0,79, 0,25 eşikte 1,18 yanlış alarmla 0,60. Dilimleme recall–yanlış alarm dengesinde tam kare büyütmeden belirgin şekilde iyi.
3. **Maliyet:** 2560 tam kare, 1280'in ~4 katı piksel işliyor (çalışma süresi 1600'de 68 sn, 2560'ta 142 sn). Yalnızca bant parçalarını işlemek daha ucuz ve daha iyi.
4. **Deney 4 ile birlikte yorum:** Uzak mesafe kazancı "modelin daha ince ölçekte bakması"ndan (P2) veya "her şeyin büyütülmesi"nden değil, uzak bölgenin **ayrıca, dar bir kadrajla** incelenmesinden geliyor. Muhtemel sebep: parçada yakın araçların farları ve yansımaları yok, uzak ışıklar göreceli olarak öne çıkıyor.

## Deney 6 · Takip ve bant zamanlaması (video, Faz 4)

**Veri:** MOT 2020 Images kareleri (5 fps). Bu videolar 100K setinin videolarıyla aynı. Etiketli test görüntüsü, videonun 10. saniyesindeki kare. `scripts/prepare_video_eval.py` bu kareyi görüntü benzerliğiyle buldu: 189 videonun 152'sinde tam 51. kare, hepsi test setinden, sızıntı yok. 68 gece videosu kullanıldı. Her videoda etiketli kare ve öncesindeki 15 kare (3 sn) işlendi. Ölçüm etiketli karede yapıldı. Takipçi: `adb/tracking.py` (iki aşamalı eşleştirme, düşük güvenli aday izler, onay için 2 görüş).

### Etiketli kare, güven ≥ 0,10 (araç kutusu veya ışığı)

| Mod | < 62 m (n=450) | 62–125 m (n=194) | 125–249 m (n=40) | Yanlış alarm / kare | Süre / kare |
|---|---|---|---|---|---|
| Tam kare | 0,93 | 0,91 | 0,82 | 3,75 | 14,4 ms |
| Tam + bant | 0,98 | 0,94 | **0,93** | 6,54 | 47,5 ms |
| Takip · tam | 0,89 | 0,88 | 0,70 | 3,41 | 14,9 ms |
| Takip · tam + bant | 0,93 | 0,92 | 0,82 | 6,18 | 48,1 ms |
| Takip · bant iki karede bir | 0,91–0,92 | 0,89–0,92 | 0,80–0,82 | 4,63–4,81 | 31,5 ms |

125–249 m için standart hata ±4–7 puan (n=40). "İki karede bir" satırı, etiketli karede bandın çalıştığı ve çalışmadığı iki durumun aralığı.

### Yanlış alarmların kalıcılığı (tek kare modları, etiketli kare)

| Mod | Eşik | Yanlış alarm | Önceki karede de var | Önceki 2 karede de var |
|---|---|---|---|---|
| Tam | 0,10 | 255 | %53 | %31 |
| Tam | 0,25 | 123 | %63 | %34 |
| Tam + bant | 0,10 | 445 | %56 | %34 |
| Tam + bant | 0,25 | 187 | %66 | %42 |

### Sonuç

1. **İlk sürüm yanlış alarmı ikiye katladı** (3,75 → 7,87). Sebep: tahminle sürdürülen yakın araç kutuları ("hayalet" kutular) ve izin taşıdığı eski yüksek güven. v2'de görünmeyen iz yalnızca küçükse (uzak) ve 1 kare çıktıya veriliyor. Çıktı güveni o karedeki tespitin güveni. Tablo v2'nin sonucu.
2. **Takip yanlış alarm süzgeci olarak zayıf:** −%9 (3,75 → 3,41). Bedeli uzak recall'da: 0,82 → 0,70; bantla 0,93 → 0,82. Yanlış alarmların ~%55–65'i önceki karede de aynı yerde. Tek karelik olanlar ise çoğunlukla kesikli tekrar eden aynı nesneler: sokak lambası, yansıma, BDD'nin etiketlemediği uzak araç. "En az 2 kez gör" kuralı bunları eleyemiyor. Uzak gerçek araçlar da kesikli tespit edildiği için kural onları geciktiriyor.
3. **Bant iki karede bir:** süre 47,5 → 31,5 ms. Ancak bu örnekte uzak recall bantsız tam kare düzeyinde kaldı (0,80–0,82). Yani zamanlamanın kazancı takibin kaybıyla siliniyor. n=40 olduğu için kesin değil, ama olumlu bir işaret de yok.
4. **5 fps sınırlaması:** Gerçek kamera 30 fps'te onay 33 ms'de gelir ve kareler arası kayma 6 kat azdır. Takibin recall bedeli burada olduğundan büyük ölçülüyor. Yine de kalıcı yanlış alarmlar kare hızından bağımsız.
5. **Maske:** Gece şehir karelerinde segmentlerin ~%72'si kısık, ham maskede dakikada ~210–250 segment değişimi var (hedef ≤ 2). Takip bunu çözmüyor. Titreme far kararındaki histerezisin işi.

### Deney 6b · ADB sistem metrikleri (far kararı modülü, `adb/decision.py`)

Far kararı: güven ≥ 0,25, paralaks düzeltmesi, 1° güvenlik payı, 0,5 sn histerezis. 68 gece videosu.

| Mod | Kamaşma ihlali < 62 / 62–125 / 125–249 m | Uzun far kullanımı | Kısık oran | Segment değişimi/dk ham → histerezis | Süre |
|---|---|---|---|---|---|
| Tam kare | 0,02 / 0,01 / 0,00 | 0,68 | 0,82 | 181 → 118 | 42 ms |
| Tam + bant | 0,01 / 0,01 / 0,00 | 0,63 | 0,83 | 209 → 127 | 108 ms |
| Takip · tam | 0,02 / 0,02 / 0,03 | **0,76** | 0,81 | 151 → 104 | 44 ms |
| Takip · bant iki karede bir | 0,02 / 0,02–0,03 / 0,03 | 0,74–0,75 | 0,81 | 158 → 103–107 | 77 ms |

(Süreler bu çalıştırmada önceki çalıştırmanın ~3 katı ölçüldü. Dizüstü GPU'da ölçüm kararsız, bkz. Deney 2.)

**Gözlemler:**
1. **Ölçülen sahneler sistem metriği için uygun değil:** 68 gece videosunun 48'i şehir içi. Segmentlerin %82'si zaten kısık olduğu için kamaşma ihlali her modda neredeyse sıfır. Metrik modları ayırt etmiyor. Gerçek ADB şehir içinde (aydınlatmalı yolda) devre dışıdır.
2. **Takip uzun far kullanımını artırıyor** (0,68 → 0,76): yanlış alarm kaynaklı gereksiz kısmalar azalıyor. Bedeli: 125–249 m'de 40 aracın 1'i kamaştırıldı (0,00 → 0,03).
3. **Bant** şehir içinde kamaşmayı azaltmıyor (zaten ~0), yanlış alarmı artırdığı için kullanımı düşürüyor (0,68 → 0,63).
4. **"Segment değişimi" titreme ölçüsü olarak yanlış:** araç segmentler arasında kayarken oluşan doğal değişimleri de sayıyor. Histerezis bunu %35 azaltıyor, ama hedefle (≤ 2/dk) karşılaştırılamaz. Ölçü değiştirildi: titreme = 1 sn'den kısa süren açılma veya kısılma.
5. **Sonraki ölçüm:** sahne filtresi eklendi (`scripts/scene_index.py`, `eval_video.py --scenes highway`). Test setindeki 3.363 gece görüntüsünün 1.007'si otoyol. Video setinde 18 otoyol gece videosu var (küçük örnek).

### Deney 6c · Otoyol ve yerleşim alanı (20 gece videosu)

`eval_video.py --scenes highway residential`. Far kararı: güven ≥ 0,25, histerezis 0,5 sn. Titreme: 1 sn'den kısa süren açılma veya kısılma.

| Mod | Recall 125–249 m (n=18) | Yanlış alarm / kare | Kamaşma ihlali < 62 / 62–125 / 125–249 m | Uzun far kullanımı | Titreme/dk ham → histerezis |
|---|---|---|---|---|---|
| Tam kare | 0,83 | 4,10 | 0,04 / 0,00 / 0,00 | 0,65 | 74 → 34 |
| Tam + bant | 0,94 | 7,85 | 0,03 / 0,00 / 0,00 | 0,60 | 110 → 36 |
| Takip · tam | 0,72 | 3,25 | 0,04 / 0,02 / 0,06 | **0,77** | 45 → 26 |
| Takip · bant iki karede bir | 0,72–0,83 | 4,90–5,35 | 0,04–0,05 / 0,02–0,05 / 0,06 | 0,76 | 46–61 → 24–29 |

**Gözlemler:**
1. Otoyolda segmentlerin %67–71'i kısık (şehirde %82). Metrikler modları ayırmaya başlıyor.
2. **Takip, uzun far kullanımını +12 puan artırıyor** (0,65 → 0,77) ve titremeyi azaltıyor (34 → 26/dk). Bedeli: 18 uzak aracın 1'i kamaştırıldı. Bu örnekte farkın anlamlı olup olmadığı söylenemez.
3. **Bant** uzak recall'ı artırıyor (0,83 → 0,94, yalnız kutu 0,50 → 0,83), ama kamaşma zaten 0 olduğu için sistem düzeyinde kazanç görünmüyor; yanlış alarm kullanımı düşürüyor. 18 araçla bu soru cevaplanamaz.
4. **Titreme hedefin çok üstünde** (en iyi 24/dk, hedef ≤ 2). Sıradaki: güven eşiği ve histerezis taraması (`--sweep-conf`, `--sweep-release`) ile kamaşma–kullanım–titreme dengesinin çıkarılması.

### Deney 6d · Far kararı taraması: güven eşiği × histerezis

`eval_video.py --sweep-conf 0.15 0.25 0.35 0.5 --sweep-release 0.5 1 2`. Seçilmiş satırlar (tablonun tamamı çalıştırma çıktısında):

| Sahne | Mod | Güven | Histerezis | Kamaşma ihlali (tümü) | 125 m+ | Uzun far kullanımı | Titreme / dk |
|---|---|---|---|---|---|---|---|
| Tüm gece (68) | Tam kare | 0,25 | 0,5 sn | 0,016 (11/686) | 0/42 | 0,68 | 24,3 |
| Tüm gece (68) | Takip | 0,35 | 0,5 sn | 0,028 (19/686) | 1/42 | 0,77 | 14,0 |
| Tüm gece (68) | **Takip** | **0,35** | **1,0 sn** | **0,016 (11/686)** | **1/42** | **0,61** | **6,3** |
| Tüm gece (68) | Takip | 0,35 | 2,0 sn | 0,012 (8/686) | 1/42 | 0,51 | 0,4 |
| Tüm gece (68) | Takip | 0,5 | 1,0 sn | 0,034 (23/686) | 4/42 | 0,63 | 7,0 |
| Otoyol + yerleşim (20) | Tam kare | 0,35 | 1,0 sn | 0,021 (4/189) | 0/18 | 0,57 | 3,8 |
| Otoyol + yerleşim (20) | **Takip** | **0,35** | **1,0 sn** | **0,032 (6/189)** | **1/18** | **0,65** | **1,2** |
| Otoyol + yerleşim (20) | Takip | 0,35 | 2,0 sn | 0,026 (5/189) | 1/18 | 0,61 | 0,0 |

Not: Bu çalıştırmada kamaşma oranının paydasında far kapsamı (±16°) dışındaki araçlar da vardı. Onlar hiç kamaştırılamayacağı için oranlar olduğundan iyi görünüyor. `glare_and_utilization` düzeltildi, sonraki çalıştırmalar yalnızca kapsam içindeki araçları sayıyor.

**Gözlemler:**
1. **Titremeyi belirleyen histerezis.** 0,5 → 1,0 sn titremeyi 2–4 kat düşürüyor. 2,0 sn titremeyi neredeyse sıfırlıyor, ama uzun far kullanımı 0,1 puan daha düşüyor. Güven eşiğinin titremeye etkisi daha küçük.
2. **Güven 0,5 tehlikeli:** uzak kamaşma 1/42'den 4/42'ye çıkıyor. 0,15–0,35 arası benzer.
3. **Takip her ayarda daha az titreme ve daha yüksek kullanım veriyor** (otoyol, 0,35 / 1,0 sn: titreme 3,8 → 1,2, kullanım 0,57 → 0,65). Bedeli kamaşmada küçük bir artış (4 → 6 / 189).
4. **Seçilen ayar: takip, güven 0,35, histerezis 1,0 sn.** Otoyolda titreme hedefi (≤ 2/dk) karşılanıyor (1,2). Tüm gecede karşılanmıyor (6,3, şehir içi yoğun). Kamaşma ihlali %1,6–3,2, hedef ≤ %1: **karşılanmıyor.**
5. **Bant iki karede bir** sistem düzeyinde takip · tam ile aynı sonucu veriyor (otoyolda kamaşma, kullanım ve titreme aynı), ek ~20 ms karşılığında. Uzak araç sayısı (42) farkı göstermeye yetmiyor.

## Deney 7 · Kamaşma ihlali teşhisi (`eval/diagnose_glare.py`)

68 gece videosu. Takip açık, far kararı güven ≥ 0,35, histerezis 1,0 sn. Etiketli karede far kapsamındaki 550 aracın 11'i açık bir segmentte kaldı (%2,0). Her ihlal sebebine göre gruplandı ve görseli çıkarıldı. "Işıklı araç": kutusunun alt %75'inde parlak nokta (gri ≥ 200) olan araç, yani lambası yanan ve içinde sürücü olan, ADB'nin korumakla yükümlü olduğu araç.

| Araç | Far kapsamında | İhlal | Oran | Görülmedi | Düşük güven | Takip onayı | Yalnız ışık | Geometri |
|---|---|---|---|---|---|---|---|---|
| Işıklı | 374 | 5 | **%1,3** | 2 | 2 | 1 | 0 | 0 |
| Işıksız (park etmiş) | 176 | 6 | %3,4 | 4 | 2 | 0 | 0 | 0 |

**Gözlemler:**
1. **Geometri hatası sıfır.** Model bir aracı eşiğin üstünde bulduğunda doğru segment her seferinde kısıldı. Açı dönüşümü, paralaks düzeltmesi ve 1° güvenlik payı doğrulandı. Tüm ihlaller tespit kaynaklı.
2. **İhlallerin yarısından fazlası ışıksız, park etmiş araçlarda.** Kamaşacak sürücü yok. BDD bu araçları da etiketlediği için ölçülen %2,0 olduğundan kötü görünüyordu. ADB için asıl oran ışıklı araçlarda: %1,3 (hedef ≤ %1, n=374, ±0,6).
3. **Işıklı araç ihlallerinin 4/5'i iki videodan, ikisinde de görüntü bozuk:** b1e2346e'de ön camda su damlası, leke ve buğu var, uzak farlar dağınık lekeye dönüşmüş. b24380ab'de kamera sarsıntısından aşırı hareket bulanıklığı var, her ışık uzun bir iz bırakmış. Bu durumlar REQ-S-02 kapsamında: görüntü bozuksa uzun far kapanmalı. Güvenlik izleyici bu iki videoyu yakalarsa ışıklı araç ihlali 1/374'e (%0,3) düşer.
4. **Işıklı araç tespiti görüntüden yapılıyor:** test etiketlerinde lamba sınıfı yok. Yansımalı park etmiş araçlar "ışıklı" sayılabilir. Görsel kontrolde 5 ışıklı ihlalin 4'ü gerçekten lambası yanan araçtı.
5. **Sıradaki:** güvenlik izleyici. Bulanıklık eşiği (`system.yaml` → `blur_laplacian_min: 15`) tahmindi. `eval/calibrate_safety.py` ile veriden belirlenecek.

## Deney 8 · Güvenlik izleyici: bulanıklık / kirli cam algılanabilir mi? (`eval/calibrate_safety.py`)

Deney 7'de ışıklı araç ihlallerinin 4/5'i iki bozuk videodan geldi: b1e2346e'de ıslak ve lekeli ön cam, b24380ab-63272e5a'da kamera sarsıntısı. REQ-S-02'ye göre görüntü bozuksa uzun far kapanmalı. 68 gece videosunda dört görüntü kalitesi ölçüsü denendi. Tabloda bozuk videoların, bozulma yönündeki sırası var (1 = en bozuk görünen).

| Ölçü | Islak cam · 6 kare ortancası | Islak cam · yalnız etiketli kare | Sarsıntı · 6 kare ortancası | Sarsıntı · yalnız etiketli kare |
|---|---|---|---|---|
| Keskinlik (Laplace varyansı, düşük) | 14 | 10 | 21 | 33 |
| Parlak kenar enerjisi (düşük) | 12 | 12 | 34 | 68 |
| Işık lekesi uzaması (yüksek) | **4** | 65 | 22 | **5** |
| Hale oranı (yüksek) | 34 | 4 | 58 | 59 |

**Sonuç: olumsuz.**
1. **Genel keskinlik ölçüleri işe yaramıyor.** Hareket bulanıklığı ışıkları yumuşatmıyor, keskin kenarlı çizgilere çeviriyor. Sarsıntılı video parlak kenar ölçüsünde en keskin video çıktı (68/68).
2. **Işık lekesi uzaması** iki bozulmayı da bir ölçüm ayarında yakalıyor, ama aynı ayarda ikisini birden değil. Islak camı 6 karenin ortancası, sarsıntıyı tek kare yakalıyor. İki bozulma da anlık ve kareden kareye değişiyor.
3. **Yalnızca 2 bozuk örnek var.** Bu iki videoya göre seçilen bir eşik verinin rastlantısına uyar, genellenmez.
4. **Karar:** Bulanıklık kontrolü kapalı (`blur_laplacian_min: null`). REQ-S-02'nin bulanıklık kısmı açık. Gerçek çözüm muhtemelen etiketli bozuk görüntülerle eğitilmiş bir görüntü kalitesi sınıflandırıcısı ya da yağmur sensörü / silecek durumu gibi araç sinyalleri. Kamaşma hedefi (ışıklı araçlarda ≤ %1) bu yüzden karşılanmıyor: %1,3.
## Deney 9 · TensorRT FP16 motor gecikmesi (Faz 5, `trtexec`)

RTX 5050 Laptop (8 GB, Blackwell sm_120), TensorRT 10.16.1, CUDA 12.8. Prizde, Windows güç modu "En iyi performans". `trtexec --fp16 --warmUp=2000 --iterations=1000 --useSpinWait`, batch sabit. Ölçülen yalnızca motor: ön/son işleme, NMS, takip hariç.

| Motor | Giriş | GPU hesaplama p50 | p95 | p99 | Veri aktarımı dahil p50 | p95 | p99 | Değişim katsayısı |
|---|---|---|---|---|---|---|---|---|
| Tam kare | 736×1280, batch 1 | 2,90 | 4,22 | 8,76 | 4,08 | 5,38 | 9,78 | %31,9 |
| **Tam kare + CUDA Graph** | 736×1280, batch 1 | **2,31** | **2,33** | **2,78** | **3,23** | **3,36** | **3,86** | **%4,3** |
| Bant | 4 × 704×768 | 5,56 | 5,84 | 5,95 | 7,76 | 8,53 | 9,29 | düşük |

**Gözlemler:**
1. **Algılama, tam kare + bant birlikte ~11 ms (p95 ~12 ms).** Deney 2'de aynı iş PyTorch ile 48,5 ms sürmüştü (okuma ve ön/son işleme dahil, bu yüzden doğrudan karşılaştırılamaz). 30 fps kare aralığı 33 ms, gecikme bütçesi 50 ms.
2. **Tam kare dalgalanmasının kaynağı CPU.** CUDA Graph olmadan `trtexec` sıraya koyma (enqueue) süresinin GPU süresini aştığını bildirdi: küçük model çok sayıda kısa çekirdek çalıştırıyor, dizüstünün CPU güç yönetimi bu süreyi oynatıyor. CUDA Graph ile sıraya koyma ~3 ms'den ~0,01 ms'ye indi, p99 üçte bire düştü. Faz 3–4'teki 16–44 ms kararsızlığının da büyük ihtimalle kaynağı bu.
3. **Bant tam karenin 1,9 katı** (CUDA Graph'sız ölçümle). Faz 3 tahminiyle (~1,7–2 kat) uyumlu. Bant batch 4 ile uzun ve az çekirdek çalıştırdığı için zaten kararlı.
4. **Bantta veri aktarımı 2 ms:** giriş FP32, 4 parça 26 MB. C++ hattında görüntüyü 8 bit gönderip ölçekleme/normalizasyonu GPU'da yapmak aktarımı dörtte birine indirir.
5. **Henüz ölçülmedi:** FP16'nın doğruluğa etkisi (mAP, uzak recall, kamaşma) ve Python hattının uçtan uca gecikmesi (`eval/bench_latency.py`). Bant zamanlaması kararı (ADR-016) bunlardan sonra verilecek.

## Deney 10 · FP16 doğruluk eşitliği (TensorRT FP16 ve PyTorch, aynı görüntüler)

Motorlar `deploy/export.py` ile (TensorRT 10.16.1, RTX 5050). PyTorch ölçümleri aynı gün aynı kodla tekrarlandı.

**Mesafe bazlı recall, gece (3.363 görüntü), araç kutusu veya ışığı, güven ≥ 0,10** (PyTorch / FP16):

| Mesafe | > 249 m (n=55) | 166–249 m (n=665) | 125–166 m (n=1.877) | 83–125 m (n=5.007) | Yanlış alarm / kare |
|---|---|---|---|---|---|
| Tam kare | 0,527 / 0,527 | 0,750 / 0,752 | 0,833 / 0,833 | 0,894 / 0,894 | 2,87 / 2,87 |
| Tam kare + bant | 0,691 / 0,691 | **0,848 / 0,850** | 0,895 / 0,896 | 0,921 / 0,921 | 4,27 / 4,27 |

Güven ≥ 0,25'te de tüm hücreler üç basamağa kadar aynı.

**mAP50 (araç sınıfları)** (PyTorch / FP16). FP16 için kare 1280×1280 doğrulama motoru; PyTorch `val` dikdörtgen batch kullanıyor, dolgu farklı:

| Koşul | Gündüz | Gece | Yağmurlu gece | Alacakaranlık |
|---|---|---|---|---|
| mAP50 | 0,609 / 0,611 | 0,593 / 0,589 | 0,594 / 0,588 | 0,621 / 0,617 |
| car | 0,841 / 0,841 | 0,786 / 0,785 | 0,786 / 0,783 | 0,836 / 0,831 |

**Video sistem metrikleri** (20 otoyol + yerleşim gece videosu, güven ≥ 0,25, histerezis 1,0 sn): kamaşma ihlali, uzun far kullanımı ve kısık segment oranı altı modun hepsinde PyTorch ile **birebir aynı**. Titreme (histerezisli) aynı; histerezissiz ham segment değişimi ±1,25/dk.

**Gözlemler:**
1. **FP16 uzak recall'ı korudu.** 220 m hedef bandında fark en fazla 2 tespit (665'te). Yanlış alarm aynı.
2. **mAP farkı ≤ 0,6 puan**, yönü koşula göre değişiyor (gündüz +0,2, yağmurlu gece −0,6). Doğrulama motorunun kare dolgusu PyTorch'un dikdörtgen batch'inden farklı olduğu için bu fark tamamen hassasiyetten gelmiyor; recall ölçümü (aynı dikdörtgen giriş) daha doğrudan bir karşılaştırma.
3. **Karar: FP16** (ADR-020). Sistem metrikleri birebir aynı.

## Deney 11 · Python hattında uçtan uca gecikme (`eval/bench_latency.py`)

RTX 5050, FP16 motorlar, 600 gece video karesi (50 ısınma), kareler bellekte. Ultralytics ön/son işleme dahil, CUDA Graph yok. Milisaniye.

| Aşama | Bant her karede: p50 | p95 | Bant 2 karede bir: p50 | p95 |
|---|---|---|---|---|
| Tam kare | 9,6 | 15,3 | 6,7 | 10,9 |
| Bant (işlendiği karede) | 36,7 | 58,0 | 21,1 | 30,2 |
| Birleştir + takip + karar | 2,3 | 7,3 | 1,1 | 2,2 |
| Güvenlik izleyici | 3,1 | 6,9 | 1,8 | 2,7 |
| **Toplam** | **53,9** | **81,2** | **26,1** | **39,3** |

**Gözlemler:**
1. **REQ-P-06 (p95 ≤ 50 ms) Python'da yalnızca bant iki karede bir işlenirse karşılanıyor** (39,3 ms). 30 fps kare aralığına (33 ms) p95'te sığmıyor.
2. **Gecikmenin çoğu motor dışında.** Bant motoru 7,8 ms (Deney 9), aşama 36,7 ms: ~29 ms parça kesme/büyütme, Ultralytics'in 4 parça için CPU'da yaptığı letterbox ve tensör dönüşümü, FP32 aktarım ve son işlem. Tam karede motor 3,2 ms, aşama 9,6 ms.
3. **Güvenlik izleyicinin 3 ms'si tam çözünürlükte ortalama parlaklık hesabı.** Küçültülmüş gri görüntüde alınırsa ihmal edilir.
4. **Yüzdelikler geniş** (p99 ≈ 1,7 × p50): Python ve CUDA Graph'sız çekirdek başlatma, Deney 9'daki dalgalanmanın aynısı.
5. **Sonuç:** darboğaz model değil, Python'daki ön/son işleme. C++ hattında (CUDA'da ön işleme, 8 bit aktarım, CUDA Graph) motor süresine yakın bir toplam bekleniyor; bant zamanlaması kararı (ADR-016) C++ ölçümünden sonra. INT8 motor süresini kısaltır ama darboğazı çözmez.

## Deney 12 · GPU hattı: ön/son işleme GPU'da, CUDA Graph (`--backend trt`, ADR-021)

Aynı FP16 motorlar, aynı 600 gece video karesi (Deney 11 ile aynı ölçüm). Kare GPU'ya 8 bit bir kez gider; dolgu, bant parçaları, kutu çözümleme ve NMS GPU'da; motorlar Ultralytics'siz, TensorRT Python API'siyle. Milisaniye, bant her karede.

| Aşama · p95 | Ultralytics (Deney 11) | GPU hattı, CUDA Graph yok | **GPU hattı + CUDA Graph** |
|---|---|---|---|
| Tam kare | 15,3 | 13,7 | **6,0** |
| Bant | 58,0 | 25,7 | **13,7** |
| Birleştir + takip + karar + izleyici | 14,2 | 6,9 | **1,9** |
| **Toplam p50** | 53,9 | 29,2 | **16,2** |
| **Toplam p95** | 81,2 | 42,7 | **20,4** |
| Toplam p99 | 91,3 | 49,7 | **21,9** |

Bant iki karede bir: toplam p50 10,3, p95 19,6 ms.

**Doğruluk** (gece, 3.363 görüntü, güven ≥ 0,10; Ultralytics motor yolu / GPU hattı):

| | 166–249 m recall | Yanlış alarm / kare |
|---|---|---|
| Tam kare | 0,752 / 0,752 | 2,87 / 2,87 (birebir aynı) |
| Tam kare + bant | 0,850 / 0,851 | 4,27 / 4,30 |

**Gözlemler:**
1. **REQ-P-06 bant her karede işlenirken karşılanıyor:** p95 20,4 ms (hedef ≤ 50), 30 fps kare aralığına (33 ms) da sığıyor. Ortalama 16,6 ms, ~60 FPS (REQ-P-05: ≥ 25). Bu yüzden bant her karede işlenecek (ADR-016 kapandı).
2. **Kazancın çoğu ön/son işlemeyi GPU'ya taşımaktan ve CUDA Graph'tan.** Bant aşaması 58 → 13,7 ms. Uyarı: CUDA Graph'sız çalıştırmada GPU kullanmayan takip ve karar adımları da 2–3 kat yavaş çıktı; o çalıştırmada CPU saat hızı da düşüktü, CUDA Graph'ın payı tablodakinden biraz küçük olabilir.
3. **Tam kare çıktısı Ultralytics ile birebir aynı.** İlk ölçümde GPU hattı tam karede +%13 yanlış alarm verdi. `eval/compare_backends.py` adım adım ayırdı: ön işleme aynı (fark 2·10⁻⁵ gri seviye), CUDA Graph ham çıktısı aynı (fark 0), son işleme aynı çıktıdan aynı kutular. Fark Ultralytics `predict`'in NMS IoU eşiğindeydi: 0,5 (varsaydığımız 0,7 değil). Düzeltmeden sonra tüm hücreler aynı.
4. **Bantta kalan küçük fark** (yanlış alarm +%0,7, recall +0,001): parça büyütmede cv2'nin sabit noktalı yuvarlamasıyla en fazla 1 gri seviyelik fark (birim testte ölçüldü).
5. **Gecikme ölçümü NMS IoU 0,7 ile yapıldı.** Eşik değişikliği yalnızca NMS'e giren kutu sayısını etkiliyor (~50/kare), süreye etkisi ihmal edilebilir.
