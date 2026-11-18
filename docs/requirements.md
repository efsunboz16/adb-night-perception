# Sistem Gereksinimleri

Sürüm 0.1 · Faz 0 · Bu doküman projenin "bitti" tanımıdır. Her gereksinimin bir kimliği (ID) var. Değerlendirme scriptleri (`eval/`) ve README'deki sonuç tabloları bu kimliklere atıf yapar.

## 1. Problem tanımı

Gece sürüşünde uzun far görüşü yaklaşık iki katına çıkarır, fakat diğer yol kullanıcılarını kamaştırır. Bu yüzden sürücüler uzun farı az kullanır. Adaptif uzun far (ADB) sistemi uzun farı açık tutar ve **yalnızca diğer yol kullanıcılarının bulunduğu yatay açı dilimlerini** karartır.

Bu proje ADB'nin algılama ve karar kısmını gerçekleştirir:

- **Girdi:** tek ön kamera görüntüsü, 1280×720, 30 FPS
- **Çıktı:** her kare için 16 segmentlik uzun far maskesi (`True` = kıs)
- **Kapsam dışı:** gerçek far donanımı (çıktı LED matrisi simülasyonuyla gösterilir), dikey açı kontrolü, viraj aydınlatması

## 2. Çalışma alanı (ODD)

| Boyut | Kapsam içi | Kapsam dışı |
|---|---|---|
| Aydınlatma | gece, alacakaranlık | gündüz (sistem kapalı) |
| Hava | açık, yağmurlu | yoğun sis, kar |
| Yol | şehirlerarası yol, otoyol, aydınlatmasız şehir dışı | aydınlatılmış şehir içi (güvenlik izleyici kapatır) |
| Hız | 0–130 km/s | — |
| Hedefler | otomobil, kamyon, otobüs, motosiklet, bisiklet, yaya | hayvanlar |

## 3. Tasarımın dayanağı: piksel bütçesi

Kamera: 1280 px, 60° yatay görüş açısı → fx ≈ 1109 px, 1 piksel ≈ 0,05°. Tablo `scripts/pixel_budget.py` ile üretilmiştir.

| Mesafe (m) | Far çifti arası (1,5 m) | Araç genişliği (1,8 m) | Tek far/lamba (0,2 m) |
|---|---|---|---|
| 30 | 55 px | 67 px | 7 px |
| 100 | 17 px | 20 px | 2 px |
| 220 | 8 px | 9 px | 1 px |
| 400 | 4 px | 5 px | 0,6 px |

**Sonuçlar:**
- 220 m'de araç gövdesi gece görünmez, far çifti ise sadece **8 piksellik iki ışık noktasıdır**. Gövde tabanlı bir dedektör (BDD `car` sınıfı) bu mesafede çalışamaz. Bu durum `head_light` / `tail_light` sınıflarını ve anahtar nokta tabanlı etiketlemeyi zorunlu kılar.
- Tek bir lamba 220 m'de yaklaşık 1 piksel. Görünen şey lambanın kendisi değil, kameradaki parlama halesidir (blooming). Etiketleme ve augmentasyon bunu hesaba katmalıdır.
- Algılama ağına giriş çözünürlüğü küçültülürse (örneğin 640 px) bu ışıklar yarım piksele iner. Ufuk bandında tam çözünürlükte işlem (SAHI tarzı) bu yüzden gerekli olabilir. Faz 3'te ölçülecek.

**Paralaks:** Farlar kameradan ~1,8 m ileride ve ~0,7 m yanda. Kamerada 5° sağda görünen bir araç, sol fara göre:

| Mesafe | 15 m | 30 m | 50 m | 100 m | 220 m |
|---|---|---|---|---|---|
| Açı farkı | +3,7° | +1,7° | +1,0° | +0,5° | +0,2° |

Segment genişliği 2° olduğundan 50 m'nin altında düzeltme yapılmazsa yanlış segment kısılır. Bu da REQ-F-05 gereksinimini doğurur.

## 4. Fonksiyonel gereksinimler

| ID | Gereksinim | Doğrulama | Faz |
|---|---|---|---|
| REQ-F-01 | Karşıdan gelen aracı **≥ 220 m**'den algıla (hedef 400 m) | Mesafe bazlı recall ≥ 0,90 | 2–3 |
| REQ-F-02 | Öndeki aracı (arka lamba) **≥ 100 m**'den algıla | Mesafe bazlı recall ≥ 0,90 | 2–3 |
| REQ-F-03 | Sokak lambası ve reflektörlü tabelaları araç sanma | Negatif sınıflardaki yanlış pozitif oranı | 3 |
| REQ-F-04 | Her hedefi kareler boyunca takip et, kısa kayıplarda (≤ 0,5 s) kimliği koru | ID switch, HOTA | 4 |
| REQ-F-05 | Kamera açısını far açısına çevirirken paralaksı düzelt | Birim test (`tests/test_geometry.py`) | 0 / 4 |
| REQ-F-06 | Hedefin açısına her iki yanda **1°** güvenlik payı ekle | Birim test | 4 |
| REQ-F-07 | Hedef kaybolduktan sonra segmenti **0,5 s** bekleyip aç (histerezis) | Titreme metriği | 4 |
| REQ-F-08 | Takipten gelen hızla **50 ms** ileriyi öngör | Tepki gecikmesi | 4 |

## 5. Performans gereksinimleri

| ID | Gereksinim | Hedef |
|---|---|---|
| REQ-P-01 | Kamaştırma ihlali oranı: görüş alanındaki bir hedefin yanık segmentte kaldığı kare oranı | ≤ %1 |
| REQ-P-02 | Uzun far kullanılabilirliği: güvenli olduğu halde açık bırakılan segment oranı | ≥ %80 |
| REQ-P-03 | Kısma tepki süresi: hedef görünür → segment kısıldı | ≤ 200 ms |
| REQ-P-04 | Titreme: gereksiz açma-kapama | ≤ 2 / dakika |
| REQ-P-05 | Gömülü cihazda işlem hızı | ≥ 25 FPS (Jetson'da doğrulanacak; o zamana kadar ADR-002 vekil ölçümleri) |
| REQ-P-06 | Uçtan uca gecikme (kare yakalama → maske) | ≤ 50 ms (RTX 5050'de ölçülür) |

Referans olarak ABD'deki ADB kuralı (FMVSS 108) karşıdan gelen aracı düz yolda 15–220 m, öndeki aracı 15–100 m aralığında test ediyor. SAE J3069'un aniden beliren araç testi ise sisteme 2,5 s tepki süresi tanıyor. REQ-P-03 bilinçli olarak bundan çok daha sıkı tutuldu.

REQ-P-01 ile REQ-P-02 birbirine karşı çalışır. Her şeyi kısan bir sistem P-01'i geçer, P-02'de kalır. Sonuçlar her zaman ikisi birlikte raporlanır.

## 6. Güvenlik gereksinimleri

| ID | Gereksinim |
|---|---|
| REQ-S-01 | Kamera karesi 200 ms'den uzun gelmezse tüm uzun far kapanır |
| REQ-S-02 | Görüntü aşırı parlaksa (aydınlık şehir içi) veya bulanıksa (kirli lens) tüm uzun far kapanır |
| REQ-S-03 | Belirsizlikte güvenli taraf seçilir: güveni eşiğin altındaki bir tespit yok sayılmaz, o bölge kısılır |
| REQ-S-04 | Bilinen sınırlamalar (sis, ani virajlar, ışıksız hedefler) README'de belgelenir |

Hata türlerinin önemi eşit değildir. **Hatalı açık bırakma** (kamaştırma) karşıdaki sürücü için tehlikelidir. **Hatalı kısma** ise sadece konforu düşürür. Bu yüzden eşikler kısma lehine seçilir. Faz 6'daki FMEA tablosu bu ayrıma dayanır.

## 7. Varsayımlar ve açık konular

- Kamera parametreleri (60° HFOV) varsayımdır. BDD100K kameraları farklıdır. Gerçek kamera kalibre edilince `configs/system.yaml` güncellenir ve bu doküman yeniden üretilir.
- Far geometrisi (16 segment, ±16°) tipik matris LED farlardan alınmış bir varsayımdır.
- Mesafe etiketi veri setlerinde yoktur. Mesafe, kutu boyutundan ve far çifti aralığından yaklaşık olarak türetilir. Sınırlılık olarak raporlanır.
- Donanım: RTX 5050 dizüstü, Jetson doğrulaması sona bırakıldı (ADR-002).
