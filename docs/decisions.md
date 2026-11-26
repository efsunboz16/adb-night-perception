# Karar Kayıtları (ADR)

Her önemli teknik karar burada kısa olarak kayıt altına alınır: bağlam, karar, sonuç.

## ADR-001 · Uzak araçlar için ışık sınıfları ve anahtar nokta etiketleme
- **Bağlam:** 220 m'de araç gövdesi gece görünmez, far çifti ~8 px (bkz. requirements.md §3). PVDN çalışmasında uzmanlar arası kutu IoU uyumu sadece %52–59 çıkmış.
- **Karar:** `head_light`, `tail_light`, `street_light`, `reflector` sınıfları eklendi. Işıklar anahtar nokta (en parlak nokta) olarak etiketlenir, kutu otomatik üretilir.
- **Sonuç:** Kendi etiketleme işi gerekir (Faz 1, 3. hafta). Etiket uyumu ölçülür.

## ADR-002 · Donanım
- **Bağlam:** Jetson yok. Geliştirme makinesi: Windows, NVIDIA RTX 5050 (Blackwell, sm_120), 231 GB boş disk.
- **Karar:** Eğitim, TensorRT dönüşümü ve C++ çıkarım RTX 5050'de yapılır. TensorRT C++ API'si Jetson'da da aynı olduğundan `deploy/cpp/` kodu taşınabilir kalır.
- **Gömülü uygunluk nasıl gösterilecek:** Jetson ölçümü olmadan "Jetson'da 25 FPS" iddiası yapılmaz. Bunun yerine raporlananlar:
  1. RTX 5050'de TensorRT FP16/INT8 gecikmesi (uçtan uca, batch 1)
  2. Model başına GFLOPs ve parametre sayısı
  3. Aynı modelin Ultralytics'in yayınladığı Jetson Orin benchmark'larıyla karşılaştırması (kaynak gösterilerek, tahmin olarak işaretli)
- **Açık:** Faz 5 sonunda araştırma merkezinden veya Tetra kulübünden ödünç bir Jetson ile REQ-P-05/06 doğrulanır. Bulunamazsa README'de sınırlılık olarak yazılır.
- **Sonuç:** RTX 50 serisi PyTorch 2.7+ ve CUDA ≥ 12.8 derlemesi ister. CPU sürümü torch kaldırılıp CUDA'lı sürüm kurulmalı.

## ADR-003 · Eğitim verisi ayrımı
- **Karar:** BDD100K train → train/val (%90/%10, koşula göre katmanlı). Resmi BDD val → test. ACDC → sadece zor koşul testi, eğitimde kullanılmaz.
- **Gerekçe:** Test setine sızıntıyı önlemek ve gece/gündüz performansını ayrı raporlamak.

## ADR-004 · Lisans  *(AÇIK)*
- **Bağlam:** Ultralytics YOLO AGPL-3.0 lisanslı. BDD100K ve ACDC araştırma amaçlı lisanslar. PVDN lisansı kontrol edilecek.
- **Karar:** Portfolyo projesi olarak AGPL-3.0 uyumlu yayın. Ticarileşme düşünülürse dedektör Apache lisanslı bir alternatifle değiştirilir.

## ADR-005 · Eğitimde tüm koşullar, değerlendirme koşul bazında
- **Bağlam:** BDD train'de 25.226 gece, 28.213 gündüz karesi var. Kutuların %43'ü küçük (< 32×32 px). Motosiklet sadece 2.695 kutu (%0,35).
- **Karar:** Temel model tüm koşullarla eğitilir (gündüz verisi araç şeklini öğretir). Sonuçlar gündüz / gece / yağmurlu gece için ayrı raporlanır. "Sadece gece" eğitimi Faz 3'te karşılaştırma deneyi.
- **Risk:** Motosiklet sınıfı zayıf kalabilir. Sınıf bazında AP ayrıca izlenir.

## ADR-006 · Kendi etiketler için kare seçimi
- **Karar:** (ADR-007 ile 1.000 kareye güncellendi) Kareler sadece bizim train bölümündeki temiz gece karelerinden seçilir. Sahne kotası: otoyol %50, yerleşim %25, şehir caddesi %25. Karelerin %80'i en az bir uzak araç içerir, %20'si rastgeledir (negatif ışık örnekleri için). 100 kare çift etiketlenir.
- **Gerekçe:** ADB'nin asıl çalıştığı yer aydınlatmasız yollar. Test setine sızıntı olmaz.
- **Araç:** `scripts/select_label_frames.py`, kurallar `docs/labeling_guide.md`.

## ADR-007 · BDD gece etiketlerinde uzak araç eksikliği → hedefli etiketleme
- **Bulgu (`scripts/box_size_stats.py`, train):** Kare başına araç kutusu, gece/gündüz oranı: < 62 m %83–88 (gecenin doğal trafik oranı), 125–166 m %55, 166–249 m %37, > 249 m %29. Uzak mesafede gece etiket yoğunluğu, trafik oranının beklediğinden yaklaşık yarı yarıya düşük.
- **Yorum:** Gece uzak araçların bir kısmı etiketlenmemiş. Bunlar eğitimde arka plan sayılır ve modelin uzak ışıkları bastırmasına yol açar. Çekince: gece ve gündüz sahne dağılımı farkı da etkili olabilir.
- **Karar:** Kendi etiketleme 2.000'den 1.000 kareye indirildi, seçim genişliği < 16 px araç içeren karelere (~125 m+) yoğunlaştırıldı (ADR-006'yı günceller). Çift etiketlenen 100 karede BDD eksik etiket oranı doğrudan ölçülecek.

## ADR-008 · Elle etiketleme yerine BDD kutularına dayalı otomatik ışık etiketleri
- **Bağlam:** Elle nokta etiketleme kare başına 2–4 dakika sürdü (1.000 kare ≈ 30–60 saat). Ön-etiketler kare başına ~40 adayla çok gürültülüydü.
- **Karar:** Işık etiketleri otomatik üretilir (`scripts/autolabel_lights.py`). İnsan tarafından çizilmiş bir BDD araç kutusunun alt %75'indeki parlak leke o aracın ışığıdır: kırmızı halka → `tail_light`, değilse → `head_light`. Kutu dışında ve ufkun üstündeki leke → `street_light`. Kutu dışında ve ufkun altındaki lekeler etiketlenmez. Uygulandığı kareler: train/val gece ve alacakaranlık.
- **Kazanç:** ~1.000 elle etiketli kare yerine ~29.000 gece/alacakaranlık karesi, elle iş olmadan.
- **Sınırlar:**
  1. BDD'nin kaçırdığı uzak araçların ışıkları (ADR-007) bu yöntemle de etiketlenmez. Faz 3'te eğitilmiş modelin yüksek güvenli tespitleriyle ikinci tur sözde etiketleme (pseudo-labeling) yapılacak.
  2. `reflector` sınıfı otomatik üretilemiyor. Model reflektörleri arka plan olarak öğrenir.
  3. Test bölümünde ışık etiketi yok. Işık sınıflarının doğruluğu, ~150 karelik insan kontrollü bir setle ölçülecek (otomatik etiketleri onaylama/düzeltme, sıfırdan etiketleme değil).
- **Önceki karar:** ADR-006/007'deki elle etiketleme planının yerini alır. `data/own_lights` seçimi kontrol seti için kullanılacak.

## ADR-009 · Otomatik street_light etiketleri çıkarıldı
- **Bulgu:** Otomatik etiketleme train'de kare başına 8,9 head + 8,3 tail (araç başına ~2 ışık, makul) ama **65 street_light** üretti. Bunlar aydınlık pencereler, tabelalar ve şehir ışıklarını da içeriyor.
- **Karar:** `street_light` (sınıf 8) etiketleri silinir (`scripts/drop_class.py`), `autolabel_lights.py` varsayılan olarak üretmez. Sokak lambaları eğitimde etiketsiz kalır, YOLO bunları arka plan olarak öğrenir. Bu, negatif örnek vermenin zaten olağan yoludur.
- **Risk:** Park halindeki araçların kaportasındaki yansımalar da araç kutusunun içinde kaldığı için head/tail olarak etiketlenmiş olabilir. ~150 karelik kontrol setinde ölçülecek.

## ADR-010 · Yerel eğitim düzeni: gece ağırlıklı liste, küçük doğrulama, süre sınırı
- **Bağlam:** RTX 5050'de 1280 px'te ~19 görüntü/sn. Tam veri 63K kare → ~55 dk/epoch. Windows'ta dataloader işçileri disk belleği sınırına takılıyor (WinError 1455, worker exited).
- **Karar:**
  - Eğitim listesi 25.000 kare: %75 gece, %10 alacakaranlık, %15 gündüz (ADR-005'teki "tüm koşullar" ilkesi korunuyor ama ağırlık ADB'nin çalıştığı geceye kaydırıldı). Rastgele %40 alt küme ~10K gece karesi verirdi, bu düzen ~19K veriyor.
  - Eğitim sırasında doğrulama 1.500 karelik alt kümede yapılır, tam değerlendirme sonda test setinde koşul bazında yapılır.
  - Süre `--hours` ile sınırlanır (varsayılan 8 saat). İşçi sayısı RAM ve disk belleğine göre otomatik seçilir.
- **Etki:** Gündüz sonuçları rastgele örneklemeye göre biraz düşebilir. Bu bilinçli bir takas.

## ADR-011 · Ufuk bandı dilimleme: aday, koşullu kabul
- **Bulgu (Deney 2):** Gece 220 m bandında kutu+ışık recall'ı 0,75 → 0,85. Gecikme 15,9 → 48,5 ms (RTX 5050, optimize edilmemiş).
- **Karar:** Dilimleme sisteme aday olarak alındı. Kalıcı kabul için iki koşul: (1) kare başına yanlış alarm artışı kabul edilebilir olmalı, (2) gecikme, bant her iki karede bir işlenerek ve/veya TensorRT ile bütçe içine çekilmeli (hedef: toplam ≤ 30 ms, Jetson payı için).
- **Ayarlar:** `configs/slicing.yaml` (`scripts/horizon_band.py` üretir).

## ADR-012 · Uzak ışık ve araçlar için sözde etiketleme (Deney 3)
- **Bağlam:** Mesafe analizi (Deney 1b) recall'ın 125 m ötesinde düştüğünü gösterdi. BDD'nin uzak gece etiketleri eksik (ADR-007) ve otomatik ışık etiketleri yalnızca BDD kutularının içinden üretildiği için o araçların ışıkları da etiketsiz. Model uzak ışıkları bastırmayı öğreniyor.
- **Karar:** Temel model + ufuk bandı dilimleme öğretmen olarak kullanılır (`scripts/pseudo_label.py`). Train/val gece ve alacakaranlık karelerine, mevcut etiketlerle çakışmayan üç tür etiket eklenir:
  - **A:** BDD araç kutusunun alt %75'indeki ışık tahmini (güven ≥ 0,45)
  - **B:** Kutusuz ışık, yalnızca ufuk bandında, küçük ve aynı sınıftan bir eşiyle çift halindeyse. Far çifti eşleştirmesi, tek ışık sokak lambası olabileceği için eklenmez.
  - **C:** Etiketsiz uzak araç (genişlik < 32 px, güven ≥ 0,5)
- **Deney tasarımı:** Yeni model temel modelle birebir aynı ayarlarla eğitilir (Kaggle 2×T4, 11 saat, aynı 25.000 karelik liste). Tek fark etiketler. Değerlendirme aynı test seti ve aynı scriptlerle yapılır.
- **Risk:** Öğretmenin hataları etikete dönüşür (onay yanlılığı). Yüksek güven eşikleri ve çift şartı bunu sınırlamak için. Etiket kalitesi önizlemeyle gözle kontrol edilir.
- **Sonuç (2026-11-20):** Etki yok. Gece 220 m bandında recall tam karede 0,75 → 0,75, bantla 0,85 → 0,83 (belirsizlik ±1,5–2 puan). Yanlış alarm değişmedi. Sözde etiketler varsayılan eğitim hattına alınmadı. Ayrıntı: results.md, Deney 3.

## ADR-013 · Uzak mesafe için öncelik: model içi çözünürlük
- **Bağlam:** Deney 2 (dilimleme, +10 puan) ve Deney 3 (sözde etiket, +0) birlikte, uzak mesafe darboğazının etiket değil çözünürlük olduğunu gösteriyor.
- **Karar:** Sonraki recall deneyleri dilimlemenin kazancını modelin içine taşımayı hedefler: (1) P2 tespit kafası (stride 4) ve/veya (2) büyütülmüş bant parçalarıyla eğitim (SAHI ince ayarı). Başarı ölçütü: tam kare çıkarımla 220 m'de ≥ 0,85, yani dilimlemenin seviyesine gecikme bedeli olmadan ulaşmak.
- **Paralel:** Yanlış alarmı (~2,9/kare) düşürmek için VLM ile ışık etiketi doğrulama (Deney 5 adayı).
- **Uygulama (Deney 4):** `configs/models/yolo11s-p2.yaml`, YOLO11 boynuna stride-4 dalını ekler (katman 2 ile birleşim, P2–P5 dört çıkış). Ölçülen bedel: 9,46M → 9,58M parametre, 88,8 → 117,9 GFLOPs @1280 (+%33), eğitim belleği ~1,47 kat. Omurga yolo11s.pt'den aktarılır. DDP'de alt süreçler modeli yaml'dan yeniden kurduğu için ağırlıklar `pretrained=` ile verilir, `.load()` ile değil.

## ADR-014 · P2 kafası benimsenmedi; sıradaki test girişi büyütmek
- **Bağlam:** Deney 4'te P2 kafası (3 saat ince ayar) 220 m'de tam kare recall'ı değiştirmedi (0,75 → 0,74), hesap maliyetini ise %33 artırdı.
- **Karar:** Varsayılan model temel YOLO11s olarak kalır. P2 uzun eğitimle yeniden denenmeyecek, önce daha ucuz bir ayırt edici test yapılacak.
- **Sıradaki:** Eğitimsiz bir test. Temel model tam karede daha büyük girişle (imgsz 1600 / 1920) değerlendirilecek. Kazanç büyütmeden geliyorsa bu test dilimlemeye yakın recall verir. O zaman seçenekler: bant parçalarıyla ince ayar (SAHI eğitimi) veya yalnızca ufuk bandının büyütüldüğü tek geçişli giriş. Vermiyorsa kazanç parça bağlamından (dar görüş, daha az dikkat dağıtıcı) geliyordur.

## ADR-015 · Uzak mesafe için dilimleme kalıcı; maliyet takip ile düşürülecek
- **Bağlam:** Deney 4 (P2: +0) ve Deney 5 (2x tam kare büyütme: +4 puan, 4 kat maliyet) dilimlemenin +10 puanını tek geçişli modele taşıyamadı. Dilimleme eşit yanlış alarmda da daha iyi.
- **Karar:** Uzak mesafe algılaması ufuk bandı dilimlemesiyle yapılır. ADR-013'teki "kazancı model içine taşıma" hedefi bırakıldı. Bant parçalarıyla ince ayar (SAHI eğitimi) ileride, eğitim kotası olunca tekrar ele alınabilir, çünkü kanıt doğrudan parça görüşünü işaret ediyor.
- **Sonraki:** Faz 4. Takip (ByteTrack) ile (1) bandı her karede değil, iki karede bir veya yalnızca gerektiğinde işleyerek gecikmeyi düşürmek, (2) tek karelik yanlış alarmları kare tutarlılığıyla elemek. Eğitim gerektirmez, yerelde yapılır.

## ADR-016 · Takip yanlış alarm süzgeci değil; yanlış alarm görünüşle, titreme histerezisle çözülecek
- **Bağlam:** Deney 6'da takip yanlış alarmı yalnızca %9 azalttı, uzak recall'ı 11–12 puan düşürdü. Yanlış alarmların çoğu sahnede kalıcı ya da kesikli tekrar eden nesneler.
- **Karar:** Takipçi far kararı için tutuluyor (kimlik, hız, öngörü, bantsız karede uzak izi taşıma), ama tespit çıktısını süzmek için kullanılmıyor. Değerlendirmelerde "takip" ayrı mod olarak raporlanmaya devam ediyor.
- **Yanlış alarm için:** Görünüşe dayalı doğrulama (ışık/araç adayı kırpıntısını sınıflandıran küçük bir model; etiketleri VLM ile). Ayrıca ODD: ADB şehir içinde zaten devre dışı. Asıl hedef aydınlatmasız yol.
- **Titreme için:** Far kararı modülünde histerezis (`decision.release_delay_s`). Sistem metrikleri artık nesne düzeyinde değil, ADB düzeyinde ölçülecek: kamaşma ihlali (araç açık segmentte mi), uzun far kullanımı, dakikadaki segment değişimi.
- **Bant zamanlaması:** Karar ertelendi. 30 fps veri olmadan "iki karede bir" ile "her kare" ayırt edilemiyor.

## ADR-017 · Far kararı varsayılanları: takip, güven 0,35, histerezis 1,0 sn
- **Bağlam:** Deney 6d'de 4 güven eşiği × 3 histerezis süresi tarandı. Titremeyi belirleyen histerezis. Güven 0,5 uzak kamaşmayı artırıyor.
- **Karar:** `decision.release_delay_s` 0,5 → 1,0 sn. `min_confidence` 0,35 kalıyor. Far kararı takipçi çıktısıyla çalışır.
- **Sonuç:** Otoyolda titreme 1,2/dk (hedef ≤ 2), uzun far kullanımı 0,65. Kamaşma ihlali %1,6–3,2: hedef (≤ %1) karşılanmıyor. Sebepler ayrıştırılacak: tespit kaçırma mı, açı/paralaks hatası mı, güvenlik payı mı?
- **Sınırlama:** 5 fps veri. Histerezis saniye cinsinden olduğu için 30 fps'e taşınabilir, ama onay ve kayıp davranışı değişir. Etiket yalnızca videonun bir karesinde.

## ADR-018 · Bulanıklık / kirli cam kontrolü devre dışı
- **Bağlam:** Deney 8'de dört basit görüntü kalitesi ölçüsü iki bozuk videoyu (ıslak cam, sarsıntı) tutarlı şekilde ayırmadı. Uzama ölçüsü her birini ayrı ayarda yakaladı, ikisini birden değil. Pozitif örnek sayısı 2.
- **Karar:** `safety_monitor.blur_laplacian_min: null`. Kalibre edilmemiş bir eşikle uzun farı kapatmak, ya hiçbir şey yakalamaz ya da sağlam videolarda sebepsiz kısa fara düşer. Diğer iki kontrol (kare gecikmesi, aşırı parlaklık) kalıyor.
- **Sonuç:** REQ-S-02'nin bulanıklık kısmı ve kamaşma hedefi (ışıklı araç ≤ %1, ölçülen %1,3) açık. İleride: etiketli bozuk görüntülerle eğitilmiş kalite sınıflandırıcı veya araç sinyalleri (yağmur sensörü, silecek).
## ADR-019 · Gömülü çıkarım: iki sabit boyutlu TensorRT motoru, dikdörtgen giriş, CUDA Graph
- **Bağlam:** Faz 5. Tam kare ve ufuk bandı farklı boyutlarda çalışıyor. Ultralytics kare giriş kullanıyor (1280×1280); 720p karede piksellerin %42'si dolgu. Deney 9'da tam kare motorunun gecikmesi CPU tarafındaki çekirdek başlatma süresi yüzünden dalgalı çıktı (değişim katsayısı %32).
- **Karar:**
  1. İki ayrı motor: tam kare **736×1280, batch 1**; bant **704×768, batch 4** (352×384'lük 4 parça, 2x). Boyutlar `system.yaml` ve `slicing.yaml`'dan hesaplanır (`adb/models.py`).
  2. Motorlar ONNX'ten kendi derleyicimizle üretilir (`deploy/build_engine.py`), Ultralytics meta verisi dosya başına yazılır: değerlendirme scriptleri motorları `.pt` gibi açar.
  3. INT8 kalibrasyonu train bölümünden gece kareleriyle yapılır; bant motoru bant parçalarıyla kalibre edilir (test verisi kullanılmaz).
  4. C++ hattında çıkarım CUDA Graph ile başlatılır.
  5. Sabit boyutlu motora uymayan girdi sessizce yeniden ölçeklenmez, hata verir (uzak ışıklar ölçeklemeyle kaybolur).
- **Sonuç:** Tam kare p99 8,8 → 2,8 ms, algılama toplamı ~11 ms. Motor dosyaları GPU'ya bağlı: Jetson'da ONNX'ten yeniden derlenir. Ultralytics `val` dikdörtgen girişi desteklemediği için mAP ayrı bir kare (1280×1280) doğrulama motoruyla ölçülür. Hassasiyet seçimi (FP16 / INT8) doğruluk eşitliği ölçülünce ADR-020'de.

## ADR-020 · Hassasiyet: FP16; INT8 ertelendi
- **Bağlam:** Deney 10'da TensorRT FP16 motorları PyTorch ile karşılaştırıldı. 166–249 m recall 0,848 → 0,850 (bantlı), yanlış alarm aynı, video sistem metrikleri birebir aynı, mAP50 farkı ≤ 0,6 puan. Deney 11'de Python hattının gecikmesinin çoğu motor dışında çıktı (bant aşaması 36,7 ms, motor 7,8 ms).
- **Karar:** Dağıtım hassasiyeti **FP16**. INT8 şimdilik denenmiyor: motor süresi zaten bütçenin küçük bir kısmı, darboğaz ön/son işleme. INT8'in uzak ışıklar için riski (1–8 px'lik zayıf aktivasyonlar) bu kazanç için alınmaya değmez. Jetson gibi daha zayıf bir cihazda motor süresi darboğaz olursa yeniden açılır; derleyici hazır (`deploy/build_engine.py --precision int8`).
- **Sonuç:** Sıradaki adım C++ hattı: CUDA'da ön işleme, 8 bit aktarım, CUDA Graph. Bant zamanlaması (ADR-016) C++ ölçümüyle kapanacak.
