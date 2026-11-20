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
- **Deney 4 tasarımı (P2 kafası):** `configs/yolo11s-p2.yaml`. YOLO11s'e backbone P2/4 çıktısı bağlanır, Detect dört ölçekte çalışır. Backbone ve ilk FPN katmanları `yolo11s.pt`'den aktarılır (297/593 öğe), yeni katmanlar rastgele başlar. Etiketler, eğitim listesi, süre (11 saat, Kaggle 2×T4) ve batch (16) temel modelle aynı. Değerlendirme aynı test seti ve scriptlerle.
  - **Maliyet (yerel ölçüm, 1280 px):** 9,46 → 9,58 M parametre, 88,8 → 117,9 GFLOPs (+%33). Yalnız ağ gecikmesi FP16 7,3 → 10,0 ms (RTX 5050). Bant dilimleme ~2 kat ekliyordu. Eğitim belleği +%52.
  - **Karıştırıcı:** Aynı süre sınırında P2 modeli ~%25 daha az epoch görür. Sonuç temel modelden kötü çıkarsa önce bu kontrol edilir (epoch sayısı raporlanır).
  - **Uygulama notu:** Çift GPU'da (DDP) Ultralytics modeli alt süreçte yaml'dan yeniden kurar. Ağırlıklar bu yüzden `.load()` ile değil `train(pretrained=...)` ile verilir, aksi halde model sıfırdan eğitilir.
- **Paralel:** Yanlış alarmı (~2,9/kare) düşürmek için VLM ile ışık etiketi doğrulama (Deney 4 adayı).