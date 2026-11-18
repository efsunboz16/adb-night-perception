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
