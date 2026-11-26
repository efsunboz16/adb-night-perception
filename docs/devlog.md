# Geliştirme günlüğü

Bu doküman projenin nasıl ilerlediğini anlatır: her fazda ne yapıldı, hangi sorunlarla karşılaşıldı, hangileri nasıl çözüldü, hangileri hâlâ açık. Sayısal sonuçlar [results.md](results.md), kararların gerekçeleri [decisions.md](decisions.md) içinde.

---

## Faz 0 · Kapsam ve altyapı

**Yapılanlar:** Problem tanımı ve piksel bütçesi hesabı. 220 m'de bir far çifti 1280 px'lik kamerada yalnızca ~8 piksel aralıklı iki nokta oluyor. Kimlik numaralı gereksinimler (REQ-F/P/S), sistem yapılandırması (`configs/system.yaml`), geometri modülü (piksel → açı, paralaks düzeltmesi, 16 segmentlik far maskesi) ve birim testleri yazıldı. Literatür taraması yapıldı (SmartBeam, PVDN, 2PCNet, SAHI).

| Sorun | Çözüm |
|---|---|
| Plan Jetson üzerine kuruluydu, ama donanım yok. Elde RTX 5050 dizüstü var. | Geliştirme dizüstünde yapılıyor. Gömülü performans iddiası Jetson'da ölçülene kadar yapılmıyor ([ADR-002](decisions.md)). |
| PyTorch'un CPU sürümü kurulmuştu, GPU görünmüyordu. | CUDA 12.8 sürümü (`--index-url .../cu128`) kuruldu. `scripts/check_env.py` ortamı tek komutla kontrol ediyor. |

## Faz 1 · Veri ve etiketleme

**Yapılanlar:** BDD100K → YOLO dönüşümü, koşul bazlı (gece / gündüz / alacakaranlık / yağmurlu) bölümler, kutu boyutu ve mesafe analizi, far/arka lamba etiketleri.

| Sorun | Çözüm |
|---|---|
| Elimizdeki BDD sürümü belgelerdeki tek JSON formatında değil, görsel başına ayrı JSON (2018 formatı) çıktı. | `bdd_prepare.py` iki formatı da okuyup ortak yapıya getiriyor. Etiket klasörünü kendisi buluyor. `--check` modu dosya yazmadan özet veriyor. |
| 70.000 görüntünün kopyalanması yarıda kesildi (pencere çöktü). | Kopya yerine hard link (anında, ek disk yok), ilerleme göstergesi, kaldığı yerden devam. |
| **BDD gece etiketlerinde uzak araçlar eksik.** 166–249 m'de gece etiket yoğunluğu gündüzün %37'si. | Bulgu olarak kaydedildi ([ADR-007](decisions.md)). Değerlendirmelerde sınırlama olarak belirtiliyor. Faz 3'te sözde etiketlemeyle ele alındı. |
| BDD'de far/arka lamba etiketi yok. Label Studio'da elle nokta etiketleme denendi: kare başına 2–4 dakika, 1.000 kare ≈ 30–60 saat. **Çok yorucu ve sürdürülemez.** | Otomatik etiketleme ([ADR-008](decisions.md)): parlak leke, insan çizimi bir araç kutusunun alt %75'indeyse o aracın ışığı sayılıyor. Kırmızılık arka lambayı ayırıyor. ~29.000 karede, elle iş olmadan. |
| Ön-etiket adayları gökyüzündeki ışıklarla dolup 40'lık sınıra takılıyordu. | Ufuk altı ve üstü için ayrı kota. |
| Otomatik sokak lambası etiketleri kare başına ~65 gürültülü kutu üretti. | Sınıf çıkarıldı ([ADR-009](decisions.md)). |

## Faz 2 · Temel model

**Yapılanlar:** YOLO11s, 1280 px, gece ağırlıklı 25.000 karelik eğitim listesi ([ADR-010](decisions.md)), koşul bazlı ve mesafe bazlı değerlendirme.

### Yerel eğitim denemesi (başarısız)

| Sorun | Denenen | Sonuç |
|---|---|---|
| CUDA bellek hatası (OOM) | batch 8 → 4 | Geçti, ama sonraki hataya takıldı |
| `WinError 1455` (disk belleği yetersiz), DataLoader işçileri çöküyor | İşçi sayısı düşürüldü. Eğitim öncesi RAM ve disk belleği ön kontrolü eklendi. | Kök neden bulundu: VS Code, Chrome ve WSL açıkken boş RAM yalnızca **1,1 GB**. |
| Bellek sorunu kullanıcı tarafında çözülemedi | – | **Çözülemedi.** Eğitim Kaggle'a taşındı. |

### Kaggle'a geçiş

| Sorun | Çözüm |
|---|---|
| `/kaggle/input` salt okunur, YOLO etiket klasörüne önbellek yazmak istiyor. | `/tmp` altında görüntülere sembolik bağlantı, etiketlere kopya. |
| Veri hazırlığı 27 dakika boyunca hiçbir çıktı vermeden sürdü, donmuş gibi görünüyordu. | Paralel hazırlık (ThreadPool) ve anlık ilerleme çıktısı. |
| Doğrulama görüntüleri beklenmedik şekilde BDD train klasöründeydi. Liste klasörü (`splits`) etiket klasörü sanılıyordu. | Arama mantığı ikisini de ayırt edecek şekilde düzeltildi. |
| **Ultralytics, bitmiş bir eğitimi "devam ettir" deyince hata vermek yerine sessizce örnek veri setiyle (coco8) yeni eğitim başlattı.** | Devam etmeden önce checkpoint'te optimizer durumu kontrol ediliyor. Yoksa net bir hata mesajı ve doğru komut gösteriliyor. |
| İndirilen `best.pt` dosyası klasör olarak açılmış geliyordu (`.pt` aslında zip arşivi). Elle zip'lemek işe yaramadı. | `scripts/repack_pt.py` klasörü doğru yapıda tek dosyaya paketliyor. Sonradan `import_kaggle_output.py` eklendi: Kaggle çıktı zip'ini açmadan içe aktarıyor, klasöre açılmış ağırlıkları otomatik paketliyor. |

### Değerlendirme

| Sorun | Çözüm |
|---|---|
| Mesafe değerlendirmesinde 3.000+ görüntü listesi Ultralytics'e verilince `MemoryError` | Liste bir metin dosyasına yazılıp kaynak olarak veriliyor, görüntüler akış halinde okunuyor. |

**Bulgu:** Genel recall'ın düşüklüğü (~0,53) uzak araçlardan geliyor. 220 m bandında araç kutusu recall'ı 0,28. Işık sınıfları bunu 0,75'e çıkarıyor, yani far/lamba sınıfları kararı doğrulandı.

## Faz 3 · Uzak mesafe

Hedef: gece 220 m'de recall 0,75 → 0,90. Dört yöntem denendi.

| Deney | Hipotez | Sonuç |
|---|---|---|
| 2 · Ufuk bandı dilimleme | Uzak araçlar ince bir yatay bantta toplanıyor. O bandı parçalara bölüp 2x büyütmek küçük nesneleri bulunur hale getirir. | ✅ **0,75 → 0,85.** Bedeli: yanlış alarm +%49, gecikme yaklaşık iki katı. |
| 3 · Sözde etiketleme | Eksik uzak etiketler modele "bunlar arka plan" diye öğretiyor. Öğretmen modelle etiketlemek düzeltir. | ❌ Etki yok (0,75 → 0,75). |
| 4 · P2 tespit kafası | Kazanç daha ince özellik haritasından geliyor. Bunu model içine almak dilimlemeyi gereksiz kılar. | ❌ Etki yok (0,74). Hesap +%33. |
| 5 · Tam kare büyütme | Kazanç sadece büyütmeden geliyor. | ⚠️ +4 puan (0,79), 4 kat maliyet. Kazancın ancak ~%40'ı. |

**Faz 3'ün çıkarımı:** Uzak araçları buldurtan şey uzak bölgeye ayrıca ve dar bir kadrajla bakmak. Muhtemel sebep: parçada yakındaki parlak farlar ve yansımalar yok, uzak ışıklar öne çıkıyor. Dilimleme kalıcı oldu ([ADR-015](decisions.md)). Başarısız üç deney de neyin işe yaramadığını ve nedenini gösterdi. Bu sayede ileride "daha fazla etiket" veya "daha büyük model" gibi pahalı yollara körlemesine girilmeyecek.

### Faz 3'te karşılaşılan sorunlar

| Sorun | Çözüm |
|---|---|
| Ufuk bandı scripti numpy sayısını YAML'a yazamadı. İlk bant modele bağlam bırakmayacak kadar inceydi. | `float()` dönüşümü. En az 160 px bant yüksekliği. |
| Dilimleme recall'ı artırdı, ama yanlış alarm hiç ölçülmüyordu. Kazanç bedava görünüyordu. | Yanlış alarm metriği eklendi: hiçbir gerçek nesneye düşmeyen tahmin, kare başına. Kazancın bedeli görünür oldu. |
| Sözde etiket önizlemesinde "kutu içi ışık" kuralının **park halindeki araç gövdelerindeki yansımaları** etiketlediği görüldü. | Kural varsayılan olarak kapatıldı. Kutusuz ışık çiftlerine parlaklık şartı eklendi (sönük çift: reflektör veya yol çivisi). Kurallar birim testlerle sabitlendi. |
| P2 modeli 2 GPU'da eğitilirken alt süreçler modeli YAML'dan yeniden kuruyor. Normal yoldan yüklenen ön ağırlıklar kayboluyor, model **sessizce sıfırdan** başlıyordu. | Ağırlıklar `pretrained=` argümanıyla veriliyor. Her süreç aynı aktarımı yapıyor. Log'daki `Transferred 297/593` satırıyla doğrulanıyor. |
| Kaggle'da haftalık GPU kotasının yalnızca 4 saati kalmıştı (11 saatlik plan). | P2, sıfırdan değil, temel modelin ağırlıklarından başlatıldı (3 saat). Sonuç bu sınırlamayla birlikte raporlandı. |
| Kaggle, yüklenen `best.pt`'yi yine klasöre açtı. Oturum yeniden başlayınca paketlenmiş dosya silindi. | Paketleme notebook'ta kurulumdan hemen sonraki hücreye kondu. Her çalıştırmada otomatik yapılıyor. |
| Farklı giriş boyutlarıyla yapılan değerlendirmeler önceki sonuç dosyasının üzerine yazacaktı. | Çıktı adına giriş boyutu ekleniyor (`eval_distance_2560.json`). |
| İlk büyütme testi (1920, 1,5x) dilimlemeyle (2x) aynı ölçekte değildi. Yanlış sonuç çıkarılabilirdi. | 2560 ile eşit ölçekte tekrarlandı. Karşılaştırma eşit yanlış alarm düzeyinde de yapıldı. |

## Açık sorunlar

- **Otomatik ışık etiketleri gürültülü:** yol yüzeyindeki far yansıması bazen tek, büyük bir ışık kutusu oluyor. Araç gövdesi yansımaları ışık sayılabiliyor. Etiket temizliği deneyi planlandı.
- **Işık sınıflarının gerçek doğruluğu ölçülmedi:** doğrulama etiketleri de otomatik üretildi. İnsan kontrollü küçük bir set gerekiyor. Elle etiketleme yükü nedeniyle açık kaynak bir VLM ile aday doğrulama düşünülüyor.
- **Gecikme ölçümleri kararsız:** aynı koşulda tam kare 16–44 ms. Dizüstü GPU güç yönetimi ve okuma süresi etkili. Güvenilir ölçüm Faz 5'e (TensorRT) bırakıldı.
- **Dilimlemenin gecikme bedeli:** Faz 4'te takip ile bandı iki karede bir işleyerek yarıya indirilmesi planlanıyor.
- **Dilimlemenin yanlış alarm bedeli (+%49):** Faz 4'te kare tutarlılığı şartıyla elenmesi planlanıyor.

## Faz 4 · Takip, far kararı, video değerlendirmesi

**Yapılanlar:** Uzak ve küçük nesnelere uyarlanmış takipçi, paralaks düzeltmeli ve histerezisli far kararı modülü, video üzerinde ADB sistem metrikleri (kamaşma ihlali, uzun far kullanımı, titreme), far kararı parametre taraması.

### Video verisi bulma

| Sorun | Çözüm |
|---|---|
| Takip için etiketli video gerekiyordu. BDD'nin MOT 2020 takip etiketleri indirme sayfasında yoktu. İndirilen "Labels" dosyası zaten elimizde olan görüntü etiketleri çıktı. | Etiket dosyasında `timestamp: 10000` görüldü: her etiketli görüntü kendi videosunun 10. saniyesi. Elimizdeki MOT kareleri aynı videolar çıktı. `prepare_video_eval.py` etiketli kareyi görüntü benzerliğiyle buldu: 189 videonun 152'sinde tam 51. kare. |
| Ham video arşivi 1,8 TB. "Video Parts" ve "Video Torrent" bağlantıları çalışmıyordu. | HTTP Range ile zip'ten tek tek video çeken bir script yazıldı. Sonra MOT kareleri yeterli olduğu için kullanılmadı ve kaldırıldı. |
| MOT videoları modelin eğitim görüntüleriyle aynı havuzdan geliyor (sızıntı riski). | Her video adı eğitim listesinde arandı. 189 videonun hepsi test setinden çıktı. |

### Takip

| Sorun | Çözüm |
|---|---|
| Sahte dedektörle yapılan testte, bant iki karede bir çalışınca uzak araç izi bantsız karede siliniyordu. | Aday izler bir kare görünmeden bekleyebiliyor (`cand_max_age=1`). Test eklendi. |
| İlk video çalıştırmasında takip yanlış alarmı **ikiye katladı** (3,75 → 7,87). | İki sebep bulundu: (1) eşleşemeyen yakın araç izleri tahmini konumla çıktıda kalıyor ve 5 fps'te aracın dışına düşüyordu ("hayalet" kutular); (2) iz eski yüksek güveni taşıyordu. v2'de görünmeyen iz yalnızca küçükse (uzak) 1 kare çıktıya veriliyor, çıktı güveni o karenin tespit güveni. |
| v2'de bile takip yanlış alarmı yalnızca %9 azalttı. | **Çözülemedi, ama sebebi ölçüldü:** yanlış alarmların %55–65'i önceki karede de aynı yerde. Sokak lambası, yansıma ve etiketsiz uzak araçlar zamansal süzgeçle elenemez. Görünüşe dayalı doğrulamaya bırakıldı (ADR-016). |

### Far kararı ve metrikler

| Sorun | Çözüm |
|---|---|
| Şehir içi gece karelerinde segmentlerin %82'si zaten kısık. Kamaşma ihlali her modda ~0, metrik modları ayırt etmiyor. | Sahne dizini (`scene_index.py`) ve sahne filtresi eklendi. ADB'nin asıl kullanıldığı otoyol ve yerleşim alanı ayrı ölçülüyor. |
| İlk titreme ölçüsü ("segment değişimi / dk") araç segmentler arasında kayarken oluşan doğal değişimleri de sayıyordu. Hedefle (≤ 2/dk) karşılaştırılamazdı. | Titreme yeniden tanımlandı: 1 sn'den kısa süren açılma veya kısılma. Test eklendi. |
| Kamaşma oranının paydasında far kapsamı (±16°) dışındaki araçlar da vardı. Oran olduğundan iyi görünüyordu. | Yalnızca kapsam içindeki araçlar sayılıyor. Test eklendi. |
| Histerezis 0,5 sn ile titreme otoyolda bile 24–34/dk. | Güven eşiği × histerezis taraması yapıldı. Histerezis 1,0 sn ile takipli sistemde otoyol titremesi 1,2/dk (ADR-017). |

### Kamaşma teşhisi ve güvenlik izleyici

| Sorun | Çözüm |
|---|---|
| Kamaşma ihlali %2, hedef ≤ %1. Sebebi bilinmiyordu. | `diagnose_glare.py`: her ihlal 5 sebep grubuna ayrıldı ve görseli çıkarıldı. **Geometri hatası 0 çıktı.** Far hesabı doğrulandı, tüm ihlaller tespit kaynaklı. |
| Görsellerde ihlallerin çoğu ışıkları kapalı, park etmiş araçlardı. Kamaşacak sürücü yok. | Araçlar "lambası yanan" ve "ışıksız" diye ayrıldı. İlk denemede tüm araçlar ışıksız çıktı: test etiketlerinde lamba sınıfı yoktu, bu kontrol edilmeden varsayılmıştı. Lamba görüntüden ölçüldü (kutunun alt %75'inde parlak nokta). Lambası yanan araçlarda oran %1,3. |
| Lambası yanan araç ihlallerinin 4/5'i iki bozuk videodan: ıslak ön cam, kamera sarsıntısı. | Güvenlik izleyicinin bulanıklık kontrolü için eşik veriden aranmaya çalışıldı (`calibrate_safety.py`). |
| Keskinlik ve parlak kenar ölçüleri bozuk videoları ayırmadı. Sarsıntılı video "en keskin" videolardan biri çıktı. | Hareket bulanıklığı ışıkları keskin kenarlı çizgilere çeviriyor. Işık lekesi uzaması ve hale oranı eklendi. Uzama her bozulmayı ayrı bir ayarda yakaladı, ikisini birden hiçbir ayarda yakalamadı. |
| Yalnızca 2 bozuk örnek var. Eşik seçilirse veriye uydurulmuş olur. | **Çözülemedi.** Bulanıklık kontrolü kapalı bırakıldı (ADR-018). Kare gecikmesi ve aşırı parlaklık kontrolleri yazıldı (`adb/safety.py`), arıza sonrası 1 sn toparlanma süresiyle. |

### Açık sorunlar (Faz 4 sonu)
- **Kamaşma ihlali, lambası yanan araçlarda %1,3, hedef ≤ %1.** Sebebi büyük ölçüde bozuk görüntü. Bozuk görüntü algılama çözülemedi.
- **Bantın sistem düzeyindeki etkisi ölçülemedi:** video setinde yalnızca 42 uzak araç var.
- **5 fps veri:** gerçek kamera 30 fps. Takip ve histerezis davranışı gerçek hızda tekrar ölçülmeli.
## Faz 5 · TensorRT, gecikme ölçümü

**Yapılanlar:** Windows'ta TensorRT 10.16 kurulumu, tam kare ve bant için iki sabit boyutlu FP16 motor, `trtexec` ile motor gecikmesi (Deney 9), değerlendirme scriptlerine motor desteği, kendi motor derleyicimiz ve INT8 kalibrasyonu, uçtan uca gecikme scripti.

### Kurulum (Windows, RTX 5050)

| Sorun | Çözüm |
|---|---|
| Blackwell (sm_120) eski TensorRT sürümleriyle çalışmıyor. | TensorRT 10.16.1 (CUDA 12.0–12.9 zip'i), CUDA Toolkit 12.8, Python'da `tensorrt-cu12` aynı sürüm. Zip ile Python sürümü farklı olursa birinin derlediği motoru diğeri açamaz. |
| `trtexec` bulunamadı. Kullanıcı PATH'inde klasör yerine `trtexec.exe` dosyasının kendisi, sistem PATH'inde eski sürümün olmayan klasörü yazılıydı. Yeni zip'lerde DLL'ler `lib` değil `bin` altında. | Doğru `bin` klasörü PowerShell profiline (`$PROFILE.CurrentUserAllHosts`) eklendi. `.venv\Scripts\Activate.ps1`'e eklemek denendi: dosyanın sonunda dijital imza bloğu var, imzadan sonra gelen satır betiği bozuyor; geri alındı. |

### Motorlar

| Sorun | Çözüm |
|---|---|
| Ultralytics'in ürettiği `.engine` dosyasını `trtexec --loadEngine` açamıyor. | Ultralytics dosyanın başına meta veri (4 bayt uzunluk + JSON) yazıyor. Gecikme ölçümü için motor `trtexec` ile ONNX'ten ayrıca derlendi. Kendi derleyicimiz `--raw` ile meta verisiz kopya da üretiyor. |
| Kare giriş (1280×1280) 720p'de piksellerin %42'sini dolguya harcıyor. | Tam kare motoru 736×1280. |
| Değerlendirme scriptleri tam kare ve bant için aynı modeli kullanıyordu. Sabit boyutlu motor bantta çalışamaz. | `--band-weights` eklendi (`adb/models.py`). Motor boyutu meta veriden okunuyor; uymayan girdi hata veriyor. |
| Ultralytics INT8 kalibrasyonu kare letterbox kullanıyor; dikdörtgen motorla ve bant parçalarıyla uyuşmuyor. | Kendi derleyicimiz (`deploy/build_engine.py`): kalibrasyon girdisi Ultralytics ön işlemesinin aynısı, bant motoru bant parçalarıyla kalibre ediliyor. |
| Ultralytics `val` dikdörtgen girişi tek sayıya indiriyor. | mAP için ayrı kare doğrulama motoru (`export.py --val`). |

### Gecikme

| Sorun | Çözüm |
|---|---|
| Faz 3–4'te gecikme ölçümleri aynı koşulda 16–44 ms arasında oynuyordu. Tam kare motorunda da değişim katsayısı %32 çıktı. | `trtexec` darboğazı gösterdi: GPU değil, CPU'nun çekirdekleri sıraya koyma süresi. CUDA Graph ile p99 8,8 → 2,8 ms, değişim %4. **Çözüldü:** C++ hattında CUDA Graph (ADR-019). |
| FP16 motorlar hızlı (algılama ~11 ms), ama Python hattının toplamı bant her karede p95 81 ms çıktı (Deney 11). | Aşama aşama ölçüm: bant aşaması 36,7 ms, motoru 7,8 ms. Fark CPU'daki parça hazırlama ve Ultralytics ön/son işlemesi. **Açık:** C++ hattında GPU'da ön işleme ile çözülecek. Şimdilik Python'da bant iki karede bir (p95 39 ms). |

### Doğruluk eşitliği

| Sorun | Çözüm |
|---|---|
| FP16'nın uzak ışıkları kaybettirip kaybettirmediği bilinmiyordu. | Aynı görüntülerde PyTorch ve FP16 (Deney 10): 220 m bandında fark 665 araçta en fazla 2 tespit, video sistem metrikleri birebir aynı. FP16 seçildi, INT8 ertelendi (ADR-020). |
| Ultralytics `val` dikdörtgen motoru desteklemediği için mAP kare doğrulama motoruyla ölçüldü; dolgu PyTorch'tan farklı. | mAP farkı (≤ 0,6 puan) bu yüzden yalnızca hassasiyete bağlanmadı; asıl karşılaştırma aynı dikdörtgen girişle yapılan recall ölçümü. |

### GPU hattı (ADR-021)

| Sorun | Çözüm |
|---|---|
| C++ hattı ~1–1,5 hafta ve Windows derleme ortamı gerektiriyordu; asıl ihtiyaç iki sayıydı (REQ-P-05/06). | C++'ın kazandıracağı işler Python'da yapıldı: kare GPU'ya 8 bit bir kez, dolgu ve bant parçaları GPU'da, motorlar TensorRT Python API'si ve CUDA Graph ile, NMS GPU'da (`adb/trt_runtime.py`). Bant her karede p95 81 → 20 ms. C++ gelecek çalışma. |
| GPU hattı tam karede +%13 yanlış alarm verdi; aynı motor, aynı görüntüler. | `eval/compare_backends.py` farkı adım adım ayırdı: ön işleme, CUDA Graph, son işleme aynıydı. Fark Ultralytics `predict`'in NMS IoU eşiğindeydi (0,5, varsayılan sanılan 0,7 değil). Eşitlenince tam kare çıktısı birebir aynı. Ders: "varsayılan" bir değeri kaynağından okumadan kopyalamamak. |

## Faz 6 · Doğrulama, FMEA, demo

**Yapılanlar:** gereksinim doğrulama matrisi (`eval/verify_requirements.py` → docs/verification.md), FMEA (docs/fmea.md), hat düzeyinde gereksinim ve hata enjeksiyonu testleri (`tests/test_requirements.py`), demo videosu üreticisi (`scripts/render_demo.py`).

| Sorun | Çözüm |
|---|---|
| Faz 4'ün sistem metrikleri (titreme 1,2/dk, kullanım 0,65) "takip · tam kare" modunda ölçülmüştü. Dağıtılan yapılandırma artık takip + tam kare + bant her karede. | Son ölçümler dağıtılan modda tekrarlanıyor (`takip·tam+bant`, GPU hattı). |
| `eval_video.py`'nin far kararı eşiği varsayılanı 0,25'te kalmıştı, sistem 0,35 kullanıyor (ADR-017). | Varsayılan artık `system.yaml`'dan okunuyor. |
| Doğrulama sırasında iki gereksinimin kodda karşılığı olmadığı görüldü: REQ-F-08 (50 ms öngörü) yapılandırmada var ama kullanılmıyor; REQ-S-03 (eşik altı tespit bölgesini kısma) ADR-017'deki eşik kararıyla çelişiyor. | Matriste "uygulanmadı" ve "kaldı" olarak, sebepleriyle yazıldı; riskleri FMEA'da (FM-05, FM-04) hesaplandı. Sapmayı belgeleyen test eklendi. |
| REQ-P-03 (kısma tepki süresi) video verisiyle ölçülemiyor: videoda tek etiketli kare var. | Tepki analitik olarak ayrıştırıldı: takip onayı (1 kare) + hat gecikmesi p95. Onay davranışı testle doğrulandı. |
| Takip parametreleri kare cinsinden ve 5 fps veriyle ayarlı; 30 fps'te iz 0,1 s'de düşer. | FMEA'ya yazıldı (FM-08). Far kararındaki 1 sn histerezis etkisini kapatıyor (testli). |
