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