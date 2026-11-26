# FMEA · Hata türleri ve etkileri analizi

Sürüm 1.0 · Faz 6 · Kapsam: algılama + takip + far kararı + güvenlik izleyici (docs/requirements.md §1). Far donanımı, LED sürücüsü ve araç ağı kapsam dışı. Her satırın kanıtı bir deney, birim testi veya karar kaydıdır. Doğrulama durumu: [verification.md](verification.md).

## Puanlama

Hata türlerinin önemi eşit değil (requirements.md §6): **hatalı açık bırakma** karşıdaki sürücüyü kamaştırır, **hatalı kısma** yalnızca görüşü azaltır. Şiddet bu ayrıma göre verildi.

| Puan | Şiddet (S) | Olasılık (O) | Tespit edilebilirlik (D) |
|---|---|---|---|
| 9–10 | Yakın mesafede kamaştırma, sistem fark etmiyor | Çoğu gece sürüşünde | Sistem hatayı hiç fark edemez |
| 7–8 | Orta mesafede kamaştırma | Sık (her saat) | Ancak dolaylı bir sinyalle fark edilir |
| 5–6 | Uzak mesafede kısa süreli kamaştırma | Ara sıra | İzleyici kısmen yakalar |
| 3–4 | Gereksiz kısma, görüş azalır | Nadir | İzleyici çoğunlukla yakalar |
| 1–2 | Fark edilmeyen konfor kaybı | Çok nadir | İzleyici her zaman yakalar, kısa fara döner |

RPN = S × O × D. Olasılıklar ölçülen oranlardan (veri setinde), tespit edilebilirlik güvenlik izleyicinin kapsamından verildi. Puanlar mühendislik yargısıdır; sıralama için kullanılır, mutlak risk değildir.

## Tablo (RPN'e göre)

| ID | Hata türü | Etki | Sebep | S | O | D | RPN | Mevcut önlem (kanıt) | Önerilen eylem |
|---|---|---|---|---|---|---|---|---|---|
| FM-01 | Bozuk görüntüde yakın/orta araç kaçırılır | Lambası yanan araç açık segmentte: kamaştırma | Islak/kirli ön cam, kamera sarsıntısı (hareket bulanıklığı) | 9 | 4 | 8 | **288** | Yok. Bulanıklık kontrolü güvenilir eşik bulunamadığı için kapalı (Deney 8, ADR-018). Işıklı araç ihlallerinin 4/5'i iki bozuk videodan (Deney 7) | Etiketli bozuk görüntülerle görüntü kalitesi sınıflandırıcısı; araç sinyalleri (yağmur sensörü, silecek) ile kısa fara dönüş |
| FM-02 | Kamera montajı / kalibrasyon hatası | Açı kayar, yanlış segment kısılır: kamaştırma | 60° HFOV varsayımı, montaj gevşemesi, yükleme ile eğim | 8 | 4 | 8 | **256** | 1° güvenlik payı (≈ 19 px). Geometri birim testli (REQ-F-05/06). Gerçek kamera kalibre edilmedi (requirements.md §7) | Kamera kalibrasyonu ve montaj kontrolü; ufuk/kaybolma noktası takibiyle çevrimiçi eğim kontrolü |
| FM-03 | Uzak karşı araç (166–249 m) kaçırılır | Uzak mesafede kamaştırma, araç yaklaşınca kısılır | 1–8 px ışık, BDD gece etiketlerinde uzak araç eksikliği (ADR-007) | 5 | 6 | 8 | **240** | Ufuk bandı dilimleme her karede (recall 0,75 → 0,85, Deney 2, 12), far/arka lamba sınıfları (ADR-001). REQ-F-01 karşılanmıyor (0,85 / 0,90) | İnsan etiketli uzak araç seti; ışık sınıflarının gerçek doğruluğunu ölçmek (README sınırlamaları) |
| FM-04 | Düşük güvenli gerçek araç yok sayılır | Kamaştırma | Far kararı eşiği 0,35 (REQ-S-03'ten bilinçli sapma) | 7 | 4 | 7 | **196** | Eşik taramayla seçildi; 0,5 uzak kamaşmayı artırdı, düşük eşik yanlış kısmayı ve titremeyi artırdı (Deney 6d, ADR-017). Lambası yanan araç ihlallerinin 2/5'i "düşük güven" (Deney 7). Test: `test_low_confidence_vehicle_ignored_by_design` | İki eşik: onaylı iz 0,35'te; 0,15–0,35 arası izler ancak N kare kalıcıysa kıssın |
| FM-05 | Yakın mesafede yanal hızlı geçen araç segmentin dışına kayar | Kısa süreli kamaştırma (yakın) | Öngörü yok (REQ-F-08 uygulanmadı); tepki 1 kare onay + hat gecikmesi ≈ 54 ms | 8 | 3 | 7 | **168** | Histerezis segmenti 1 sn kısık tutar (arkada kalan segment); 1° pay. Hesap aşağıda | REQ-F-08: takip hızıyla 50 ms öngörü veya yakın mesafede açıya bağlı pay |
| FM-06 | Mesafe tahmini hatası (kamyon/otobüs) | Paralaks düzeltmesi yanlış (yalnız yakında anlamlı) | Mesafe kutu genişliğinden, 1,8 m araç genişliği varsayımıyla | 6 | 4 | 6 | **144** | Paralaks 50 m üstünde < 1° (requirements.md §3); 1° pay. Deney 7'de geometri kaynaklı ihlal 0 | Sınıfa göre genişlik (kamyon 2,5 m) veya far çifti aralığından mesafe |
| FM-07 | Sokak lambası / yansıma araç sanılır | Gereksiz kısma: uzun far kullanımı düşer | Kalıcı parlak kaynaklar, etiketsiz uzak araçlar | 3 | 8 | 5 | **120** | Takip onayı tek karelik lekeleri eler (`test_single_frame_false_alarm_does_not_dim`); yanlış alarmların %55–65'i kalıcı, takiple elenemez (Deney 6, ADR-016) | Görünüşe dayalı doğrulama (ışık çifti, hareket tutarlılığı); REQ-P-02 açık |
| FM-08 | Takip parametreleri 30 fps'te kısa kalır | İz erken düşer; kısa kayıpta kimlik kaybı | Parametreler kare cinsinden, 5 fps veriyle ayarlandı (iz 3 kare: 5 fps'te 0,6 s, 30 fps'te 0,1 s) | 4 | 6 | 5 | **120** | Far kararında 1 sn histerezis izin düşmesinin etkisini kapatır (`test_detection_dropout_keeps_segment_dimmed_for_release_time`) | Takip parametrelerini saniye cinsine çevirmek; 30 fps gerçek kayıtla ayar |
| FM-09 | Işıksız, park etmiş araç açık segmentte | Kamaştırılacak sürücü yok | BDD park etmiş araçları da etiketliyor | 2 | 6 | 8 | **96** | Kamaşma oranı ışıklı/ışıksız ayrı raporlanıyor (Deney 7) | – (metrik ayrımı yeterli) |
| FM-10 | Aşırı pozlama / aydınlık şehir içi | Algılama güvenilmez | Karşıdan gelen güçlü far, aydınlatılmış cadde, gündüz | 6 | 4 | 3 | **72** | Ortalama parlaklık > 0,85'te kısa far (REQ-S-02; `test_req_s02_overexposure_dims_all`) | Şehir içi/aydınlık alan sınıflandırması (ODD dışı) |
| FM-11 | Gecikme taşması | Maske gecikir; hızlı değişen sahnede kamaştırma | Dizüstü güç yönetimi, CPU çekirdek başlatma dalgalanması, ısınma | 5 | 3 | 3 | **45** | CUDA Graph: p99 21,9 ms, en kötü 31,9 ms (Deney 12); 200 ms'yi aşarsa bekçi kısa fara döner | Hat içinde gecikme izleme; Jetson'da ölçüm |
| FM-12 | Kamera donar / kare gelmez | Eski maske uygulanmaya devam eder | Kablo, sürücü, işlemci tıkanması | 9 | 2 | 2 | **36** | Bekçi 200 ms'de kısa fara döner, 1 sn toparlanma (REQ-S-01; `test_req_s01_camera_freeze_dims_all_and_recovers`) | – |
| FM-13 | Motor açılmaz (farklı GPU / TensorRT sürümü) | Hat başlamaz | Motor dosyası GPU ve TensorRT sürümüne bağlı (ADR-019) | 6 | 2 | 1 | **12** | Açılış hatası açık mesajla durur; motor ONNX'ten yeniden derlenir. Hat başlamazsa uzun far hiç açılmamalı (entegrasyon kuralı) | Başlatmada motor/GPU uyumu kontrolü, hata durumunda kısa far |

## FM-05 hesabı: yanal hızlı geçen araç

Karşılıklı 130 km/s (bağıl 72 m/s), yanal mesafe 3,5 m. Araç z metre ilerideyken açısal hız ω = x·v / (x² + z²). Tepki 54 ms (1 kare onay + hat p95; verification.md REQ-P-03) boyunca açı kayması:

| Mesafe | Açı | Açısal hız | 54 ms'de kayma | 1° payı aşıyor mu |
|---|---|---|---|---|
| 15 m | 13,1° | 61°/s | 3,3° | evet |
| 30 m | 6,7° | 16°/s | 0,85° | hayır, sınırda |
| 50 m | 4,0° | 5,8°/s | 0,31° | hayır |
| 100 m | 2,0° | 1,4°/s | 0,08° | hayır |

Kayma yalnızca 30 m'nin altında payı aşıyor. O mesafede araç far kapsamının (±16°) kenarına yaklaşmış ve hızla dışına çıkıyor; araç geride kalan segmentlerden zaten geçmiş olduğu için histerezis bu segmentleri 1 sn kısık tutar. Risk, aracın hareket yönündeki komşu segmentte kısa süreli kamaştırma. Öngörü (REQ-F-08) bunu kapatır.

## Sonuç

En yüksek riskler algılamanın kendisinde, karar mantığında değil:
1. **Bozuk görüntü (FM-01)** sistemin fark edemediği tek yakın mesafe kamaştırma kaynağı. Ölçülen ışıklı araç ihlallerinin çoğunu açıklıyor.
2. **Kalibrasyon (FM-02)** ölçülmedi, çünkü gerçek kamera yok. Gerçek araçta ilk iş.
3. **Uzak recall (FM-03)** ve **düşük güven eşiği (FM-04)** kamaşma hedefini (≤ %1) belirliyor.

Geometri, histerezis ve güvenlik izleyici birim testlerle doğrulandı ve video değerlendirmesinde geometri kaynaklı ihlal görülmedi (Deney 7).
