# Gereksinim doğrulama matrisi

`eval/verify_requirements.py` ile üretildi. Yapılandırma: TensorRT FP16, GPU hattı, takip, tam kare + bant her karede (ADR-016, ADR-021). Sonuç klasörü: `runs/trt/full_fp16_gpu` (kopyaları: [data/](data/)). Gereksinimler: [requirements.md](requirements.md), riskler: [fmea.md](fmea.md).

**Özet:** ✅ GEÇTİ 8 · ❌ KALDI 4 · ⚠️ KISMEN 4 · ➖ ÖLÇÜLMEDİ 1 · ⛔ UYGULANMADI 1

| ID | Gereksinim | Hedef | Ölçülen | Durum | Kanıt |
|---|---|---|---|---|---|
| REQ-F-01 | Karşıdan gelen araç ≥ 220 m | recall ≥ 0,90 (166–249 m) | 0,851 (n=665; tam kare 0,752) | ❌ KALDI | `eval_distance.json` · gece test, güven ≥ 0,10, araç kutusu veya ışığı, bantlı (Deney 2, 12) |
| REQ-F-02 | Öndeki araç (arka lamba) ≥ 100 m | recall ≥ 0,90 (83–125 m) | 0,921 (n=5007) | ⚠️ KISMEN | Aynı ölçüm. Yön ayrımı yok: öndeki ve karşıdan gelen araçlar birlikte sayılıyor |
| REQ-F-03 | Sokak lambası / reflektörü araç sanma | negatif sınıflarda yanlış pozitif oranı | yanlış alarm 4,29 / kare (tüm kaynaklar) | ➖ ÖLÇÜLMEDİ | Sokak lambası ve reflektör etiketi yok (ADR-009). Yanlış alarmların %55–65'i kalıcı (Deney 6) |
| REQ-F-04 | Takip, ≤ 0,5 s kayıpta kimlik korunur | ID switch, HOTA | iz 3 kare sürdürülüyor: 5 fps'te 0,6 s, 30 fps'te 0,1 s | ⚠️ KISMEN | MOT etiketi yok, HOTA ölçülemedi. Parametreler kare cinsinden, 5 fps'e göre ayarlı (FM-08). ✅ 2 test: `test_coasting_small_box_output_one_frame`, `test_every_other_frame_detection_confirmed_and_kept` |
| REQ-F-05 | Paralaks düzeltmesi | birim test | – | ✅ GEÇTİ | ✅ 2 test: `test_parallax_vanishes_far_away`, `test_near_vehicle_parallax_widens_interval` |
| REQ-F-06 | Her yana 1° güvenlik payı | birim test | – | ✅ GEÇTİ | ✅ 1 test: `test_glare_violation_and_utilization` |
| REQ-F-07 | Histerezis: kaybolan hedefte segment bekleyip açılır | 0,5 s (ADR-017 ile 1,0 s) | 1,0 s | ✅ GEÇTİ | ✅ 2 test: `test_hysteresis_holds_then_releases`, `test_detection_dropout_keeps_segment_dimmed_for_release_time`. Süre titreme taramasıyla seçildi (Deney 6d) |
| REQ-F-08 | Takip hızıyla 50 ms ileriyi öngör | tepki gecikmesi telafisi | – | ⛔ UYGULANMADI | `decision.predict_ahead_ms` yapılandırmada var, kodda kullanılmıyor. Etkisi ve risk: [fmea.md](fmea.md) FM-05 |
| REQ-P-01 | Kamaşma ihlali | ≤ %1 | %0,8 (3/374 ışıklı araç) | ✅ GEÇTİ | `glare_diag/rates.json` · 68 gece videosu, mod takip·tam+bant; sebepler: düşük güven 2, takip onayı 1 |
| REQ-P-02 | Uzun far kullanımı | ≥ %80 | %63 | ❌ KALDI | `eval_video.json` · 20 otoyol + yerleşim gece videosu, takip·tam+bant |
| REQ-P-03 | Kısma tepki süresi | ≤ 200 ms | 54 ms = 1 kare onay (33 ms) + hat p95 20,4 ms | ✅ GEÇTİ | Tespit edildikten sonra; kaçırılan tespit (recall) ve LED sürücüsü hariç. ✅ 1 test: `test_req_p03_new_vehicle_dimmed_on_second_frame` |
| REQ-P-04 | Titreme | ≤ 2 / dk | 5,0 / dk | ❌ KALDI | Aynı ölçüm. 1 sn'den kısa açılma/kısılma (Deney 6). 5 fps veri |
| REQ-P-05 | Gömülü cihazda işlem hızı | ≥ 25 FPS | 60 FPS (NVIDIA GeForce RTX 5050 Laptop GPU) | ⚠️ KISMEN | `latency.json`. Jetson'da ölçülmedi (ADR-002); motorlar ONNX'ten yeniden derlenebilir |
| REQ-P-06 | Uçtan uca gecikme | ≤ 50 ms (p95) | p95 20,4 ms (p50 16,2, p99 21,9) | ✅ GEÇTİ | `latency.json` · bant her karede, kare bellekte (kamera yakalama hariç) (Deney 12) |
| REQ-S-01 | Kare 200 ms gelmezse kısa far | birim test | eşik 200 ms, bekçi | ✅ GEÇTİ | ✅ 3 test: `test_frame_gap_triggers_and_recovers_after_delay`, `test_watchdog_detects_missing_frames`, `test_req_s01_camera_freeze_dims_all_and_recovers` |
| REQ-S-02 | Aşırı parlak veya bulanık görüntüde kısa far | birim test + veri | parlaklık > 0,85: var · bulanıklık: kapalı | ⚠️ KISMEN | ✅ 2 test: `test_overexposed_frame_disables_high_beam`, `test_req_s02_overexposure_dims_all`. Bulanıklık için güvenilir eşik bulunamadı (Deney 8, ADR-018, FM-01) |
| REQ-S-03 | Belirsizlikte güvenli taraf: eşik altı tespit bölgesi kısılır | tasarım | eşik altı (0,35) tespit yok sayılıyor | ❌ KALDI | Bilinçli sapma: düşük eşik yanlış kısmayı ve titremeyi artırdı (Deney 6d, ADR-017, FM-04). ✅ 1 test: `test_low_confidence_vehicle_ignored_by_design` |
| REQ-S-04 | Bilinen sınırlamalar belgelenir | README | – | ✅ GEÇTİ | README › Bilinen sınırlamalar, docs/fmea.md |

Ölçümlerin sınırları: mesafeler kutu genişliğinden yaklaşık (60° HFOV varsayımı); video verisi 5 fps ve videoda tek etiketli kare; ışık sınıfı etiketleri otomatik; gecikme dizüstü GPU'da, prizde ve en iyi performans modunda. Ayrıntı: README › Bilinen sınırlamalar.
