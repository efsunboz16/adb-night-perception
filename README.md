# ADB Night Perception

Kamerayla gece araç ve far/lamba algılama, ardından adaptif uzun far (ADB) segment kontrolü. Hedef: gömülü cihazda (Jetson) gerçek zamanlı çalışmak.

> **Durum:** Faz 0 — kapsam ve altyapı tamamlanıyor. Sonuçlar henüz yok.

## Ne yapıyor?

Uzun far açık kalır. Sistem yalnızca diğer yol kullanıcılarının bulunduğu açı dilimlerini karartır.

- **Girdi:** tek ön kamera (1280×720, 30 FPS)
- **Çıktı:** her kare için 16 segmentlik far maskesi

Asıl zorluk mesafe: 220 m'de karşıdan gelen bir araç gece sadece 8 piksellik iki ışık noktasıdır ([piksel bütçesi](docs/requirements.md#3-tasarımın-dayanağı-piksel-bütçesi)).

## Hedefler

| Metrik | Hedef | Sonuç |
|---|---|---|
| Karşıdan gelen araç algılama mesafesi | ≥ 220 m | — |
| Öndeki araç algılama mesafesi | ≥ 100 m | — |
| Kamaştırma ihlali oranı | ≤ %1 | — |
| Uzun far kullanılabilirliği | ≥ %80 | — |
| Kısma tepki süresi | ≤ 200 ms | — |
| Jetson FPS / uçtan uca gecikme | ≥ 25 / ≤ 50 ms | — |

Tam liste ve gerekçeler: [docs/requirements.md](docs/requirements.md)

## Mimari

Kamera → algılama → takip → geometri (piksel→açı, paralaks) → far kararı → LED matrisi. Ayrıca bir güvenlik izleyici var: belirsiz durumda sistemi kısa fara döndürür. Ayrıntı: [docs/architecture.md](docs/architecture.md)

## Repo yapısı

```
configs/   system.yaml (tüm parametreler), classes.yaml
adb/       geometri, takip, karar, güvenlik izleyici (Python referans)
scripts/   veri hazırlama, ortam kontrolü, piksel bütçesi
train/     model eğitimi
eval/      koşul bazlı ve fonksiyonel metrikler
deploy/    ONNX/TensorRT dönüşümü, C++ çıkarım
docs/      gereksinimler, mimari, kararlar (ADR), literatür
tests/     birim testler
```

## Başlangıç

```bash
pip install -r requirements.txt
python scripts/check_env.py      # donanım/ortam kontrolü
python scripts/pixel_budget.py   # gereksinim tablosunu üret
pytest -q                        # birim testler
```

## Yol haritası

| Faz | İçerik | Durum |
|---|---|---|
| 0 | Kapsam, gereksinimler, altyapı | 🔄 |
| 1 | Veri: BDD100K, ACDC, PVDN + kendi far/lamba etiketleri | ⏳ |
| 2 | Temel model ve koşul bazlı değerlendirme | ⏳ |
| 3 | Zor koşul iyileştirmesi (augmentasyon, kayıp, dilimleme) | ⏳ |
| 4 | Takip ve ADB karar mantığı | ⏳ |
| 5 | Jetson'da TensorRT + C++ dağıtım | ⏳ |
| 6 | Doğrulama, FMEA, dokümantasyon | ⏳ |

## Lisans ve veri

Kod lisansı ve veri setlerinin kullanım koşulları için bkz. [ADR-004](docs/decisions.md). Veri setleri repoya dahil değildir.
