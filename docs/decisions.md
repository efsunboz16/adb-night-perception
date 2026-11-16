# Karar Kayıtları (ADR)

Her önemli teknik karar burada kısa olarak kayıt altına alınır: bağlam, karar, sonuç.

## ADR-001 · Uzak araçlar için ışık sınıfları ve anahtar nokta etiketleme
- **Bağlam:** 220 m'de araç gövdesi gece görünmez, far çifti ~8 px (bkz. requirements.md §3). PVDN çalışmasında uzmanlar arası kutu IoU uyumu sadece %52–59 çıkmış.
- **Karar:** `head_light`, `tail_light`, `street_light`, `reflector` sınıfları eklendi. Işıklar anahtar nokta (en parlak nokta) olarak etiketlenir, kutu otomatik üretilir.
- **Sonuç:** Kendi etiketleme işi gerekir (Faz 1, 3. hafta). Etiket uyumu ölçülür.

## ADR-002 · Donanım  *(AÇIK)*
- **Seçenekler:** (a) Jetson Orin Nano: hedef platform. (b) NVIDIA GPU'lu dizüstü + TensorRT: geçici. (c) Sadece bulut GPU (Colab/Kaggle): eğitim için, gömülü ölçüm yok.
- **Karar:** _henüz verilmedi_. `python scripts/check_env.py` çıktısı buraya eklenecek.

## ADR-003 · Eğitim verisi ayrımı
- **Karar:** BDD100K train → train/val (%90/%10, koşula göre katmanlı). Resmi BDD val → test. ACDC → sadece zor koşul testi, eğitimde kullanılmaz.
- **Gerekçe:** Test setine sızıntıyı önlemek ve gece/gündüz performansını ayrı raporlamak.

## ADR-004 · Lisans  *(AÇIK)*
- **Bağlam:** Ultralytics YOLO AGPL-3.0 lisanslı. BDD100K ve ACDC araştırma amaçlı lisanslar. PVDN lisansı kontrol edilecek.
- **Karar:** Portfolyo projesi olarak AGPL-3.0 uyumlu yayın. Ticarileşme düşünülürse dedektör Apache lisanslı bir alternatifle değiştirilir.
