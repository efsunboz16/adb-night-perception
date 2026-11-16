# Literatür Özeti

| Çalışma | Ne yapıyor | Projeye katkısı |
|---|---|---|
| Alcantarilla vd., SmartBeam, IEEE IV 2008 | Parlak leke + hale analizi + SVM ile otomatik uzun far. Farlar 300–500 m'den, arka lambalar 50–80 m'den algılanıyor | Mesafe hedefleri, hale özelliği, arka lamba zayıflığı |
| Pham & Yoo, Applied Sciences 2020 | Ufuk altı ROI, AdaBoost ile far/lamba/gürültü ayrımı, far çifti eşleştirme | Ufuk ROI ön bilgisi, a* kırmızılık özelliği |
| Saralajew vd., PVDN veri seti, 2021 | 59.746 gece görüntüsü. Araç görünmeden önceki dolaylı yansımalar da etiketli. Anahtar nokta etiketleme | Etiketleme yöntemi (ADR-001), ek veri seti |
| Ohnemus vd., Provident Vehicle Detection for ADAS, 2021 | Klasik aday + sığ CNN, YOLOv5s'i geçiyor. Seri sistemden ~1,6 s önce algılama | Hibrit mimari karşılaştırması |
| Optimized Loss Functions for Nighttime Vehicle Detection, 2020 | Negatiflere IoU ağırlıklı kayıp. Küçük nesne AP 24,7 → 31,4 | Faz 3 sokak lambası deneyi |
| 2PCNet, CVPR 2023 | Gündüz→gece alan uyarlaması, NightAug augmentasyonu | Faz 3 augmentasyon deneyi |
| Debiased Teacher, ICCV 2025 | Fiziğe dayalı gece dönüşümü. BDD100K gece mAP@0.5 50,9 (2PCNet 47,6) | Güncel referans değer |
| IA-YOLO, AAAI 2022 | Türevlenebilir görüntü iyileştirme + YOLO | Opsiyonel deney (gecikme maliyeti ölçülmeli) |
| SAHI, Akyon vd., ICIP 2022 | Dilimli çıkarım ile küçük nesne tespiti | Ufuk bandı dilimleme mimarisi |
| Greer, Trivedi vd., Patterns of Vehicle Lights, 2023 | Işık etiket temsilleri, mesafeye göre normalize metrik | Etiket kılavuzu, metrik tasarımı |
| NHTSA FMVSS 108 ADB kuralı, 2022 | Karşıdan gelen 15–220 m, öndeki 15–100 m test aralıkları | REQ-F-01, F-02 dayanağı |

**Tespit edilen boşluk:** ADB'ye özel derin öğrenme çalışmalarının çoğu tek bir doğruluk değeri raporluyor. Kamaştırma ihlali, kullanılabilirlik ve tepki süresi gibi sistem düzeyi değerlendirme ile gömülü gecikmeyi birlikte raporlayan açık bir çalışma bulunamadı. Bu proje o boşluğu hedefliyor.
