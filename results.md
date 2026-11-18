# Sonuçlar

## Deney 1 · Temel model (baseline)

- **Model:** YOLO11s, 1280 px, 9,4 M parametre
- **Eğitim:** Kaggle 2×T4, 11 saat, batch 16, 25.000 kare (%75 gece, ADR-010), BDD araç etiketleri + otomatik ışık etiketleri (ADR-008/009)
- **Test:** BDD100K val (bizim test bölümümüz), sadece araç/insan sınıfları (0–5). Işık sınıflarının test etiketi yok.
- **Tarih:** 2026-11-18

### Koşul bazında (test)

| Koşul | Görüntü | P | R | mAP50 | mAP50-95 | car | truck | bus | motorcycle | bicycle | person |
|---|---|---|---|---|---|---|---|---|---|---|---|
| day | 4435 | 0,696 | 0,542 | 0,609 | 0,374 | 0,841 | 0,616 | 0,628 | 0,396 | 0,487 | 0,685 |
| night | 3363 | 0,658 | 0,543 | 0,593 | 0,347 | 0,786 | 0,594 | 0,560 | 0,443 | 0,529 | 0,643 |
| night_adverse | 566 | 0,634 | 0,532 | 0,594 | 0,372 | 0,786 | 0,619 | 0,692 | 0,462 | 0,383 | 0,620 |
| dusk | 648 | 0,770 | 0,528 | 0,621 | 0,381 | 0,836 | 0,618 | 0,666 | 0,412 | 0,542 | 0,652 |
| day_adverse | 823 | 0,720 | 0,530 | 0,599 | 0,371 | 0,839 | 0,651 | 0,543 | 0,435 | 0,442 | 0,683 |

Hız: RTX 5050 Laptop, FP16, 1280 px: 6,5 ms çıkarım + 2,5 ms ön/son işlem (görüntü başına, batch 4).

### Işık sınıfları (eğitim sırası doğrulama, 1.500 kare)

| Sınıf | P | R | mAP50 |
|---|---|---|---|
| head_light | 0,613 | 0,636 | 0,644 |
| tail_light | 0,650 | 0,745 | 0,728 |

Dikkat: Doğrulama etiketleri de otomatik üretildiği için bu değerler gerçek doğruluğu değil, modelin otomatik etiketleyiciyi ne kadar taklit ettiğini ölçer. Gerçek doğruluk için insan kontrollü set gerekir.

### Gözlemler

1. Gece–gündüz farkı küçük: genel mAP50'de 1,6 puan, car sınıfında 5,5 puan. Yağmurlu gece normal geceyle aynı seviyede.
2. Recall her koşulda ~0,53–0,54 ile sınırlı, precision 0,63–0,77. Ana darboğaz recall. Koşuldan bağımsız olması, sorunun küçük/uzak araçlarda olduğunu düşündürüyor (doğrulanacak: mesafe bazlı recall).
3. Motosiklet en zayıf sınıf (AP50 0,40–0,46). Eğitim verisinde %0,35.
4. Literatürdeki BDD100K sonuçlarıyla doğrudan karşılaştırılamaz (farklı sınıf seti, bölüm ve çözünürlük).
