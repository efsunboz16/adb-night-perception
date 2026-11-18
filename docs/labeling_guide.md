# Etiket Kılavuzu: Gece Işıkları

> **Not (ADR-008):** Eğitim etiketleri artık `scripts/autolabel_lights.py` ile otomatik üretiliyor. Bu kılavuz sadece ~150 karelik insan kontrollü değerlendirme seti için geçerli.

Sürüm 0.1 · Faz 1 / 3. hafta · Kapsam: `data/own_lights/images` altındaki 1.000 gece karesi

## Neden kendi etiketimiz?

BDD100K gece etiketleri mesafe arttıkça seyreliyor: gündüze göre normalize edildiğinde 125–166 m'de etiket yoğunluğu %55'e, 166–249 m'de %37'ye düşüyor (ADR-007). Etiketsiz uzak araçlar eğitimde arka plan sayılır ve model onları bastırmayı öğrenir. Bu etiketler o boşluğu doldurur, ayrıca sistemi yanıltan ışıkları (sokak lambası, reflektör) da öğretir.

## Yöntem: anahtar nokta, kutu değil

Her ışık için **tek bir nokta** işaretlenir: ışık lekesinin **en parlak noktası**. Kutular sonradan script ile otomatik üretilir.

Gerekçe: Bir ışık lekesinin sınırı belirsizdir (parlama, hale). PVDN çalışmasında uzmanlar arası kutu uyumu IoU %52–59'da kalmış. Nokta ise hızlı işaretlenir ve tutarlıdır.

## Sınıflar

| Sınıf | Ne | Örnek |
|---|---|---|
| `head_light` | Karşıdan gelen aracın farı (beyaz/sarımsı) | Karşı şeritteki araç, uzaktaki iki beyaz nokta |
| `tail_light` | Öndeki aracın arka/fren lambası (kırmızı) | Aynı yöndeki araç |
| `street_light` | Sokak/yol aydınlatması, trafik ışığı, bina ışığı | Direk tepesindeki lamba, ufkun üstündeki ışıklar |
| `reflector` | Kendi farımızı yansıtan yüzey | Tabelalar, şerit reflektörleri, bariyer reflektörleri, plakalar |

## Kurallar

1. **Her ışık ayrı işaretlenir.** Bir aracın iki farı = iki `head_light` noktası.
2. **Birleşmiş ışıklar:** Çok uzaktaki aracın iki farı tek leke halinde birleşmişse **tek nokta** koy. Sonra iki ayrı noktaya ayrılabildiği mesafeyi not etmek değerli bir bulgu olur.
3. **Minimum boyut yok.** 1–2 piksellik ışık noktası bile işaretlenir. Bu projenin özü tam olarak bu uzak ışıklar.
4. **Yansımalar:** Araç ışığının ıslak yoldaki yansıması **işaretlenmez**. Kafa karıştırır ve ADB kararında işe yaramaz.
5. **Emin değilsen:** Noktayı koy ve `uncertain` işaretini ekle. Bu örnekler eğitimde kullanılmaz ama değerlendirmede ayrıca raporlanır.
6. **Kısmen kesik ışıklar:** Görüntü kenarında yarım görünen ışıklar işaretlenir.
7. **Sınırlar:** `street_light` ve `reflector` için sadece ufuk çizgisinin altındaki ve yola yakın olanlar zorunludur. Gökyüzündeki ve uzak binalardaki ışıklar isteğe bağlıdır.

## Araç

**Label Studio** (yerel, Windows'ta pip ile kurulur):

```powershell
pip install label-studio
label-studio start
```

Proje şablonu: *Computer Vision → Keypoint Labeling*. Etiketler: `head_light`, `tail_light`, `street_light`, `reflector`. Ek seçenek: `uncertain`.

## Ön-etiketleme (hız için)

Etiketleme el ile sıfırdan yapılmaz. Önce bir model ön-etiket üretir, sen sadece düzeltirsin. Ön-etiketleme scripti ayrıca hazırlanacak (`scripts/prelabel_lights.py`). Düzeltme süresini kaydet: rapor için değerli bir rakam olur.

## Kalite kontrol

- `selection.csv` dosyasında `double_label=1` olan **100 kare iki kez etiketlenir**: aralarına en az bir gün koy, ilk etiketlere bakmadan.
- **BDD eksik etiket oranı:** Bu 100 karede `head_light`/`tail_light` noktalarının kaçının bir BDD araç kutusunun içinde veya 5 px yakınında olduğu sayılır. Kutusu olmayan ışık çiftleri = BDD'nin kaçırdığı araçlar.
- Uyum ölçüsü: aynı sınıfta, birbirine **≤ 3 piksel** uzaklıktaki noktalar eşleşmiş sayılır. Eşleşme oranı raporlanır.
- Her 500 karede bir, 20 rastgele kare gözden geçirilip kural ihlali sayılır.
