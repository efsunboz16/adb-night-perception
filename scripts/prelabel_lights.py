"""Gece karelerinde ışık adaylarını bulup Label Studio ön-etiketi üretir (Faz 1 / p1i).

Klasik yöntem (SmartBeam / PVDN aday üretimi gibi):
  1. Parlak ve çevresine göre belirgin lekeler: gri >= --bright VE beyaz top-hat >= --contrast
     (top-hat, aydınlık vitrin gibi geniş parlak alanları bastırır, küçük ışık noktalarını tutar)
  2. Her lekenin en parlak pikseli = anahtar nokta
  3. Kaba sınıf tahmini:
       - lekenin çevresindeki halka kırmızıysa  -> tail_light
       - ufuk çizgisinin üstündeyse            -> street_light
       - diğerleri                             -> head_light   (reflektörleri sen düzelteceksin)
Bu sadece başlangıç önerisidir; her kareyi kontrol edip düzelt.

Kullanım:
  python scripts/prelabel_lights.py --preview 12          # önce eşikleri gözle kontrol et
  python scripts/prelabel_lights.py                       # Label Studio dosyalarını üret
Çıktı (data/own_lights/):
  label_studio/tasks_prelabeled.json   ön-etiketli 900 kare
  label_studio/tasks_double.json       çift etiket için 100 kare, ÖN-ETİKETSİZ (uyum ölçümü yanlı olmasın)
  label_studio/labeling_config.xml     proje şablonu
  preview/*.jpg                         (--preview ile)
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

CONFIG_XML = """<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true" brightnessControl="true"/>
  <KeyPointLabels name="kp" toName="image" strokeWidth="2">
    <Label value="head_light" background="#FFD400"/>
    <Label value="tail_light" background="#FF2D2D"/>
    <Label value="street_light" background="#2D9CFF"/>
    <Label value="reflector" background="#2DFF8F"/>
  </KeyPointLabels>
  <Choices name="uncertain" toName="image" perRegion="true">
    <Choice value="uncertain"/>
  </Choices>
</View>
"""
COLORS = {"head_light": (0, 212, 255), "tail_light": (45, 45, 255),
          "street_light": (255, 156, 45), "reflector": (143, 255, 45)}


def detect_all(img, a):
    import copy
    b = copy.copy(a)
    b.max_below, b.max_above = 10**6, 10**6
    return detect(img, b)


def detect(img: np.ndarray, a) -> list[dict]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (a.tophat_k, a.tophat_k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    bright = (gray >= a.bright).astype(np.uint8)
    # Geniş parlak bölgeler (vitrin, aydınlık tabela): top-hat ortalarını bastırır ama köşelerini
    # bırakır. Bu bölgelere değen adaylar elenir.
    nb, blab, bst, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
    big_ids = np.flatnonzero(bst[:, cv2.CC_STAT_AREA] > a.max_area)
    big_ids = big_ids[big_ids != 0]
    big = cv2.dilate(np.isin(blab, big_ids).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)

    mask = (bright.astype(bool) & (tophat >= a.contrast)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h_img = img.shape[0]
    horizon = a.horizon * h_img
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    out = []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area > a.max_area:
            continue
        pad = 6
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(img.shape[1], x + w + pad), min(h_img, y + h + pad)
        sub_lab = lab[y0:y1, x0:x1] == i
        sub_gray = np.where(sub_lab, gray[y0:y1, x0:x1], 0)
        py, px = np.unravel_index(np.argmax(sub_gray), sub_gray.shape)
        px, py = px + x0, py + y0
        if big[py, px]:
            continue
        ring = cv2.dilate(sub_lab.astype(np.uint8), ring_k).astype(bool) & ~sub_lab
        b, g, r = (img[y0:y1, x0:x1][ring].astype(np.float32).mean(axis=0)
                   if ring.any() else (0.0, 0.0, 0.0))
        redness = r - (g + b) / 2
        if redness >= a.red and r >= 90:
            cls = "tail_light"
        elif py < horizon:
            cls = "street_light"
        else:
            cls = "head_light"
        out.append({"x": int(px), "y": int(py), "cls": cls, "peak": int(gray[py, px]),
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "area": int(area), "redness": round(float(redness), 1)})
    # Kota: ADB için önemli olan ufkun altı. Üstteki sokak lambaları ayrı ve küçük kotayla tutulur;
    # böylece uzak ve sönük farlar, parlak sokak lambaları yüzünden listeden düşmez.
    below = sorted((d for d in out if d["y"] >= horizon), key=lambda d: -d["peak"])
    above = sorted((d for d in out if d["y"] < horizon), key=lambda d: -d["peak"])
    return below[:a.max_below] + above[:a.max_above]


def ls_task(name: str, w: int, h: int, cands: list[dict] | None, url_prefix: str) -> dict:
    task = {"data": {"image": f"{url_prefix}{name}"}}
    if cands:
        task["predictions"] = [{
            "model_version": "prelabel_v1",
            "result": [{
                "id": f"p{j}",
                "from_name": "kp", "to_name": "image", "type": "keypointlabels",
                "original_width": w, "original_height": h,
                "value": {"x": 100 * c["x"] / w, "y": 100 * c["y"] / h, "width": 0.25,
                          "keypointlabels": [c["cls"]]},
            } for j, c in enumerate(cands)],
        }]
    return task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/own_lights"))
    ap.add_argument("--bright", type=int, default=200, help="en parlak piksel eşiği (0-255)")
    ap.add_argument("--contrast", type=int, default=40, help="çevreye göre parlaklık farkı (top-hat)")
    ap.add_argument("--tophat-k", type=int, default=31, help="top-hat çekirdeği; bundan geniş lekeler bastırılır")
    ap.add_argument("--red", type=float, default=25, help="halka kırmızılık eşiği")
    ap.add_argument("--horizon", type=float, default=0.45, help="ufuk çizgisi, görüntü yüksekliğinin oranı")
    ap.add_argument("--max-area", type=int, default=3000)
    ap.add_argument("--max-below", type=int, default=25, help="ufkun altındaki en fazla aday")
    ap.add_argument("--max-above", type=int, default=5, help="ufkun üstündeki (sokak lambası) en fazla aday")
    ap.add_argument("--preview", type=int, default=0, help="N kare için çizim üret ve çık")
    ap.add_argument("--url-prefix", default="/data/local-files/?d=own_lights/images/",
                    help="Label Studio yerel dosya adresi (DOCUMENT_ROOT = proje/data)")
    a = ap.parse_args()

    rows = list(csv.DictReader((a.dir / "selection.csv").open(encoding="utf-8")))
    img_dir = a.dir / "images"

    if a.preview:
        pdir = a.dir / "preview"
        pdir.mkdir(exist_ok=True)
        for r in rows[:a.preview]:
            img = cv2.imread(str(img_dir / r["name"]))
            cands = detect(img, a)
            vis = img.copy()
            cv2.line(vis, (0, int(a.horizon * img.shape[0])), (img.shape[1], int(a.horizon * img.shape[0])),
                     (120, 120, 120), 1)
            for c in cands:
                cv2.circle(vis, (c["x"], c["y"]), 7, COLORS[c["cls"]], 1)
            cv2.imwrite(str(pdir / r["name"]), vis)
            n_raw = len(detect_all(img, a))
            print(f"  {r['name']}: {n_raw} ham aday -> {len(cands)} tutuldu  "
                  f"{dict((k, sum(c['cls'] == k for c in cands)) for k in COLORS)}")
        print(f"Önizlemeler: {pdir}  (sarı=head, kırmızı=tail, mavi=street, gri çizgi=ufuk)")
        return

    pre, dbl, n_c = [], [], 0
    for i, r in enumerate(rows, 1):
        img = cv2.imread(str(img_dir / r["name"]))
        if img is None:
            print(f"  okunamadı: {r['name']}")
            continue
        h, w = img.shape[:2]
        if r["double_label"] == "1":
            dbl.append(ls_task(r["name"], w, h, None, a.url_prefix))
        else:
            cands = detect(img, a)
            n_c += len(cands)
            pre.append(ls_task(r["name"], w, h, cands, a.url_prefix))
        if i % 100 == 0:
            print(f"\r  {i}/{len(rows)}", end="", flush=True)
    print()

    out = a.dir / "label_studio"
    out.mkdir(exist_ok=True)
    (out / "tasks_prelabeled.json").write_text(json.dumps(pre, ensure_ascii=False), encoding="utf-8")
    (out / "tasks_double.json").write_text(json.dumps(dbl, ensure_ascii=False), encoding="utf-8")
    (out / "labeling_config.xml").write_text(CONFIG_XML, encoding="utf-8")
    print(f"Ön-etiketli: {len(pre)} kare, ortalama {n_c / max(len(pre), 1):.1f} aday/kare")
    print(f"Çift etiket (ön-etiketsiz): {len(dbl)} kare")
    print(f"Yazıldı: {out}")


if __name__ == "__main__":
    main()
