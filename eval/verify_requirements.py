"""Faz 6 · Gereksinim doğrulama matrisi (docs/requirements.md -> docs/verification.md).

Her gereksinim için: hedef, ölçülen değer, durum ve kanıt (sonuç dosyası veya birim testi).
Ölçümler dağıtılan yapılandırmadan okunur: TensorRT FP16, GPU hattı (--backend trt), takip,
tam kare + bant her karede (ADR-016, ADR-021).

Durumlar:
  GEÇTİ        hedef ölçüldü ve karşılandı
  KALDI        ölçüldü, karşılanmadı
  KISMEN       bir kısmı karşılandı veya yalnızca dolaylı kanıt var
  ÖLÇÜLMEDİ    veri yok (sebebi yazılır)
  UYGULANMADI  gereksinim kodda yok (sebebi yazılır)

Birim testler bu script içinden çalıştırılır (pytest, junit çıktısı).

Kullanım:
  python eval/verify_requirements.py
  python eval/verify_requirements.py --out docs/verification.md
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adb.geometry import load_config  # noqa: E402
from adb.tracking import Tracker  # noqa: E402

RUN = ROOT / "runs/trt/full_fp16_gpu"
MODE = "takip·tam+bant"            # dağıtılan yapılandırma

TESTS = {
    "REQ-F-04": ["tests/test_tracking.py::test_coasting_small_box_output_one_frame",
                 "tests/test_tracking.py::test_every_other_frame_detection_confirmed_and_kept"],
    "REQ-F-05": ["tests/test_geometry.py::test_parallax_vanishes_far_away",
                 "tests/test_decision.py::test_near_vehicle_parallax_widens_interval"],
    "REQ-F-06": ["tests/test_decision.py::test_glare_violation_and_utilization"],
    "REQ-F-07": ["tests/test_decision.py::test_hysteresis_holds_then_releases",
                 "tests/test_requirements.py::test_detection_dropout_keeps_segment_dimmed_for_release_time"],
    "REQ-P-03": ["tests/test_requirements.py::test_req_p03_new_vehicle_dimmed_on_second_frame"],
    "REQ-S-01": ["tests/test_safety.py::test_frame_gap_triggers_and_recovers_after_delay",
                 "tests/test_safety.py::test_watchdog_detects_missing_frames",
                 "tests/test_requirements.py::test_req_s01_camera_freeze_dims_all_and_recovers"],
    "REQ-S-02": ["tests/test_safety.py::test_overexposed_frame_disables_high_beam",
                 "tests/test_requirements.py::test_req_s02_overexposure_dims_all"],
    "REQ-S-03": ["tests/test_requirements.py::test_low_confidence_vehicle_ignored_by_design"],
}


def run_tests(nodes: list[str]) -> dict[str, bool]:
    with tempfile.TemporaryDirectory() as d:
        xml = Path(d) / "junit.xml"
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={xml}", *nodes],
                       cwd=ROOT, capture_output=True, text=True)
        if not xml.exists():
            return {n: False for n in nodes}
        res = {}
        for tc in ET.parse(xml).getroot().iter("testcase"):
            ok = not any(ch.tag in ("failure", "error", "skipped") for ch in tc)
            res[f"{tc.get('classname').replace('.', '/')}.py::{tc.get('name')}"] = ok
    return {n: res.get(n, False) for n in nodes}


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=RUN, help="sonuç klasörü (GPU hattı, FP16)")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/verification.md")
    a = ap.parse_args()

    cfg = load_config()
    tg = cfg["targets"]
    fps = float(cfg["camera"]["fps"])
    dist = load(a.run / "eval_distance.json")
    video = load(a.run / "eval_video.json")
    rates = load(a.run / "glare_diag/rates.json")
    lat = load(a.run / "latency.json")

    nodes = sorted({n for v in TESTS.values() for n in v})
    tests = run_tests(nodes)

    def tested(req):
        ns = TESTS.get(req, [])
        ok = all(tests[n] for n in ns)
        names = ", ".join(f"`{n.split('::')[1]}`" for n in ns)
        return ok, f"{'✅' if ok else '❌'} {len(ns)} test: {names}"

    rows = []

    import re

    def tr(x: str) -> str:                       # ondalık ayırıcı virgül
        return re.sub(r"(\d)\.(\d)", r"\1,\2", x)

    def row(req, text, target, measured, status, evidence):
        rows.append((req, text, tr(target), tr(measured), status, evidence))

    # ------------------------------------------------ fonksiyonel
    if dist:
        r = dist["results"]["0.1"]["night·bant"]
        rec = r["any"][1] / r["gt"][1]
        row("REQ-F-01", "Karşıdan gelen araç ≥ 220 m", f"recall ≥ {tg['recall_at_range']:.2f} (166–249 m)",
            f"{rec:.3f} (n={r['gt'][1]}; tam kare {dist['results']['0.1']['night·tam']['any'][1] / r['gt'][1]:.3f})",
            "GEÇTİ" if rec >= tg["recall_at_range"] else "KALDI",
            "`eval_distance.json` · gece test, güven ≥ 0,10, araç kutusu veya ışığı, bantlı (Deney 2, 12)")
        rec2 = r["any"][3] / r["gt"][3]
        row("REQ-F-02", "Öndeki araç (arka lamba) ≥ 100 m", f"recall ≥ {tg['recall_at_range']:.2f} (83–125 m)",
            f"{rec2:.3f} (n={r['gt'][3]})", "KISMEN" if rec2 >= tg["recall_at_range"] else "KALDI",
            "Aynı ölçüm. Yön ayrımı yok: öndeki ve karşıdan gelen araçlar birlikte sayılıyor")
        fp = r["fp"][0] / r["img"][0]
    else:
        row("REQ-F-01", "Karşıdan gelen araç ≥ 220 m", "recall ≥ 0,90", "–", "ÖLÇÜLMEDİ", "`eval_distance.json` yok")
        row("REQ-F-02", "Öndeki araç ≥ 100 m", "recall ≥ 0,90", "–", "ÖLÇÜLMEDİ", "`eval_distance.json` yok")
        fp = None
    row("REQ-F-03", "Sokak lambası / reflektörü araç sanma", "negatif sınıflarda yanlış pozitif oranı",
        f"yanlış alarm {fp:.2f} / kare (tüm kaynaklar)" if fp is not None else "–", "ÖLÇÜLMEDİ",
        "Sokak lambası ve reflektör etiketi yok (ADR-009). Yanlış alarmların %55–65'i kalıcı (Deney 6)")
    ok, ev = tested("REQ-F-04")
    mc = Tracker().max_coast
    row("REQ-F-04", "Takip, ≤ 0,5 s kayıpta kimlik korunur", "ID switch, HOTA",
        f"iz {mc} kare sürdürülüyor: 5 fps'te {mc / 5:.1f} s, 30 fps'te {mc / fps:.1f} s", "KISMEN",
        f"MOT etiketi yok, HOTA ölçülemedi. Parametreler kare cinsinden, 5 fps'e göre ayarlı (FM-08). {ev}")
    for req, text in (("REQ-F-05", "Paralaks düzeltmesi"), ("REQ-F-06", "Her yana 1° güvenlik payı")):
        ok, ev = tested(req)
        row(req, text, "birim test", "–", "GEÇTİ" if ok else "KALDI", ev)
    ok, ev = tested("REQ-F-07")
    rel = cfg["decision"]["release_delay_s"]
    row("REQ-F-07", "Histerezis: kaybolan hedefte segment bekleyip açılır", "0,5 s (ADR-017 ile 1,0 s)",
        f"{rel} s", "GEÇTİ" if ok else "KALDI", f"{ev}. Süre titreme taramasıyla seçildi (Deney 6d)")
    row("REQ-F-08", "Takip hızıyla 50 ms ileriyi öngör", "tepki gecikmesi telafisi", "–", "UYGULANMADI",
        "`decision.predict_ahead_ms` yapılandırmada var, kodda kullanılmıyor. Etkisi ve risk: [fmea.md](fmea.md) FM-05")

    # ------------------------------------------------ performans
    if rates:
        n, v = rates["lit"]["n"], rates["lit"]["violations"]
        row("REQ-P-01", "Kamaşma ihlali", f"≤ %{100 * tg['glare_violation_rate_max']:.0f}",
            f"%{100 * v / max(n, 1):.1f} ({v}/{n} ışıklı araç)",
            "GEÇTİ" if v / max(n, 1) <= tg["glare_violation_rate_max"] else "KALDI",
            f"`glare_diag/rates.json` · {rates['videos']} gece videosu, mod {rates['mode']}; "
            f"sebepler: {', '.join(f'{k} {c}' for k, c in rates['lit']['by_cause'].items()) or '–'}")
    else:
        row("REQ-P-01", "Kamaşma ihlali", "≤ %1", "–", "ÖLÇÜLMEDİ", "`glare_diag/rates.json` yok")
    if video:
        u = video["utilization"][MODE]
        row("REQ-P-02", "Uzun far kullanımı", f"≥ %{100 * tg['high_beam_utilization_min']:.0f}", f"%{100 * u:.0f}",
            "GEÇTİ" if u >= tg["high_beam_utilization_min"] else "KALDI",
            f"`eval_video.json` · {video['videos']} otoyol + yerleşim gece videosu, {MODE}")
        fl = video["toggles_per_min"][MODE]["histerezis"]
        row("REQ-P-04", "Titreme", f"≤ {tg['flicker_per_min_max']} / dk", f"{fl:.1f} / dk",
            "GEÇTİ" if fl <= tg["flicker_per_min_max"] else "KALDI",
            "Aynı ölçüm. 1 sn'den kısa açılma/kısılma (Deney 6). 5 fps veri")
    else:
        for req, text, t in (("REQ-P-02", "Uzun far kullanımı", "≥ %80"), ("REQ-P-04", "Titreme", "≤ 2 / dk")):
            row(req, text, t, "–", "ÖLÇÜLMEDİ", "`eval_video.json` yok")
    if lat:
        tot = lat["results"]["1"]["TOPLAM"]
        ok, ev = tested("REQ-P-03")
        frames = Tracker().min_hits - 1
        react = frames * 1000 / fps + tot["p95"]
        row("REQ-P-03", "Kısma tepki süresi", f"≤ {1000 * tg['dim_response_s_max']:.0f} ms",
            f"{react:.0f} ms = {frames} kare onay ({1000 / fps:.0f} ms) + hat p95 {tot['p95']:.1f} ms",
            "GEÇTİ" if ok and react <= 1000 * tg["dim_response_s_max"] else "KALDI",
            f"Tespit edildikten sonra; kaçırılan tespit (recall) ve LED sürücüsü hariç. {ev}")
        f = 1000 / tot["mean"]
        row("REQ-P-05", "Gömülü cihazda işlem hızı", f"≥ {tg['embedded_fps_min']} FPS",
            f"{f:.0f} FPS ({lat['gpu']})", "KISMEN",
            "`latency.json`. Jetson'da ölçülmedi (ADR-002); motorlar ONNX'ten yeniden derlenebilir")
        row("REQ-P-06", "Uçtan uca gecikme", f"≤ {tg['end_to_end_latency_ms_max']:.0f} ms (p95)",
            f"p95 {tot['p95']:.1f} ms (p50 {tot['p50']:.1f}, p99 {tot['p99']:.1f})",
            "GEÇTİ" if tot["p95"] <= tg["end_to_end_latency_ms_max"] else "KALDI",
            "`latency.json` · bant her karede, kare bellekte (kamera yakalama hariç) (Deney 12)")
    else:
        for req, text in (("REQ-P-03", "Kısma tepki süresi"), ("REQ-P-05", "İşlem hızı"), ("REQ-P-06", "Gecikme")):
            row(req, text, "–", "–", "ÖLÇÜLMEDİ", "`latency.json` yok")

    # ------------------------------------------------ güvenlik
    ok, ev = tested("REQ-S-01")
    row("REQ-S-01", "Kare 200 ms gelmezse kısa far", "birim test",
        f"eşik {cfg['safety_monitor']['max_frame_gap_ms']} ms, bekçi", "GEÇTİ" if ok else "KALDI", ev)
    ok, ev = tested("REQ-S-02")
    blur = cfg["safety_monitor"].get("blur_laplacian_min")
    row("REQ-S-02", "Aşırı parlak veya bulanık görüntüde kısa far", "birim test + veri",
        f"parlaklık > {cfg['safety_monitor']['max_mean_brightness']}: var · bulanıklık: "
        f"{'kapalı' if blur is None else blur}", "KISMEN" if ok else "KALDI",
        f"{ev}. Bulanıklık için güvenilir eşik bulunamadı (Deney 8, ADR-018, FM-01)")
    ok, ev = tested("REQ-S-03")
    row("REQ-S-03", "Belirsizlikte güvenli taraf: eşik altı tespit bölgesi kısılır", "tasarım",
        f"eşik altı ({cfg['decision']['min_confidence']}) tespit yok sayılıyor", "KALDI",
        f"Bilinçli sapma: düşük eşik yanlış kısmayı ve titremeyi artırdı (Deney 6d, ADR-017, FM-04). {ev}")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    row("REQ-S-04", "Bilinen sınırlamalar belgelenir", "README", "–",
        "GEÇTİ" if "## Bilinen sınırlamalar" in readme else "KALDI", "README › Bilinen sınırlamalar, docs/fmea.md")

    # ------------------------------------------------ çıktı
    rows.sort(key=lambda r: (r[0][4], int(r[0].split("-")[2])))
    from collections import Counter
    cnt = Counter(r[4] for r in rows)
    icon = {"GEÇTİ": "✅", "KALDI": "❌", "KISMEN": "⚠️", "ÖLÇÜLMEDİ": "➖", "UYGULANMADI": "⛔"}
    lines = [
        "# Gereksinim doğrulama matrisi",
        "",
        f"`eval/verify_requirements.py` ile üretildi. Yapılandırma: TensorRT FP16, GPU hattı, takip, tam kare + bant "
        f"her karede (ADR-016, ADR-021). Sonuç klasörü: `{a.run.relative_to(ROOT) if a.run.is_relative_to(ROOT) else a.run}`. "
        f"Gereksinimler: [requirements.md](requirements.md), riskler: [fmea.md](fmea.md).",
        "",
        "**Özet:** " + " · ".join(f"{icon[k]} {k} {cnt[k]}" for k in icon if cnt[k]),
        "",
        "| ID | Gereksinim | Hedef | Ölçülen | Durum | Kanıt |",
        "|---|---|---|---|---|---|",
    ]
    for req, text, target, measured, status, ev in rows:
        lines.append(f"| {req} | {text} | {target} | {measured} | {icon[status]} {status} | {ev} |")
    lines += ["", "Ölçümlerin sınırları: mesafeler kutu genişliğinden yaklaşık (60° HFOV varsayımı); video verisi "
              "5 fps ve videoda tek etiketli kare; ışık sınıfı etiketleri otomatik; gecikme dizüstü GPU'da, "
              "prizde ve en iyi performans modunda. Ayrıntı: README › Bilinen sınırlamalar.", ""]
    a.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    (a.run / "verification.json").write_text(json.dumps(
        [dict(zip(("id", "text", "target", "measured", "status", "evidence"), r)) for r in rows],
        indent=1, ensure_ascii=False), encoding="utf-8") if a.run.exists() else None
    print(f"\nYazıldı: {a.out}")


if __name__ == "__main__":
    main()
