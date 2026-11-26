"""Faz 4 · Video üzerinde sistem değerlendirmesi: takip, bant zamanlaması, maske titremesi.

Veri: scripts/prepare_video_eval.py'nin ürettiği data/bdd_video/videos.json. Her video test setimizdeki
bir görüntünün videosu; etiketli kare (key) yaklaşık 10. saniye. Her videoda key'den önceki
--warmup kare + key işlenir (5 fps). Ölçüm key karede gerçek etiketlerle yapılır.

Modlar (aynı tespitlerden, model her kare için bir kez çalışır):
  tam                 tek kare, tam görüntü (Deney 1b ile aynı)
  tam+bant            tek kare, tam + ufuk bandı (Deney 2 ile aynı)
  takip·tam           takip, her karede tam görüntü
  takip·tam+bant      takip, her karede tam + bant
  takip·bant/2·var    takip, bant iki karede bir; key karede bant VAR
  takip·bant/2·yok    takip, bant iki karede bir; key karede bant YOK (kötü durum)

Ölçütler:
  recall (kutu / kutu+ışık), mesafe aralıklarına göre, key karede
  yanlış alarm / kare, key karede
  maske: kısık segment oranı ve segment değişimi / dk (titreme), tüm karelerde
  süre / kare: tam + bant × (işlenme oranı) + takip

Kullanım:
  python eval/eval_video.py --weights runs/baseline/weights/best.pt
  python eval/eval_video.py --weights deploy/models/full_fp16.engine --band-weights deploy/models/band_fp16.engine
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from adb.decision import BeamController, glare_and_utilization  # noqa: E402
from adb.safety import SafetyMonitor  # noqa: E402
from adb.geometry import load_config  # noqa: E402
from adb.tracking import GROUPS, Tracker, _grow, iou_matrix  # noqa: E402
from eval_distance import BINS, LIGHTS, VEH, accumulate, new_stats, read_gt  # noqa: E402

MODES = ["tam", "tam+bant", "takip·tam", "takip·tam+bant", "takip·bant/2·var", "takip·bant/2·yok"]
SUMMARY_BINS = {"< 62 m": (5, 7), "62–125 m": (3, 5), "125–249 m": (1, 3), "249 m+": (0, 1)}


def resolve(p: str) -> Path:
    q = Path(p.replace("\\", "/"))
    return q if q.is_absolute() and q.exists() else ROOT / q


def short_runs(masks, max_frames):
    """Titreme: iki yanında farklı durum olan ve max_frames kareden kısa süren segment durumları
    (kısa açılıp yeniden kısılma veya kısa kısılıp yeniden açılma). Araç segmentler arasında
    kayarken oluşan doğal değişimler sayılmaz."""
    if len(masks) < 3:
        return 0
    m = np.array(masks, bool)                        # (kare, segment)
    n = 0
    for s in range(m.shape[1]):
        col = m[:, s]
        ch = np.flatnonzero(col[1:] != col[:-1]) + 1  # durum değişim noktaları
        for a, b in zip(ch, ch[1:]):                  # iki değişim arası = iki yanı sınırlı bir durum
            if b - a < max_frames:
                n += 1
    return n


def fp_mask(xyxy, cls, gt_all, margin=4.0):
    """eval_distance.false_alarms ile aynı tanım, ama hangi kutuların yanlış alarm olduğunu döndürür."""
    sel = np.isin(cls, VEH + LIGHTS)
    if len(gt_all) == 0:
        return sel
    cx, cy = (xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2
    inside = ((cx[:, None] >= gt_all[None, :, 0] - margin) & (cx[:, None] <= gt_all[None, :, 2] + margin) &
              (cy[:, None] >= gt_all[None, :, 1] - margin) & (cy[:, None] <= gt_all[None, :, 3] + margin))
    return sel & ~inside.any(axis=1)


def persistent(box, c, prev_dets, min_size=16.0, thr=0.2):
    """Kutu, önceki karelerin her birinde aynı yerde (aynı sınıf grubu) tespit edilmiş mi?"""
    for xyxy, cls, _ in prev_dets:
        same = np.array([GROUPS.get(int(k), 3) == GROUPS.get(int(c), 3) for k in cls], bool)
        if not same.any():
            return False
        if iou_matrix(_grow(box[None], min_size), _grow(xyxy[same], min_size)).max() < thr:
            return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".pt veya tam kare .engine")
    ap.add_argument("--band-weights", help="bant motoru (.engine); .pt için gerekmez")
    ap.add_argument("--videos", type=Path, default=ROOT / "data/bdd_video/videos.json")
    ap.add_argument("--data", type=Path, default=ROOT / "data/bdd_adb")
    ap.add_argument("--conds", nargs="+", default=["night", "night_adverse"])
    ap.add_argument("--warmup", type=int, default=15, help="key'den önce işlenecek kare (5 fps: 15 = 3 sn)")
    ap.add_argument("--conf", type=float, nargs="+", default=[0.1, 0.25])
    ap.add_argument("--mask-conf", type=float, default=0.25, help="far kararı için güven eşiği")
    ap.add_argument("--release", type=float, default=None, help="histerezis süresi (sn); boş: system.yaml")
    ap.add_argument("--sweep-conf", type=float, nargs="+", help="far kararı güven eşiği taraması, ör. 0.15 0.25 0.35 0.5")
    ap.add_argument("--sweep-release", type=float, nargs="+", help="histerezis taraması (sn), ör. 0.5 1 2")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scenes", nargs="+", help="ör. highway residential (scripts/scene_index.py gerekir)")
    ap.add_argument("--flicker-s", type=float, default=1.0, help="bundan kısa süren segment durumu titreme sayılır")
    a = ap.parse_args()

    import cv2
    from adb.models import load_detectors, run_dir_for
    from adb.slicing import load_slicing, merge, predict_band, predict_full

    vids = {k: v for k, v in json.loads(a.videos.read_text(encoding="utf-8")).items() if v["cond"] in a.conds}
    if a.scenes:
        sf = a.data / "scenes.json"
        if not sf.exists():
            sys.exit(f"{sf} yok. Önce: python scripts/scene_index.py")
        sc = json.loads(sf.read_text(encoding="utf-8"))
        vids = {k: v for k, v in vids.items() if sc.get(k) in a.scenes}
    if a.limit:
        vids = dict(list(vids.items())[:a.limit])
    if not vids:
        sys.exit(f"{a.videos} içinde {a.conds} videosu yok.")
    cfg = load_slicing(ROOT / "configs/slicing.yaml")
    sysc = load_config()
    model, band_model = load_detectors(a.weights, a.band_weights, a.imgsz)
    print(f"Tam kare: {model.describe()} | bant: {band_model.describe()}")
    min_conf = min(a.conf)
    print(f"{len(vids)} video ({', '.join(a.conds)}), video başına {a.warmup + 1} kare")

    stats = {m: new_stats(a.conf) for m in MODES}
    persist = {m: {c: [0, 0, 0] for c in a.conf} for m in ("tam", "tam+bant")}   # [fp, önceki 1 karede de var, önceki 2 karede de var]
    toggles = {m: {"ham": 0, "his": 0} for m in MODES}
    dimfrac = {m: [] for m in MODES}
    glare = {m: {"n": np.zeros(len(SUMMARY_BINS), int), "v": np.zeros(len(SUMMARY_BINS), int)} for m in MODES}
    util = {m: [0, 0] for m in MODES}
    safety_off, safety_n, safety_why = 0, 0, {}
    sweep_modes = ["tam", "takip·tam", "takip·bant/2·yok"]
    sweep = {}
    if a.sweep_conf or a.sweep_release:
        for m in sweep_modes:
            for c in (a.sweep_conf or [a.mask_conf]):
                for r in (a.sweep_release or [a.release]):
                    sweep[(m, c, r)] = {"flk": 0, "v": 0, "n": 0, "vf": 0, "nf": 0, "lit": 0, "free": 0}
    pairs = 0
    ref = BeamController(sysc, a.mask_conf, a.release)
    print(f"Far kararı: güven ≥ {a.mask_conf}, histerezis {ref.release} sn, güvenlik payı {ref.margin}°")
    t_full, t_band, t_trk = [], [], []
    t0 = time.time()
    for vi, (name, v) in enumerate(vids.items(), 1):
        fdir = resolve(v["frames_dir"])
        frames = sorted(fdir.glob(f"{name}-*.jpg"))
        key = v["key_index"]
        seq = frames[max(0, key - a.warmup):key + 1]
        dets, means = [], []
        for f in seq:
            img = cv2.imread(str(f))
            means.append(float(img.mean()) / 255.0)
            s0 = time.perf_counter()
            full = predict_full(model, img, min_conf, a.imgsz)
            s1 = time.perf_counter()
            band = predict_band(band_model, img, cfg, min_conf)
            s2 = time.perf_counter()
            dets.append((full, merge(full, band)))
            if vi > 1:
                t_full.append(s1 - s0)
                t_band.append(s2 - s1)

        lbl = a.data / "labels" / v.get("split", "test") / f"{name}.txt"
        gt, gt_all = read_gt(lbl), read_gt(lbl, range(6))
        n = len(dets)
        outs = {m: [] for m in MODES}
        trackers = {m: Tracker() for m in MODES if m.startswith("takip")}
        for i, (full, merged) in enumerate(dets):
            back = n - 1 - i                                  # key'e uzaklık (kare)
            outs["tam"].append(full)
            outs["tam+bant"].append(merged)
            inp = {"takip·tam": full, "takip·tam+bant": merged,
                   "takip·bant/2·var": merged if back % 2 == 0 else full,
                   "takip·bant/2·yok": merged if back % 2 == 1 else full}
            for m, d in inp.items():
                s = time.perf_counter()
                o = trackers[m].update(*d)
                t_trk.append(time.perf_counter() - s)
                outs[m].append(o[:3])

        for m in ("tam", "tam+bant"):
            xyxy, cls, conf = outs[m][-1]
            for c in a.conf:
                k = conf >= c
                fpm = fp_mask(xyxy[k], cls[k], gt_all)
                for b, cc in zip(xyxy[k][fpm], cls[k][fpm]):
                    persist[m][c][0] += 1
                    persist[m][c][1] += persistent(b, cc, outs[m][-2:-1])
                    persist[m][c][2] += persistent(b, cc, outs[m][-3:-1])
        for m in MODES:
            accumulate(stats[m], gt, *outs[m][-1], a.conf, a.iou, gt_all=gt_all)
            ham, his = BeamController(sysc, a.mask_conf, 0.0), BeamController(sysc, a.mask_conf, a.release)
            m_ham = [ham.step(*o, i / a.fps) for i, o in enumerate(outs[m])]
            sm = SafetyMonitor(sysc)                                      # REQ-S-01/02: bozuk girdide kısa far
            m_his = []
            for i, o in enumerate(outs[m]):
                ok = sm.frame(i / a.fps, mean01=means[i])
                m_his.append(SafetyMonitor.apply(his.step(*o, i / a.fps), ok))
                if m == "tam" and i >= 3:
                    safety_n += 1
                    safety_off += not ok
                    for r in sm.reasons:
                        safety_why[r] = safety_why.get(r, 0) + 1
            for key_, ms in (("ham", m_ham[3:]), ("his", m_his[3:])):          # ilk 3 kare ısınma
                toggles[m][key_] += short_runs(ms, max(1, round(a.flicker_s * a.fps)))
            dimfrac[m] += [x.mean() for x in m_his[3:]]
            viol, inb, lit, free = glare_and_utilization(m_his[-1], gt, gt_all, ref.cam, ref.edges, ref.lamps)
            w = gt[:, 2] - gt[:, 0] if len(gt) else np.zeros(0)
            for bi, (lo, hi) in enumerate(SUMMARY_BINS.values()):
                lo_px, hi_px = BINS[lo][0], BINS[hi - 1][1]
                sel = (w >= lo_px) & (w < hi_px) & inb                  # yalnızca far kapsamındaki araçlar
                glare[m]["n"][bi] += int(sel.sum())
                glare[m]["v"][bi] += int(viol[sel].sum()) if len(viol) else 0
            util[m][0] += lit
            util[m][1] += free
        far = None
        for (m, c, r), acc in sweep.items():
            bc = BeamController(sysc, c, r)
            ms = [bc.step(*o, i / a.fps) for i, o in enumerate(outs[m])]
            acc["flk"] += short_runs(ms[3:], max(1, round(a.flicker_s * a.fps)))
            viol, inb, lit, free = glare_and_utilization(ms[-1], gt, gt_all, bc.cam, bc.edges, bc.lamps)
            w = gt[:, 2] - gt[:, 0] if len(gt) else np.zeros(0)
            viol, w = viol[inb], w[inb]                                      # yalnızca far kapsamındaki araçlar
            far = w < BINS[3][0]                                             # < 16 px: ~125 m+
            acc["v"] += int(viol.sum()) if len(viol) else 0
            acc["n"] += len(viol)
            acc["vf"] += int(viol[far].sum()) if len(viol) else 0
            acc["nf"] += int(far.sum())
            acc["lit"] += lit
            acc["free"] += free
        pairs += max(len(outs["tam"][3:]) - 1, 0)
        print(f"\r  {vi}/{len(vids)}  ({time.time() - t0:.0f} sn)", end="", flush=True)
    print()

    tf, tb, tt = (1000 * float(np.mean(x)) if x else float("nan") for x in (t_full, t_band, t_trk))
    cost = {"tam": tf, "tam+bant": tf + tb, "takip·tam": tf + tt, "takip·tam+bant": tf + tb + tt,
            "takip·bant/2·var": tf + tb / 2 + tt, "takip·bant/2·yok": tf + tb / 2 + tt}
    minutes = pairs / a.fps / 60

    def rec(st, c, lo, hi, k="any"):
        g = st[c]["gt"][lo:hi].sum()
        return (st[c][k][lo:hi].sum() / g if g else float("nan")), int(g)

    out = {"videos": len(vids), "conds": a.conds, "warmup": a.warmup, "fps": a.fps, "ms": cost,
           "toggles_per_min": {}, "dim_frac": {}, "results": {}}
    for c in a.conf:
        print(f"\n### Key kare, güven ≥ {c}: recall kutu+ışık (kutu) ve yanlış alarm\n")
        hdr = list(SUMMARY_BINS)
        print("| Mod | " + " | ".join(hdr) + " | Yanlış alarm / kare |")
        print("|---|" + "---|" * (len(hdr) + 1))
        for m in MODES:
            st = stats[m]
            cells = []
            for lab, (lo, hi) in SUMMARY_BINS.items():
                r, g = rec(st, c, lo, hi)
                rb, _ = rec(st, c, lo, hi, "box")
                cells.append(f"{r:.2f} ({rb:.2f})")
            fp = st[c]["fp"][0] / max(st[c]["img"][0], 1)
            print(f"| {m} | " + " | ".join(cells) + f" | {fp:.2f} |")
            out["results"].setdefault(str(c), {})[m] = {k: st[c][k].tolist() for k in ("gt", "box", "any", "fp", "img")}
        ns = [rec(stats["tam"], c, lo, hi)[1] for lo, hi in SUMMARY_BINS.values()]
        print("| n (araç) | " + " | ".join(str(x) for x in ns) + " | |")

    print(f"\n### ADB sistem metrikleri (far kararı: güven ≥ {a.mask_conf}, histerezis {ref.release} sn)\n")
    print("Kamaşma ihlali: etiketli karede gerçek araç, açık bir segmentte kaldı mı (düşük iyi).")
    print("Uzun far kullanımı: hiç gerçek nesne olmayan segmentlerin açık olanı (yüksek iyi).")
    print(f"Titreme: {a.flicker_s} sn'den kısa süren açılma/kısılma, dakikada (hedef ≤ 2).\n")
    hdr = list(SUMMARY_BINS)
    print("| Mod | Kamaşma ihlali: " + " / ".join(hdr) + " | Uzun far kullanımı | Kısık oran | "
          f"Titreme/dk ham | Titreme/dk histerezis | Süre (ms) |")
    print("|---|---|---|---|---|---|---|")
    out["glare"], out["utilization"] = {}, {}
    for m in MODES:
        g = glare[m]
        gv = " / ".join(f"{v / n:.2f}" if n else "–" for v, n in zip(g["v"], g["n"]))
        ut = util[m][0] / max(util[m][1], 1)
        tr = toggles[m]["ham"] / minutes if minutes else float("nan")
        th = toggles[m]["his"] / minutes if minutes else float("nan")
        df = float(np.mean(dimfrac[m])) if dimfrac[m] else float("nan")
        out["toggles_per_min"][m] = {"ham": tr, "histerezis": th}
        out["dim_frac"][m], out["utilization"][m] = df, ut
        out["glare"][m] = {"n": g["n"].tolist(), "ihlal": g["v"].tolist()}
        print(f"| {m} | {gv} | {ut:.2f} | {df:.2f} | {tr:.0f} | {th:.0f} | {cost[m]:.1f} |")
    print("| n (araç) | " + " / ".join(str(x) for x in glare["tam"]["n"]) + " | | | | | |")
    print(f"\nGüvenlik izleyici: karelerin %{100 * safety_off / max(safety_n, 1):.1f}'inde uzun far kapattı "
          f"({safety_off}/{safety_n}). Sebepler: {safety_why or 'yok'}. Bulanıklık kontrolü: "
          f"{'kapalı (ADR-018)' if sysc['safety_monitor'].get('blur_laplacian_min') is None else 'açık'}.")
    out["safety"] = {"off_frames": safety_off, "frames": safety_n, "reasons": safety_why}
    print("\n### Yanlış alarmlar kalıcı mı? (tek kare modlarında, key karedeki yanlış alarmların\n"
          "### kaçı önceki karelerde de aynı yerde tespit edilmiş)\n")
    print("| Mod | Eşik | Yanlış alarm (toplam) | Önceki 1 karede de var | Önceki 2 karede de var |")
    print("|---|---|---|---|---|")
    out["persistence"] = {}
    for m, d in persist.items():
        for c, (n, p1, p2) in d.items():
            print(f"| {m} | {c} | {n} | %{100 * p1 / max(n, 1):.0f} | %{100 * p2 / max(n, 1):.0f} |")
            out["persistence"][f"{m}@{c}"] = [n, p1, p2]
    print("Kalıcı yanlış alarmlar (sokak lambası, yansıma, etiketsiz uzak araç) takiple elenemez.")
    if sweep:
        print(f"\n### Far kararı taraması (güven eşiği × histerezis), {len(vids)} video\n")
        print("| Mod | Güven | Histerezis (sn) | Kamaşma ihlali: tümü | 125 m+ | Uzun far kullanımı | Titreme / dk |")
        print("|---|---|---|---|---|---|---|")
        out["sweep"] = []
        for (m, c, r), acc in sweep.items():
            rr = BeamController(sysc, c, r).release
            v = acc["v"] / max(acc["n"], 1)
            vf = acc["vf"] / acc["nf"] if acc["nf"] else float("nan")
            ut = acc["lit"] / max(acc["free"], 1)
            fl = acc["flk"] / minutes if minutes else float("nan")
            print(f"| {m} | {c} | {rr} | {v:.3f} ({acc['v']}/{acc['n']}) | {vf:.2f} ({acc['vf']}/{acc['nf']}) "
                  f"| {ut:.2f} | {fl:.1f} |")
            out["sweep"].append({"mod": m, "conf": c, "release": rr, **acc, "flicker_per_min": fl})
    print(f"\nSüre: tam {tf:.1f} ms, bant {tb:.1f} ms, takip {tt:.2f} ms (RTX dizüstü, okuma hariç).")
    print("Not: 5 fps; gerçek kamera 30 fps. Kamaşma ihlali histerezisli maske ile, etiketli karede ölçülür.")

    run_dir = run_dir_for(a.weights)
    (run_dir / "eval_video.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Sonuçlar: {run_dir / 'eval_video.json'}")


if __name__ == "__main__":
    main()