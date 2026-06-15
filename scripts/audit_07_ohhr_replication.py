#!/usr/bin/env python3
"""
audit_07_ohhr_replication.py  (HANDOFF Task 7 — OHHR external replication)

Applies the inter-ear sum/difference decomposition + tail test to the OHHR audiograms
(Oldenburg Hearing Health Record), the only non-NHANES cohort with R/L thresholds in the
repo. Any replication outside NHANES weakens an "NHANES-specific artifact" reading.

Reconstruction (per script 25): audiogram_point -> audiogram_line (side, transducertype,
type) -> audiogram (clientid); keep HTL + AC; common frequencies [500,1000,2000,4000]
(OHHR tops out at 4 kHz). Keep clients with all four frequencies in BOTH ears.

Reports, for the OHHR contrast c = PTA_R - PTA_L:
  - |z| tail counts and R-worse fraction (binomial vs 50/50);
  - recovery of the extreme (|z|>2) cases in the DIFFERENCE subspace vs the SUM subspace
    (RobustScaler -> PCA95 -> HDBSCAN mcs=10,ms=5), mirroring script 28.
Criterion (handoff): difference-subspace recovery > 0.5, with side reported.

Output: outputs/json/audit_07_ohhr_replication.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import hdbscan

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/external/ohhr/data"
OUT = ROOT / "outputs/json/audit_07_ohhr_replication.json"
FREQS = [500, 1000, 2000, 4000]
RS = 42


def load_ohhr():
    audiogram = pd.DataFrame(json.loads((BASE / "audiogram.json").read_text(encoding="utf-8")))
    points = pd.DataFrame(json.loads((BASE / "audiogram_point.json").read_text(encoding="utf-8")))
    line = pd.DataFrame(json.loads((BASE / "audiogram_line.json").read_text(encoding="utf-8")))
    pts = points.merge(line[["audiogramlineid", "audiogramid", "side", "transducertype", "type"]],
                       on="audiogramlineid", how="left")
    pts = pts.merge(audiogram[["audiogramid", "clientid"]], on="audiogramid", how="left")
    pts = pts[(pts["type"] == "htl") & (pts["transducertype"] == "ac") & (pts["frequency"].isin(FREQS))]
    pts["level"] = pd.to_numeric(pts["level"], errors="coerce")
    pts = pts.dropna(subset=["level", "side", "clientid"])
    # per client x side x freq mean level
    tab = pts.groupby(["clientid", "side", "frequency"])["level"].mean().reset_index()
    wide = tab.pivot_table(index="clientid", columns=["side", "frequency"], values="level")
    rcols = [("right", f) for f in FREQS]; lcols = [("left", f) for f in FREQS]
    need = rcols + lcols
    wide = wide.dropna(subset=[c for c in need if c in wide.columns])
    R = wide[rcols].to_numpy(np.float64)
    L = wide[lcols].to_numpy(np.float64)
    return R, L, wide.index.to_numpy()


def recovery(feature, extreme_mask):
    """RobustScaler -> PCA95 -> HDBSCAN; recovery = largest extreme block in a
    non-dominant non-noise cluster / n_extreme."""
    X = RobustScaler(quantile_range=(25, 75)).fit_transform(feature)
    Xp = PCA(n_components=0.95, svd_solver="full", random_state=RS).fit_transform(X)
    lab = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean",
                          cluster_selection_method="eom", core_dist_n_jobs=-1).fit_predict(Xp)
    sizes = {int(v): int((lab == v).sum()) for v in np.unique(lab[lab != -1])}
    if not sizes:
        return 0.0, 0
    dom = max(sizes, key=sizes.get)
    ext_labels = lab[extreme_mask]
    nd = [l for l in ext_labels if l != -1 and l != dom]
    if not nd:
        return 0.0, 0
    vals, cnts = np.unique(nd, return_counts=True)
    return round(int(cnts.max()) / int(extreme_mask.sum()), 4), int(cnts.max())


def main():
    R, L, cid = load_ohhr()
    n = len(cid)
    pta_r = R.mean(1); pta_l = L.mean(1)
    c = pta_r - pta_l                       # >0 right-worse
    sd = float(np.nanstd(c))
    z = np.abs(c) / sd

    tail = {}
    for lv in (2, 3, 4):
        m = z > lv; nt = int(m.sum()); nr = int(np.sum(c[m] > 0))
        p = float(stats.binomtest(nr, nt, 0.5).pvalue) if nt else None
        tail[f"abs_z_gt_{lv}"] = {"n": nt, "n_right_worse": nr, "n_left_worse": nt - nr,
                                  "frac_right_worse": round(nr / nt, 3) if nt else None,
                                  "binom_p_vs_50_50": p}
    # native >40 dB tail
    m40 = np.abs(c) > 40; nt = int(m40.sum()); nr = int(np.sum(c[m40] > 0))
    tail["native_abs_gt_40dB"] = {"n": nt, "n_right_worse": nr, "n_left_worse": nt - nr,
                                  "frac_right_worse": round(nr / nt, 3) if nt else None,
                                  "binom_p_vs_50_50": float(stats.binomtest(nr, nt, 0.5).pvalue) if nt else None}

    ext = z > 2
    diff = R - L                # difference subspace (4-dim)
    summ = (R + L) / 2.0        # sum/level subspace (4-dim)
    rec_diff, blk_diff = recovery(diff, ext)
    rec_sum, blk_sum = recovery(summ, ext)

    out = {
        "script": "audit_07_ohhr_replication.py", "task": "HANDOFF Task 7 — OHHR external replication",
        "cohort": "OHHR (Oldenburg Hearing Health Record), HTL+AC, freqs [500,1000,2000,4000]",
        "n": n, "sd_native_db": round(sd, 3),
        "contrast_def": "c = PTA_R - PTA_L (mean of 4 freq); c>0 right-worse",
        "tail_by_side": tail,
        "n_extreme_z_gt2": int(ext.sum()),
        "difference_subspace_recovery": rec_diff, "difference_block": blk_diff,
        "sum_subspace_recovery": rec_sum, "sum_block": blk_sum,
        "criterion_diff_recovery_gt_0.5": bool(rec_diff > 0.5),
        "status": "EXECUTED",
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"OHHR N={n}  SD(R-L)={sd:.2f} dB  extreme(|z|>2)={int(ext.sum())}")
    print(f"difference-subspace recovery = {rec_diff} (block {blk_diff}/{int(ext.sum())})")
    print(f"sum-subspace recovery        = {rec_sum} (block {blk_sum})")
    print(f"{'tail':>18} | {'n':>3} | {'R':>3} {'L':>3} | {'fracR':>6} | {'binom p':>9}")
    for k, v in tail.items():
        pp = ('%.3g' % v['binom_p_vs_50_50']) if v['binom_p_vs_50_50'] is not None else 'NA'
        print(f"{k:>18} | {v['n']:>3} | {v['n_right_worse']:>3} {v['n_left_worse']:>3} | "
              f"{str(v['frac_right_worse']):>6} | {pp:>9}")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
