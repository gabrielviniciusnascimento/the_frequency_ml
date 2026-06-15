#!/usr/bin/env python3
"""
audit_06_cluster_stability.py  (HANDOFF Task 6 — N=13 cluster stability sweep)

The "N=13 asymmetry cluster" (script 27/28) is a single-hyperparameter snapshot. We sweep
the HDBSCAN difference/shape space over random_state x min_cluster_size x min_samples and
ask: does a distinct extreme-asymmetry cluster appear at all, how does its N vary, and what
is its R-worse fraction across cells? If the cluster is unstable (or one-sidedness flips),
the binaural-ablation claim must rest on the |z| threshold, not on "the cluster" — which is
exactly how s3.4 was reframed after audit_03.

Pipeline per cell mirrors 27_binaural_pooling_ablation.cluster_space: row-center ->
RobustScaler(25-75) -> PCA 95% -> HDBSCAN(mcs, ms). Asymmetry cluster = the non-noise
cluster with the largest mean |PTA_R - PTA_L|; "distinct" if that mean gap > 30 dB.

Output: outputs/json/audit_06_cluster_stability.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import hdbscan

ROOT = Path(__file__).resolve().parents[1]
FEATURE = ROOT / "data/processed/frequencia_feature_matrix_v1.csv"
OUT = ROOT / "outputs/json/audit_06_cluster_stability.json"
FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
R_COLS = [f"thr_R_{f}" for f in FREQS]; L_COLS = [f"thr_L_{f}" for f in FREQS]
COLS14 = R_COLS + L_COLS
RS_GRID = [0, 1, 2, 7, 42]
MCS_GRID = [5, 10, 15, 20]
MS_GRID = [1, 5, 10]
DISTINCT_GAP_DB = 30.0


def load():
    df = pd.read_csv(FEATURE, low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df = df[(age >= 20) & (age <= 69)].copy()
    thr = df[COLS14].apply(pd.to_numeric, errors="coerce")
    keep = thr.notna().sum(axis=1) >= 10
    thr = thr[keep]; thr = thr[(thr > 25).any(axis=1)]
    return thr.reset_index(drop=True)


def embed(thr, rs):
    X = thr.sub(thr.mean(axis=1, skipna=True), axis=0).fillna(0.0).to_numpy(np.float64)
    X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
    return PCA(n_components=0.95, svd_solver="full", random_state=rs).fit_transform(X)


def main():
    thr = load()
    pta_r = thr[R_COLS].mean(axis=1).to_numpy()
    pta_l = thr[L_COLS].mean(axis=1).to_numpy()
    gap = pta_r - pta_l                          # >0 right-worse
    N = len(thr)

    cells = []
    for rs in RS_GRID:
        Xp = embed(thr, rs)
        for mcs in MCS_GRID:
            for ms in MS_GRID:
                lab = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean",
                                      cluster_selection_method="eom", core_dist_n_jobs=-1).fit_predict(Xp)
                # asymmetry cluster = non-noise cluster with max mean |gap|
                best = None
                for c in np.unique(lab[lab != -1]):
                    m = lab == c
                    mg = float(np.mean(np.abs(gap[m])))
                    if best is None or mg > best[1]:
                        best = (int(c), mg, m)
                if best is None:
                    cells.append({"rs": rs, "mcs": mcs, "ms": ms, "asym_cluster": None})
                    continue
                _, meang, m = best
                n = int(m.sum())
                n_rw = int(np.sum(gap[m] > 0))
                cells.append({
                    "rs": rs, "mcs": mcs, "ms": ms,
                    "asym_n": n, "asym_mean_abs_gap_db": round(meang, 1),
                    "asym_mean_signed_gap_db": round(float(np.mean(gap[m])), 1),
                    "frac_right_worse": round(n_rw / n, 3),
                    "distinct": bool(meang > DISTINCT_GAP_DB),
                })

    distinct = [c for c in cells if c.get("distinct")]
    rs_invariant = len({(c["mcs"], c["ms"]): None for c in cells})  # sanity
    # determinism check across rs for one (mcs,ms)
    same_mcs_ms = [c for c in cells if c["mcs"] == 10 and c["ms"] == 5 and "asym_n" in c]
    rs_ns = sorted({c["asym_n"] for c in same_mcs_ms})

    summary = {
        "n_cells": len(cells), "n_distinct_cells": len(distinct),
        "frac_cells_distinct": round(len(distinct) / len(cells), 3),
        "distinct_asym_N_range": [min((c["asym_n"] for c in distinct), default=None),
                                  max((c["asym_n"] for c in distinct), default=None)],
        "distinct_asym_N_values": sorted({c["asym_n"] for c in distinct}),
        "distinct_frac_right_worse_range": [min((c["frac_right_worse"] for c in distinct), default=None),
                                            max((c["frac_right_worse"] for c in distinct), default=None)],
        "random_state_inert": (len(rs_ns) == 1),
        "note_rs": "PCA(svd_solver=full) and HDBSCAN are deterministic; random_state does not "
                   "change the partition (confirmed: identical N across rs at mcs=10,ms=5).",
    }

    OUT.write_text(json.dumps({
        "script": "audit_06_cluster_stability.py", "task": "HANDOFF Task 6 — cluster stability sweep",
        "N": N, "grid": {"random_state": RS_GRID, "min_cluster_size": MCS_GRID, "min_samples": MS_GRID},
        "distinct_gap_threshold_db": DISTINCT_GAP_DB,
        "summary": summary, "cells": cells, "status": "EXECUTED",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"N={N}; {len(cells)} cells; distinct asym cluster in {len(distinct)} "
          f"({summary['frac_cells_distinct']*100:.0f}%)")
    print(f"distinct asym N range {summary['distinct_asym_N_range']} values {summary['distinct_asym_N_values']}")
    print(f"distinct frac_right_worse range {summary['distinct_frac_right_worse_range']}")
    print(f"random_state inert: {summary['random_state_inert']}")
    print(f"\n{'rs':>3} {'mcs':>3} {'ms':>3} | {'asymN':>5} {'|gap|':>6} {'signed':>7} {'fracR':>6} {'distinct':>8}")
    for c in cells:
        if "asym_n" in c:
            print(f"{c['rs']:>3} {c['mcs']:>3} {c['ms']:>3} | {c['asym_n']:>5} {c['asym_mean_abs_gap_db']:>6} "
                  f"{c['asym_mean_signed_gap_db']:>7} {c['frac_right_worse']:>6} {str(c['distinct']):>8}")
        else:
            print(f"{c['rs']:>3} {c['mcs']:>3} {c['ms']:>3} | (no non-noise cluster)")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
