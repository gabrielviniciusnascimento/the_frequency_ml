#!/usr/bin/env python3
"""
audit_08_sensitivity_injection.py  (HANDOFF Task 8 — informative sensitivity control)

The original "discrete level control" is a strawman (it shows level structure makes no
contrast tail, which we already knew). A more informative control measures the detector's
*power*: into a clean, asymmetry-free base (the Gaussian-copula null of the real auditory
data — realistic marginals and R-L correlation, no injected excess), inject k=1% of cases
with a KNOWN one-sided inter-ear gap of G in {30,40,50,60} dB, and measure recall by G with
two detectors:
  A) the standardized contrast threshold (|z|>4 and |z|>3);
  B) the clustering pipeline (row-center -> RobustScaler -> PCA95 -> HDBSCAN mcs=10,ms=5):
     recalled if the injected case lands in a non-dominant, non-noise cluster.

Criterion (handoff): recall >= 0.90 for injected gaps >= 50 dB.

Output: outputs/json/audit_08_sensitivity_injection.json
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
FEATURE = ROOT / "data/processed/frequencia_feature_matrix_v1.csv"
OUT = ROOT / "outputs/json/audit_08_sensitivity_injection.json"
FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
R_COLS = [f"thr_R_{f}" for f in FREQS]; L_COLS = [f"thr_L_{f}" for f in FREQS]
COLS14 = R_COLS + L_COLS
RS = 42
GAPS = [30, 40, 50, 60]
K_FRAC = 0.01
CAP_DB = 120.0


def nearest_psd_corr(C):
    w, V = np.linalg.eigh(C); w = np.clip(w, 1e-6, None)
    C2 = V @ np.diag(w) @ V.T; d = np.sqrt(np.diag(C2)); return C2 / np.outer(d, d)


def copula_base(real, seed):
    """Gaussian-copula synthetic population: realistic marginals + rank corr, no excess."""
    n, p = real.shape; ranks = np.zeros_like(real); sc = []
    for j in range(p):
        col = real[:, j]; sc.append(np.sort(col[~np.isnan(col)]))
        r = stats.rankdata(col, nan_policy="omit"); ranks[:, j] = np.where(np.isnan(col), np.nan, r)
    Rc = pd.DataFrame(ranks).corr().to_numpy(); Rc = np.nan_to_num(Rc, nan=0.0); np.fill_diagonal(Rc, 1.0)
    Rc = nearest_psd_corr(Rc)
    Z = np.random.RandomState(seed).multivariate_normal(np.zeros(p), Rc, size=n)
    U = stats.norm.cdf(Z); syn = np.empty((n, p))
    for j in range(p):
        syn[:, j] = sc[j][np.clip((U[:, j] * len(sc[j])).astype(int), 0, len(sc[j]) - 1)]
    return syn


def cluster_labels(M14):
    df = pd.DataFrame(M14)
    X = df.sub(df.mean(axis=1), axis=0).fillna(0.0).to_numpy(np.float64)
    X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
    Xp = PCA(n_components=0.95, svd_solver="full", random_state=RS).fit_transform(X)
    return hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean",
                           cluster_selection_method="eom", core_dist_n_jobs=-1).fit_predict(Xp)


def main():
    df = pd.read_csv(FEATURE, low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce"); df = df[(age >= 20) & (age <= 69)]
    thr = df[COLS14].apply(pd.to_numeric, errors="coerce"); thr = thr[thr.notna().sum(axis=1) >= 10]
    thr = thr[(thr > 25).any(axis=1)]
    real = thr.to_numpy(np.float64)
    base = copula_base(real, RS)                  # clean, asymmetry-free population
    n = base.shape[0]
    k = max(1, int(round(K_FRAC * n)))
    rng = np.random.RandomState(RS + 3)

    results = {}
    for G in GAPS:
        M = base.copy()
        idx = rng.choice(n, size=k, replace=False)
        # inject: make the right ear worse by G dB (cap at CAP), left ear untouched
        M[np.ix_(idx, range(7))] = np.clip(M[np.ix_(idx, range(7))] + G, None, CAP_DB)
        inj = np.zeros(n, bool); inj[idx] = True

        pta_r = M[:, :7].mean(1); pta_l = M[:, 7:].mean(1)
        c = pta_r - pta_l; sd = float(np.nanstd(c)); z = np.abs(c) / sd
        recall_z4 = float(np.mean(z[inj] > 4))
        recall_z3 = float(np.mean(z[inj] > 3))

        lab = cluster_labels(M)
        sizes = {int(v): int((lab == v).sum()) for v in np.unique(lab[lab != -1])}
        dom = max(sizes, key=sizes.get) if sizes else None
        recalled_clust = np.array([(l != -1 and l != dom) for l in lab])
        recall_cluster = float(np.mean(recalled_clust[inj]))

        results[str(G)] = {
            "injected_k": k, "mean_injected_z": round(float(np.mean(z[inj])), 2),
            "recall_z_gt4": round(recall_z4, 3), "recall_z_gt3": round(recall_z3, 3),
            "recall_clustering": round(recall_cluster, 3),
            "meets_0.90": bool(recall_z4 >= 0.90),
        }

    OUT.write_text(json.dumps({
        "script": "audit_08_sensitivity_injection.py", "task": "HANDOFF Task 8 — sensitivity injection",
        "base": "Gaussian-copula null of real auditory data (no injected excess)",
        "n": n, "k_per_gap": k, "k_frac": K_FRAC, "sd_native_db": round(sd, 2),
        "detectors": {"A": "|z|>4 and |z|>3 on standardized contrast", "B": "non-dominant HDBSCAN cluster"},
        "results_by_gap_db": results, "status": "EXECUTED",
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"base n={n}, k={k}/gap, sd(contrast)={sd:.2f} dB")
    print(f"{'gap dB':>6} | {'mean z':>6} | {'recall|z|>4':>11} | {'recall|z|>3':>11} | {'recall clust':>12} | >=0.90")
    for G in GAPS:
        r = results[str(G)]
        print(f"{G:>6} | {r['mean_injected_z']:>6} | {r['recall_z_gt4']:>11} | {r['recall_z_gt3']:>11} | "
              f"{r['recall_clustering']:>12} | {'Y' if r['meets_0.90'] else 'n'}")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
