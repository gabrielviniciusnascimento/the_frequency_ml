#!/usr/bin/env python3
"""
Nome: vis_02_pipeline.py
Tarefa: MESMO pipeline da audiometria/grip (26 / grip_02) no espaço bilateral de
        visão (vis_R, vis_L = equivalente esférico OD/OS). Mesmos hiperparâmetros/seeds.
        NÃO adaptar — divergência é achado.

Espaço de feature bilateral: [vis_R, vis_L] (2 features; mais fino que audio 14 / grip 6 —
diferença estrutural reportada, não corrigida). Row-center -> RobustScaler -> PCA95.
Filtros: idade 20-69 (igual aos outros) + ambos os olhos válidos. Sem ANY-equivalente
(o codebook de refração não define corte de anormalidade; cutoffs clínicos são externos).
Output: outputs/json/vis_02_pipeline.json
"""

import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
import hdbscan

RANDOM_STATE = 42
K_RANGE = list(range(2, 11))
N_SEEDS = 12
GAP_B = 5
SIL_SAMPLE = 2500
PCA_VAR = 0.95
HDBSCAN_MCS = 10
HDBSCAN_MS = 5
AGE_MIN, AGE_MAX = 20, 69
COLS = ["vis_R", "vis_L"]

FEATURE = Path("data/processed/vis_feature_matrix.csv")
OUTPUT = Path("outputs/json/vis_02_pipeline.json")
LOG = Path("outputs/logs/vis_02_pipeline.log")
LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG, encoding="utf-8"), logging.StreamHandler()])
log = logging.getLogger(__name__)
rng = np.random.RandomState(RANDOM_STATE)


def load_pca_space():
    df = pd.read_csv(FEATURE, low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df = df[(age >= AGE_MIN) & (age <= AGE_MAX)].copy()
    g = df[COLS].apply(pd.to_numeric, errors="coerce")
    g = g[g.notna().all(axis=1)]
    log.info(f"VISÃO 20-69 ambos olhos: {len(g)} × {g.shape[1]}")
    X = g.sub(g.mean(axis=1), axis=0).to_numpy(np.float64)
    X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
    pca = PCA(n_components=PCA_VAR, random_state=RANDOM_STATE)
    Xp = pca.fit_transform(X)
    log.info(f"PCA: {Xp.shape[1]} comp")
    return Xp.astype(np.float64), int(len(g))


def gap_statistic(X, k, b=GAP_B):
    def inertia(d, seed, ni): return KMeans(k, n_init=ni, random_state=seed).fit(d).inertia_
    logW = np.log(inertia(X, RANDOM_STATE, 3))
    mins, maxs = X.min(0), X.max(0)
    refs = np.array([np.log(inertia(rng.uniform(mins, maxs, size=X.shape), RANDOM_STATE + i, 1)) for i in range(b)])
    return float(refs.mean() - logW), float(refs.std() * np.sqrt(1 + 1.0 / b))


def kmeans_block(X):
    per_k, gaps = {}, {}
    for k in K_RANGE:
        labels = [KMeans(k, n_init=5, random_state=s).fit_predict(X) for s in range(N_SEEDS)]
        aris = [adjusted_rand_score(labels[i], labels[j]) for i in range(N_SEEDS) for j in range(i + 1, N_SEEDS)]
        sil = silhouette_score(X, labels[0], sample_size=min(SIL_SAMPLE, len(X)), random_state=RANDOM_STATE)
        g, s = gap_statistic(X, k)
        per_k[k] = {"silhouette": round(float(sil), 4), "seed_ari_mean": round(float(np.mean(aris)), 4),
                    "gap": round(g, 4), "gap_s": round(s, 4)}
        gaps[k] = (g, s)
        log.info(f"  k={k}: sil={sil:.3f} seedARI={np.mean(aris):.3f} gap={g:.3f}")
    best = max(per_k, key=lambda k: per_k[k]["silhouette"])
    gap_opt = None
    ks = sorted(gaps)
    for i in range(len(ks) - 1):
        if gaps[ks[i]][0] >= gaps[ks[i + 1]][0] - gaps[ks[i + 1]][1]:
            gap_opt = ks[i]; break
    return {"per_k": per_k, "best_k_by_silhouette": int(best),
            "best_silhouette_value": per_k[best]["silhouette"], "gap_optimal_k": gap_opt}


def gmm_block(X):
    per_k = {k: {"bic": round(float(GaussianMixture(k, covariance_type="full", random_state=RANDOM_STATE,
                 n_init=2, max_iter=200).fit(X).bic(X)), 1)} for k in K_RANGE}
    bics = [per_k[k]["bic"] for k in K_RANGE]
    bmin = K_RANGE[int(np.argmin(bics))]
    return {"per_k": per_k, "bic_min_k": int(bmin),
            "interior_minimum": bool(bmin not in (K_RANGE[0], K_RANGE[-1])),
            "bic_range_pct_of_mean": round(float((max(bics) - min(bics)) / np.mean(bics) * 100), 3)}


def hdbscan_block(X):
    lab = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MCS, min_samples=HDBSCAN_MS,
                          metric="euclidean", cluster_selection_method="eom", core_dist_n_jobs=-1).fit_predict(X)
    nc = int(len(set(lab) - {-1})); nn = int((lab == -1).sum())
    sizes = {int(k): int(v) for k, v in zip(*np.unique(lab[lab != -1], return_counts=True))} if nc else {}
    largest = max(sizes.values()) / len(lab) if sizes else 0.0
    return {"n_clusters": nc, "n_noise": nn, "noise_fraction": round(nn / len(lab), 4),
            "largest_cluster_fraction": round(float(largest), 4)}


def main():
    X, n = load_pca_space()
    km, gm, hd = kmeans_block(X), gmm_block(X), hdbscan_block(X)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "script": "vis_02_pipeline.py",
        "note": "MESMO pipeline (26/grip_02). Feature space = [vis_R, vis_L] SE (2D).",
        "n_samples": n, "pca_components": int(X.shape[1]),
        "kmeans": km, "gmm": gm, "hdbscan": hd, "status": "EXECUTED",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nVISÃO pipeline: N={n}, PCA={X.shape[1]} comp")
    print(f"  K-means: best silhouette={km['best_silhouette_value']} @k={km['best_k_by_silhouette']}; gap-opt k={km['gap_optimal_k']}")
    print(f"  GMM: BIC min k={gm['bic_min_k']} interior={gm['interior_minimum']} depth={gm['bic_range_pct_of_mean']}%")
    print(f"  HDBSCAN: {hd['n_clusters']} clusters, {hd['noise_fraction']*100:.1f}% ruído, dominante={hd['largest_cluster_fraction']*100:.1f}%")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
