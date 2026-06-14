#!/usr/bin/env python3
"""
Nome: 27b_pooling_figure.py
Tarefa: Figura da ablação de pooling binaural (companheira de 27_*).
        3 painéis:
          (A) PCA do espaço de forma 14D (orelhas separadas) — o cluster de
              assimetria (vermelho) sobressai do contínuo dominante (cinza).
          (B) PCA do espaço de forma 7D (média binaural) — os MESMOS
              indivíduos (vermelho) dissolvidos na massa.
          (C) audiograma médio do grupo: R vs L (separadas) e a média binaural
              que apaga a assimetria.

Output: outputs/dashboards/pooling_ablation_figure.png
Reproduz o preprocessing de 27_binaural_pooling_ablation.py (mesmos filtros/seed).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import hdbscan

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RANDOM_STATE = 42
AGE_MIN, AGE_MAX = 20, 69
MIN_COMPLETENESS = 10
ANY25 = 25.0
PCA_VAR = 0.95
MCS, MS = 10, 5
FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
R_COLS = [f"thr_R_{f}" for f in FREQS]
L_COLS = [f"thr_L_{f}" for f in FREQS]
COLS14 = R_COLS + L_COLS

FEATURE = Path("data/processed/frequencia_feature_matrix_v1.csv")
OUT = Path("outputs/dashboards/pooling_ablation_figure.png")


def load():
    df = pd.read_csv(FEATURE, low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df = df[(age >= AGE_MIN) & (age <= AGE_MAX)].copy()
    thr = df[COLS14].apply(pd.to_numeric, errors="coerce")
    keep = thr.notna().sum(axis=1) >= MIN_COMPLETENESS
    df, thr = df[keep].copy(), thr[keep].copy()
    m = (thr > ANY25).any(axis=1)
    return df[m].reset_index(drop=True), thr[m].reset_index(drop=True)


def embed(thr_df):
    X = thr_df.sub(thr_df.mean(axis=1, skipna=True), axis=0).fillna(0.0).to_numpy(np.float64)
    X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
    pca = PCA(n_components=PCA_VAR, svd_solver="full", random_state=RANDOM_STATE)
    Xp = pca.fit_transform(X)
    return Xp, pca.explained_variance_ratio_


def main():
    df, thr14 = load()

    # Espaço A (14D) + identificar cluster de assimetria (menor cluster)
    Xa, vra = embed(thr14)
    ca = hdbscan.HDBSCAN(min_cluster_size=MCS, min_samples=MS, metric="euclidean",
                         cluster_selection_method="eom", core_dist_n_jobs=-1)
    la = ca.fit_predict(Xa)
    sizes = {k: int(v) for k, v in zip(*np.unique(la[la != -1], return_counts=True))}
    dom = max(sizes, key=sizes.get)
    asym_label = min((k for k in sizes if k != dom), key=lambda k: sizes[k])
    asym = la == asym_label

    # Espaço B (7D pooled)
    pooled = pd.DataFrame({f"bin_{f}": thr14[[f"thr_R_{f}", f"thr_L_{f}"]].mean(axis=1, skipna=True)
                           for f in FREQS})
    Xb, vrb = embed(pooled)

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    # (A) 14D
    ax = axes[0]
    ax.scatter(Xa[~asym, 0], Xa[~asym, 1], s=6, c="#bbbbbb", alpha=0.5, linewidths=0, label="bulk (continuum)")
    ax.scatter(Xa[asym, 0], Xa[asym, 1], s=70, c="#d62728", edgecolors="black", linewidths=0.6,
               label=f"unilateral asymmetry (N={asym.sum()})", zorder=5)
    ax.set_title(f"A. Ears kept separate (14D shape)\nPC1 {vra[0]*100:.0f}% · PC2 {vra[1]*100:.0f}%",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # (B) 7D pooled
    ax = axes[1]
    ax.scatter(Xb[~asym, 0], Xb[~asym, 1], s=6, c="#bbbbbb", alpha=0.5, linewidths=0)
    ax.scatter(Xb[asym, 0], Xb[asym, 1], s=70, c="#d62728", edgecolors="black", linewidths=0.6,
               label="same individuals", zorder=5)
    ax.set_title(f"B. Binaural mean (7D shape)\nPC1 {vrb[0]*100:.0f}% · PC2 {vrb[1]*100:.0f}%",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    # (C) audiograma médio do grupo
    ax = axes[2]
    r_mean = thr14.loc[asym, R_COLS].mean().to_numpy()
    l_mean = thr14.loc[asym, L_COLS].mean().to_numpy()
    pooled_mean = (r_mean + l_mean) / 2
    x = np.arange(len(FREQS))
    ax.plot(x, r_mean, "o-", color="#1f77b4", label="Right ear (raw)")
    ax.plot(x, l_mean, "s-", color="#2ca02c", label="Left ear (raw)")
    ax.plot(x, pooled_mean, "^--", color="#d62728", label="Binaural mean (what B sees)")
    ax.set_xticks(x); ax.set_xticklabels([f"{f//1000}k" if f >= 1000 else str(f) for f in FREQS], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("C. Mean audiogram of the asymmetry group\n(pooling collapses a 59 dB R-L gap)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Threshold (dB HL)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)

    fig.suptitle("Binaural averaging erases the only discrete signal in NHANES audiogram shape "
                 "(N=7,695; survival of asymmetry cluster under pooling = 0/13)",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Figura salva: {OUT}")


if __name__ == "__main__":
    main()
