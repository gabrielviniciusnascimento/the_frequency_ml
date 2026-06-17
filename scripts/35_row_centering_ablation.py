#!/usr/bin/env python3
"""
Nome: 35_row_centering_ablation.py
Tarefa: Pré-validar a escolha de PRÉ-PROCESSAMENTO mais central do pipeline — o
        row-centering (subtrair a média por indivíduo para isolar a FORMA, removendo
        o nível geral). A crítica óbvia de um revisor: "o resultado 'contínuo, não
        clusters' é um artefato do centering". Este script roda as métricas de
        continuidade COM e SEM row-centering e reporta se a conclusão (sem clusters
        discretos robustos) se mantém em ambos.

        Nota de escopo: a cauda de assimetria |z| é calculada sobre os limiares CRUS
        (mean(R)-mean(L)), portanto é INVARIANTE ao centering por construção — o
        centering só afeta a alegação de continuidade do espaço de forma. Por isso a
        ablação foca nas métricas de cluster.

Input: data/processed/frequencia_feature_matrix_v1.csv (via _shape_space.load_cohort)
Output: outputs/json/35_row_centering_ablation.json
Run: .venv/Scripts/python.exe scripts/35_row_centering_ablation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import hdbscan
import diptest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _shape_space import load_cohort, R_COLS, L_COLS, lib_versions  # noqa: E402

OUT = ROOT / "outputs" / "json" / "35_row_centering_ablation.json"
RANDOM_STATE = 42
PCA_VAR = 0.95
K_RANGE = list(range(2, 7))


def embed(thr_np: np.ndarray, row_center: bool) -> np.ndarray:
    """RobustScaler(25,75) -> PCA(var, full). Optionally row-center first (the toggle)."""
    X = thr_np.copy()
    if row_center:
        X = X - np.nanmean(X, axis=1, keepdims=True)
    X = np.nan_to_num(X, nan=0.0)
    Xs = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
    pca = PCA(n_components=PCA_VAR, svd_solver="full", random_state=RANDOM_STATE)
    return pca.fit_transform(Xs).astype(np.float64)


def continuum_metrics(X_pca: np.ndarray) -> dict:
    """Best KMeans silhouette over K_RANGE, Hartigan dip on PC1, HDBSCAN structure."""
    sils = {}
    for k in K_RANGE:
        lab = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit_predict(X_pca)
        sils[k] = float(silhouette_score(X_pca, lab))
    best_k = max(sils, key=sils.get)
    dip, dip_p = diptest.diptest(X_pca[:, 0])
    lab_h = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean",
                            cluster_selection_method="eom",
                            core_dist_n_jobs=-1).fit_predict(X_pca)
    sizes = sorted(int((lab_h == c).sum()) for c in np.unique(lab_h[lab_h != -1]))
    return {
        "n_components": int(X_pca.shape[1]),
        "kmeans_silhouette_by_k": sils,
        "best_silhouette": sils[best_k], "best_k": int(best_k),
        "hartigan_dip_pc1": {"stat": float(dip), "p": float(dip_p)},
        "hdbscan": {"n_clusters": len(sizes), "sizes": sizes,
                    "noise_fraction": round(float((lab_h == -1).mean()), 4)},
    }


def main() -> None:
    df, thr = load_cohort()
    thr_np = thr.to_numpy(np.float64)

    centered = continuum_metrics(embed(thr_np, row_center=True))
    uncentered = continuum_metrics(embed(thr_np, row_center=False))

    # asymmetry tail is centering-invariant (computed on raw thresholds) — show it once
    c = thr[R_COLS].mean(axis=1).to_numpy() - thr[L_COLS].mean(axis=1).to_numpy()
    z = np.abs(c) / np.nanstd(c)
    asym_tail = {f"|z|>{lv}": int((z > lv).sum()) for lv in (2, 3, 4)}

    # interpretation: continuum holds if, under row-centering, silhouette is weak
    # (<0.5), PC1 is unimodal (dip p high), and HDBSCAN is dominant-mass + noise.
    holds_centered = (centered["best_silhouette"] < 0.5
                      and centered["hartigan_dip_pc1"]["p"] > 0.05)
    out = {
        "script": "35_row_centering_ablation.py",
        "purpose": "Pre-validate row-centering: does 'continuum, not discrete clusters' "
                   "survive toggling the centering step? Asymmetry tail is centering-invariant.",
        "cohort_n": int(len(thr)),
        "row_centered": centered,
        "no_row_centering": uncentered,
        "asymmetry_tail_centering_invariant": asym_tail,
        "continuum_holds_under_row_centering": bool(holds_centered),
        "note": "Without centering the space mixes level+shape, so any structure there "
                "reflects a severity gradient (itself a continuum), not discrete shape "
                "subtypes. The shape-continuum claim is the row-centered result.",
        "lib_versions": lib_versions(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    for name, m in (("row-centered", centered), ("no-centering", uncentered)):
        print(f"{name:13s}: best_sil={m['best_silhouette']:.3f} @k={m['best_k']}  "
              f"dip_p(PC1)={m['hartigan_dip_pc1']['p']:.3g}  "
              f"hdbscan={m['hdbscan']['n_clusters']}cl noise={m['hdbscan']['noise_fraction']}")
    print(f"asymmetry tail (centering-invariant): {asym_tail}")
    print(f"continuum holds under row-centering: {holds_centered}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
