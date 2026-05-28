#!/usr/bin/env python3
"""
Nome: 20_session5_outlier_analysis.py
Tarefa: Analisar os 585 outliers (cluster_id == -1) da Sessão 4.
Input: outputs/json/session4_assignments_any25.csv, data/processed/frequencia_feature_matrix_v1.csv
Output: outputs/json/20_outlier_analysis.json, outputs/json/20_top_outliers_profile.json
"""

import logging, json
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import hdbscan
    HDBSCAN_OK = True
except ImportError:
    HDBSCAN_OK = False

RANDOM_STATE = 42
FREQ_COLS = [
    "thr_R_500","thr_R_1000","thr_R_2000","thr_R_3000","thr_R_4000","thr_R_6000","thr_R_8000",
    "thr_L_500","thr_L_1000","thr_L_2000","thr_L_3000","thr_L_4000","thr_L_6000","thr_L_8000",
]
ASSIGN = Path("outputs/json/session4_assignments_any25.csv")
FEATURE = Path("data/processed/frequencia_feature_matrix_v1.csv")
OUTPUT = Path("outputs/json/20_outlier_analysis.json")
TOP_OUT = Path("outputs/json/20_top_outliers_profile.json")
SUBCL_OUT = Path("outputs/json/20_outlier_subclusters.json")
LOG = Path("outputs/logs/20_outlier_analysis.log")

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger(__name__)

if OUTPUT.exists():
    log.info(f"Output já existe: {OUTPUT}. Pulando.")
    raise SystemExit(0)


def main():
    log.info("=" * 60)
    log.info("SESSÃO 5 — TAREFA 2: Análise dos 585 Outliers")
    log.info("=" * 60)

    assign = pd.read_csv(ASSIGN)
    df = pd.read_csv(FEATURE, low_memory=False)
    df = df.merge(assign[["SEQN", "cluster_id", "distance_to_centroid", "outlier_score", "membership_probability"]],
                  on="SEQN", how="inner", validate="one_to_one")

    outliers = df[df["cluster_id"] == -1].copy()
    c0 = df[df["cluster_id"] == 0]
    c1 = df[df["cluster_id"] == 1]
    log.info(f"Outliers: {len(outliers)}, Cluster 0: {len(c0)}, Cluster 1: {len(c1)}")

    # ── 2A: Profile geométrico comparativo ───────────────────────────
    def group_profile(g, label):
        thr_g = g[FREQ_COLS].apply(pd.to_numeric, errors="coerce")
        medians = thr_g.median(skipna=True).to_dict()
        means = thr_g.mean(skipna=True).to_dict()
        r_high = g[["thr_R_3000","thr_R_4000","thr_R_6000","thr_R_8000"]].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        r_low = g[["thr_R_500","thr_R_1000","thr_R_2000"]].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        l_high = g[["thr_L_3000","thr_L_4000","thr_L_6000","thr_L_8000"]].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        l_low = g[["thr_L_500","thr_L_1000","thr_L_2000"]].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        asym = (g[FREQ_COLS[:7]].apply(pd.to_numeric, errors="coerce").values -
                g[FREQ_COLS[7:]].apply(pd.to_numeric, errors="coerce").values)
        asym_mean = float(np.nanmean(np.abs(asym)))
        age = pd.to_numeric(g["RIDAGEYR"], errors="coerce")

        return {
            "label": label, "n": int(len(g)),
            "age_median": round(float(age.median()), 1),
            "pta_low_R": round(float(r_low.mean()), 1),
            "pta_low_L": round(float(l_low.mean()), 1),
            "pta_high_R": round(float(r_high.mean()), 1),
            "pta_high_L": round(float(l_high.mean()), 1),
            "hf_lf_contrast_R": round(float((r_high - r_low).mean()), 1),
            "hf_lf_contrast_L": round(float((l_high - l_low).mean()), 1),
            "asym_mean_abs": round(asym_mean, 1),
            "threshold_medians_db": {k: round(float(v), 1) for k, v in medians.items()},
        }

    profiles_2a = [
        group_profile(c0, "cluster_0"),
        group_profile(c1, "cluster_1"),
        group_profile(outliers, "outliers"),
    ]

    # Desvio dos outliers vs cluster 0 por frequência
    deviance = {}
    med_c0 = c0[FREQ_COLS].apply(pd.to_numeric, errors="coerce").median()
    med_out = outliers[FREQ_COLS].apply(pd.to_numeric, errors="coerce").median()
    for col in FREQ_COLS:
        deviance[col] = round(float(med_out[col] - med_c0[col]), 1) if pd.notna(med_out[col]) and pd.notna(med_c0[col]) else None

    log.info("2A: Profiles geométricos comparativos concluídos")

    # ── 2B: Distribuição de distância ao centroide ───────────────────
    dist = pd.to_numeric(outliers["distance_to_centroid"], errors="coerce").dropna()
    dist_stats = {
        "n": int(len(dist)),
        "mean": round(float(dist.mean()), 4),
        "median": round(float(dist.median()), 4),
        "p10": round(float(dist.quantile(0.10)), 4),
        "p25": round(float(dist.quantile(0.25)), 4),
        "p75": round(float(dist.quantile(0.75)), 4),
        "p90": round(float(dist.quantile(0.90)), 4),
        "p95": round(float(dist.quantile(0.95)), 4),
        "max": round(float(dist.max()), 4),
    }
    threshold_extreme = float(dist.quantile(0.90))
    n_extreme = int((dist >= threshold_extreme).sum())
    log.info(f"2B: Distância ao centroide — mediana={dist_stats['median']}, p90={dist_stats['p90']}")
    log.info(f"    Outliers extremos (top 10%): {n_extreme} pessoas")

    # ── 2C: Top 50 outliers extremos ─────────────────────────────────
    top50 = outliers.nlargest(50, "distance_to_centroid")
    top50_profiles = []
    for _, row in top50.iterrows():
        profile = {
            "SEQN": int(row["SEQN"]),
            "cycle": str(row["cycle"]),
            "age": float(row["RIDAGEYR"]) if pd.notna(row["RIDAGEYR"]) else None,
            "sex": float(row["RIAGENDR"]) if pd.notna(row.get("RIAGENDR")) else None,
            "distance_to_centroid": round(float(row["distance_to_centroid"]), 4),
            "outlier_score": round(float(row["outlier_score"]), 4),
        }
        for col in FREQ_COLS:
            profile[col] = round(float(row[col]), 1) if pd.notna(row[col]) else None
        # Calcular assimetria
        for i, freq in enumerate([500,1000,2000,3000,4000,6000,8000]):
            r = profile.get(f"thr_R_{freq}")
            l = profile.get(f"thr_L_{freq}")
            if r is not None and l is not None:
                profile[f"asym_{freq}"] = round(abs(r - l), 1)
        # Tinnitus
        if "AUQ191" in row.index:
            profile["AUQ191"] = float(row["AUQ191"]) if pd.notna(row["AUQ191"]) else None
        top50_profiles.append(profile)

    log.info(f"2C: Top 50 outliers extremos perfilados")

    # ── 2D: Sub-clustering dos outliers ───────────────────────────────
    outlier_subclusters = {}
    if HDBSCAN_OK and len(outliers) >= 20:
        thr_out = outliers[FREQ_COLS].apply(pd.to_numeric, errors="coerce")
        row_means = thr_out.mean(axis=1, skipna=True)
        X_shape = thr_out.sub(row_means, axis=0).fillna(0.0)
        X = X_shape.to_numpy(dtype=np.float32)

        from sklearn.preprocessing import RobustScaler
        from sklearn.decomposition import PCA
        scaler = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=False)
        X_scaled = scaler.fit_transform(X).astype(np.float32)
        pca = PCA(n_components=min(0.95, 1.0), svd_solver="full", random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_scaled).astype(np.float32)
        log.info(f"2D: PCA outliers: {X_pca.shape[1]} componentes")

        sub_results = []
        for mcs in [3, 5, 10, 15, 20]:
            for ms in [3, 5]:
                if ms > mcs:
                    continue
                c = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean",
                                    cluster_selection_method="eom", core_dist_n_jobs=-1)
                labels = c.fit_predict(X_pca)
                nc = int(len([x for x in np.unique(labels) if x != -1]))
                nn = int(np.sum(labels == -1))
                sizes = {int(k): int(v) for k, v in zip(*np.unique(labels[labels != -1], return_counts=True))} if nc > 0 else {}
                sub_results.append({"min_cluster_size": mcs, "min_samples": ms,
                                    "n_clusters": nc, "n_noise": nn,
                                    "noise_fraction": round(nn/len(labels), 4), "sizes": sizes})
                if nc > 0:
                    log.info(f"    mcs={mcs} ms={ms} → {nc} clusters, ruído={nn/len(labels):.4f}, sizes={sizes}")

        valid_sub = [r for r in sub_results if r["n_clusters"] >= 2]
        if valid_sub:
            best_sub = sorted(valid_sub, key=lambda r: r["noise_fraction"])[0]
        else:
            best_sub = sorted(sub_results, key=lambda r: r["noise_fraction"])[0] if sub_results else None

        outlier_subclusters = {
            "grid": sub_results,
            "best": best_sub,
            "pca_components": int(X_pca.shape[1]),
            "pca_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
        }
    else:
        log.info("2D: Sub-clustering pulado (outliers < 20 ou hdbscan indisponível)")

    # ── Salvar ───────────────────────────────────────────────────────
    result = {
        "script": "20_session5_outlier_analysis.py",
        "random_state": RANDOM_STATE,
        "geometric_profiles": profiles_2a,
        "threshold_deviance_outliers_vs_c0": deviance,
        "distance_stats": dist_stats,
        "extreme_threshold_p90": round(threshold_extreme, 4),
        "n_extreme_outliers": n_extreme,
        "top50_profiles": top50_profiles,
        "outlier_subclusters": outlier_subclusters,
        "status": "EXECUTED — sem rótulo clínico",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Salvar top 50 separadamente para leitura fácil
    TOP_OUT.parent.mkdir(parents=True, exist_ok=True)
    TOP_OUT.write_text(json.dumps(top50_profiles, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    log.info(f"Concluído. Output: {OUTPUT}")
    log.info(f"Top 50 outliers: {TOP_OUT}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
