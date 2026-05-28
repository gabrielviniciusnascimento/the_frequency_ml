#!/usr/bin/env python3
"""
Nome: 19_session5_subdivide_cluster0.py
Tarefa: Rodar HDBSCAN dentro do cluster 0 (7098 pessoas) para buscar subestrutura.
Input: outputs/json/session4_assignments_any25.csv, data/processed/frequencia_feature_matrix_v1.csv
Output: outputs/json/19_subdivide_cluster0.json, outputs/json/19_subcluster_assignments.csv
"""

import logging, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
try:
    import hdbscan
    from hdbscan.validity import validity_index
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
OUTPUT = Path("outputs/json/19_subdivide_cluster0.json")
LOG = Path("outputs/logs/19_subdivide_cluster0.log")
ASSIGN_OUT = Path("outputs/json/19_subcluster_assignments.csv")

MCS_LIST = [5, 10, 20, 30, 50]
MS_LIST = [3, 5, 10]

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger(__name__)

if OUTPUT.exists():
    log.info(f"Output já existe: {OUTPUT}. Pulando.")
    raise SystemExit(0)


def main():
    if not HDBSCAN_OK:
        raise ImportError("hdbscan não disponível")

    log.info("=" * 60)
    log.info("SESSÃO 5 — TAREFA 1: Subdivisão do Cluster 0")
    log.info("=" * 60)

    # Carregar assignments e filtrar cluster 0
    assign = pd.read_csv(ASSIGN)
    c0 = assign[assign["cluster_id"] == 0].copy()
    log.info(f"Cluster 0: {len(c0)} pessoas")

    # Carregar feature matrix e filtrar SEQNs do cluster 0
    df = pd.read_csv(FEATURE, low_memory=False)
    df = df[df["SEQN"].isin(c0["SEQN"])].copy()
    df = df.merge(c0[["SEQN"]], on="SEQN", how="inner", validate="one_to_one")
    log.info(f"Feature matrix cluster 0: {df.shape}")

    # Extrair 14 limiares e row-centering (como Sessão 4)
    thr = df[FREQ_COLS].apply(pd.to_numeric, errors="coerce")
    row_means = thr.mean(axis=1, skipna=True)
    X_shape = thr.sub(row_means, axis=0).fillna(0.0)
    X = X_shape.to_numpy(dtype=np.float32)

    # Scaling + PCA (mesma lógica da Sessão 4 — não reajustar parâmetros globais)
    scaler = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=False)
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    pca = PCA(n_components=0.95, svd_solver="full", random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)
    log.info(f"PCA dentro do cluster 0: {X_pca.shape[1]} componentes, variância={pca.explained_variance_ratio_.sum():.4f}")

    # Grid search
    rng = np.random.default_rng(RANDOM_STATE)
    grid_results = []
    fits = {}

    for mcs in MCS_LIST:
        for ms in MS_LIST:
            if ms > mcs:
                continue
            c = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="euclidean",
                                cluster_selection_method="eom", core_dist_n_jobs=-1)
            labels = c.fit_predict(X_pca)
            nc = int(len([x for x in np.unique(labels) if x != -1]))
            nn = int(np.sum(labels == -1))
            nf = nn / len(labels)

            # DBCV em amostra
            dbcv = None
            if nc >= 2:
                try:
                    idx = rng.choice(len(X_pca), size=min(5000, len(X_pca)), replace=False)
                    sub_labels = labels[idx]
                    if len([x for x in np.unique(sub_labels) if x != -1]) >= 2:
                        dbcv = float(validity_index(X_pca[idx].astype(np.float64),
                                                     sub_labels.astype(np.int64), metric="euclidean"))
                except Exception as e:
                    log.warning(f"DBCV falhou mcs={mcs} ms={ms}: {e}")

            row = {"min_cluster_size": mcs, "min_samples": ms, "n_clusters": nc,
                   "n_noise": nn, "noise_fraction": round(nf, 4), "dbcv_sample": dbcv}
            grid_results.append(row)
            fits[(mcs, ms)] = labels

            sizes = {int(k): int(v) for k, v in zip(*np.unique(labels[labels != -1], return_counts=True))} if nc > 0 else {}
            log.info(f"  mcs={mcs:3d} ms={ms:2d} → {nc} clusters, ruído={nf:.4f}, DBCV={dbcv}, sizes={sizes}")

    # Melhor config: menor ruído com ≥2 clusters
    valid = [r for r in grid_results if r["n_clusters"] >= 2]
    if valid:
        best = sorted(valid, key=lambda r: (r["noise_fraction"], -(r["dbcv_sample"] or -999)))[0]
    else:
        best = sorted(grid_results, key=lambda r: r["noise_fraction"])[0]

    best_key = (best["min_cluster_size"], best["min_samples"])
    best_labels = fits[best_key]
    best["selection_criterion"] = "menor ruído com ≥2 clusters; empate por DBCV" if valid else "menor ruído (fallback)"

    log.info(f"\nMelhor config: mcs={best['min_cluster_size']}, ms={best['min_samples']}")
    log.info(f"  Clusters: {best['n_clusters']}, Ruído: {best['noise_fraction']}")

    # Perfil dos sub-clusters
    profiles = {}
    for cid in sorted(set(best_labels)):
        mask = best_labels == cid
        g = df[mask]
        is_noise = cid == -1
        age = pd.to_numeric(g["RIDAGEYR"], errors="coerce")
        sex = pd.to_numeric(g["RIAGENDR"], errors="coerce")

        thr_g = g[FREQ_COLS].apply(pd.to_numeric, errors="coerce")
        r_high = g[["thr_R_3000","thr_R_4000","thr_R_6000","thr_R_8000"]].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        r_low = g[["thr_R_500","thr_R_1000","thr_R_2000"]].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        l_high = g[["thr_L_3000","thr_L_4000","thr_L_6000","thr_L_8000"]].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)
        l_low = g[["thr_L_500","thr_L_1000","thr_L_2000"]].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)

        # Slope
        slope_r = (thr_g["thr_R_8000"] - thr_g["thr_R_500"]) / np.log2(8000/500)
        slope_l = (thr_g["thr_L_8000"] - thr_g["thr_L_500"]) / np.log2(8000/500)

        tin = pd.to_numeric(g["AUQ191"], errors="coerce") if "AUQ191" in g.columns else pd.Series(dtype=float)

        profiles[str(cid)] = {
            "n": int(mask.sum()),
            "is_noise": bool(is_noise),
            "age_median": round(float(age.median()), 1),
            "age_mean": round(float(age.mean()), 1),
            "sex_counts": {str(k): int(v) for k, v in sex.value_counts(dropna=False).items()},
            "pta_low_R": round(float(r_low.mean()), 1),
            "pta_low_L": round(float(l_low.mean()), 1),
            "pta_high_R": round(float(r_high.mean()), 1),
            "pta_high_L": round(float(l_high.mean()), 1),
            "hf_lf_contrast_R": round(float((r_high - r_low).mean()), 1),
            "hf_lf_contrast_L": round(float((l_high - l_low).mean()), 1),
            "slope_500_8000_R": round(float(slope_r.mean()), 2),
            "slope_500_8000_L": round(float(slope_l.mean()), 2),
            "tinnitus_rate": round(float((tin == 1).mean()), 4) if tin.notna().sum() > 10 else None,
            "cycle_distribution": {str(k): int(v) for k, v in g["cycle"].value_counts().sort_index().items()},
        }
        label = f"RUÍDO" if is_noise else f"SUB-CLUSTER {cid}"
        log.info(f"  {label}: n={profiles[str(cid)]['n']}, idade_med={profiles[str(cid)]['age_median']}, "
                 f"PTA_high_R={profiles[str(cid)]['pta_high_R']}, PTA_high_L={profiles[str(cid)]['pta_high_L']}")

    # Salvar assignments
    assign_sub = pd.DataFrame({
        "SEQN": df["SEQN"].astype("int64"),
        "cycle": df["cycle"],
        "RIDAGEYR": pd.to_numeric(df["RIDAGEYR"], errors="coerce"),
        "sub_cluster_id": best_labels.astype("int32"),
    })
    ASSIGN_OUT.parent.mkdir(parents=True, exist_ok=True)
    assign_sub.to_csv(ASSIGN_OUT, index=False)

    # Salvar JSON
    result = {
        "script": "19_session5_subdivide_cluster0.py",
        "random_state": RANDOM_STATE,
        "n_cluster0": len(c0),
        "pca_components": int(X_pca.shape[1]),
        "pca_variance": round(float(pca.explained_variance_ratio_.sum()), 4),
        "grid_results": grid_results,
        "best_config": best,
        "sub_cluster_profiles": profiles,
        "status": "EXECUTED — sem rótulo clínico",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    log.info(f"Concluído. Output: {OUTPUT}")
    log.info(f"Assignments: {ASSIGN_OUT}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
