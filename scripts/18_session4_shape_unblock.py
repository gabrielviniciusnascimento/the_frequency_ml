#!/usr/bin/env python3
"""
Nome: 18_session4_shape_unblock.py
Tarefa: Sessão 4 — espaço de shape com filtro ANY25, validação por ciclo,
        e proximidade contínua para projeção do caso pessoal.
Input: data/processed/frequencia_feature_matrix_v1.csv
Output: outputs/json/18_session4_shape_unblock.json
        outputs/json/session4_assignments_any25.csv
        outputs/json/session4_centroids.json
        outputs/json/session4_pca_scaler_params.json
Dependências: 03_features_v1.py (feature matrix)
"""

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

try:
    import hdbscan
    from hdbscan.prediction import approximate_predict
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# ── Constantes ───────────────────────────────────────────────────────
RANDOM_STATE = 42
MIN_COMPLETENESS = 10        # mínimo de frequências válidas por indivíduo (de 14)
AGE_MIN = 20
AGE_MAX = 69
ANY25_THRESHOLD_DB = 25.0
PCA_VARIANCE_TARGET = 0.95
HDBSCAN_MIN_CLUSTER_SIZE_GLOBAL = 150
HDBSCAN_MIN_SAMPLES_GLOBAL = 20
HDBSCAN_MIN_CLUSTER_SIZE_TRAIN = 150
HDBSCAN_MIN_SAMPLES_TRAIN = 20
HDBSCAN_MIN_CLUSTER_SIZE_REF = 50
HDBSCAN_MIN_SAMPLES_REF = 10
MIN_FOLD_SIZE = 100

FEATURE_MATRIX = Path("data/processed/frequencia_feature_matrix_v1.csv")
OUTPUT_PATH = Path("outputs/json/18_session4_shape_unblock.json")
LOG_PATH = Path("outputs/logs/18_session4_shape_unblock.log")
ASSIGN_PATH = Path("outputs/json/session4_assignments_any25.csv")
CENTROID_PATH = Path("outputs/json/session4_centroids.json")
SCALER_PCA_PATH = Path("outputs/json/session4_pca_scaler_params.json")

FREQ_COLS = [
    "thr_R_500", "thr_R_1000", "thr_R_2000", "thr_R_3000",
    "thr_R_4000", "thr_R_6000", "thr_R_8000",
    "thr_L_500", "thr_L_1000", "thr_L_2000", "thr_L_3000",
    "thr_L_4000", "thr_L_6000", "thr_L_8000",
]

# ── Logging ──────────────────────────────────────────────────────────
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

if OUTPUT_PATH.exists():
    log.info(f"Output já existe em {OUTPUT_PATH}. Pulando.")
    raise SystemExit(0)


def main():
    if not HDBSCAN_AVAILABLE:
        raise ImportError("hdbscan não disponível. Instale com: pip install hdbscan")

    log.info("=" * 60)
    log.info("SESSÃO 4 — Shape Unblock Pipeline")
    log.info("=" * 60)

    # ── 1. Carregar feature matrix ───────────────────────────────────
    df_full = pd.read_csv(FEATURE_MATRIX, low_memory=False)
    n_original = len(df_full)
    log.info(f"Feature matrix carregada: {df_full.shape}")

    # Verificar colunas necessárias
    missing_cols = [c for c in FREQ_COLS if c not in df_full.columns]
    if missing_cols:
        raise ValueError(f"Colunas ausentes na feature matrix: {missing_cols}")
    if "RIDAGEYR" not in df_full.columns:
        raise ValueError("RIDAGEYR ausente — necessário para filtro de idade")

    # ── 2. Filtro de idade ───────────────────────────────────────────
    df = df_full.copy()
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    age_mask = (age >= AGE_MIN) & (age <= AGE_MAX)
    n_before_age = int(age_mask.sum())
    df = df[age_mask].copy()
    log.info(f"Filtro idade {AGE_MIN}–{AGE_MAX}: {n_original} → {len(df)} ({len(df)/n_original*100:.1f}%)")

    # ── 3. Filtro de completude ──────────────────────────────────────
    thr_df = df[FREQ_COLS].apply(pd.to_numeric, errors="coerce")
    valid_counts = thr_df.notna().sum(axis=1)
    completeness_mask = valid_counts >= MIN_COMPLETENESS
    n_before_comp = int(completeness_mask.sum())
    df = df[completeness_mask].copy()
    thr_df = thr_df[completeness_mask].copy()
    log.info(f"Filtro completude ≥{MIN_COMPLETENESS}/14: {n_before_age} → {len(df)}")

    # ── 4. Filtro ANY25ALTERED ───────────────────────────────────────
    any25_mask = (thr_df > ANY25_THRESHOLD_DB).any(axis=1)
    n_any25 = int(any25_mask.sum())
    n_total = len(df)
    log.info(f"ANY25 (>25 dB em pelo menos 1 frequência): {n_any25}/{n_total} ({n_any25/n_total*100:.1f}%)")
    log.info(f"Núcleo saudável removido: {n_total - n_any25} ({(n_total-n_any25)/n_total*100:.1f}%)")

    # Interção com 666
    n_666_in_any25 = None
    if "n_no_response_666_thresholds" in df.columns:
        has_666 = pd.to_numeric(df["n_no_response_666_thresholds"], errors="coerce") > 0
        n_666_in_any25 = int((has_666 & any25_mask).sum())
        n_666_outside = int((has_666 & ~any25_mask).sum())
        log.info(f"666 dentro de ANY25: {n_666_in_any25}; fora: {n_666_outside}")

    df_alt = df[any25_mask].copy()
    thr_alt = thr_df[any25_mask].copy()
    log.info(f"Subset ANY25 final: {df_alt.shape}")

    # ── 5. Row-centering (shape puro, sem nível) ─────────────────────
    row_means = thr_alt.mean(axis=1, skipna=True)
    X_shape = thr_alt.sub(row_means, axis=0)
    n_imputed = int(X_shape.isna().sum().sum())
    X_shape = X_shape.fillna(0.0)
    log.info(f"Row-centering aplicado. NaN imputados como 0: {n_imputed}")

    X = X_shape.to_numpy(dtype=np.float32)
    log.info(f"Matriz final para PCA: {X.shape}")

    # ── 6. Scaling + PCA ─────────────────────────────────────────────
    scaler = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=False)
    X_scaled = scaler.fit_transform(X).astype(np.float32)

    pca = PCA(n_components=PCA_VARIANCE_TARGET, svd_solver="full", random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)
    n_pca = int(pca.n_components_)
    var_sum = float(np.sum(pca.explained_variance_ratio_))
    log.info(f"PCA: {n_pca} componentes, variância explicada = {var_sum:.4f}")

    # Salvar parâmetros do scaler + PCA para projeção futura
    scaler_pca_meta = {
        "scaler_center": scaler.center_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "pca_components": pca.components_.tolist(),
        "pca_mean": pca.mean_.tolist(),
        "pca_n_components": n_pca,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "freq_cols": FREQ_COLS,
        "row_centering": True,
        "imputation_value": 0.0,
    }
    SCALER_PCA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALER_PCA_PATH.write_text(json.dumps(scaler_pca_meta, ensure_ascii=False), encoding="utf-8")
    log.info(f"Parâmetros scaler+PCA salvos: {SCALER_PCA_PATH}")

    # ── 7. HDBSCAN global (macro-estrutura) ──────────────────────────
    log.info(f"Rodando HDBSCAN global: min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE_GLOBAL}, min_samples={HDBSCAN_MIN_SAMPLES_GLOBAL}")
    clusterer_global = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE_GLOBAL,
        min_samples=HDBSCAN_MIN_SAMPLES_GLOBAL,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        core_dist_n_jobs=-1,
    )
    labels_global = clusterer_global.fit_predict(X_pca)
    n_clusters = int(len([x for x in np.unique(labels_global) if x != -1]))
    n_noise = int(np.sum(labels_global == -1))
    noise_frac = n_noise / len(labels_global)
    cluster_sizes = {int(c): int(np.sum(labels_global == c)) for c in set(labels_global) if c != -1}
    log.info(f"HDBSCAN global: {n_clusters} clusters, {n_noise} ruído ({noise_frac:.4f})")
    log.info(f"Tamanhos dos clusters: {cluster_sizes}")

    # ── 8. Distâncias contínuas aos centroides ───────────────────────
    centroids = {}
    for cid in sorted(set(labels_global) - {-1}):
        mask_c = labels_global == cid
        centroids[int(cid)] = X_pca[mask_c].mean(axis=0).tolist()

    distances = np.full(len(X_pca), np.nan, dtype=np.float32)
    outlier_scores = clusterer_global.outlier_scores_.astype(np.float32)
    membership_probs = clusterer_global.probabilities_.astype(np.float32)

    # Distância ao centroide do próprio cluster (para não-ruído)
    for cid, centroid_arr in centroids.items():
        mask_c = labels_global == cid
        centroid_np = np.array(centroid_arr, dtype=np.float32)
        dists = np.sqrt(np.sum((X_pca[mask_c] - centroid_np) ** 2, axis=1))
        distances[mask_c] = dists

    # Distância ao centroide mais próximo (para ruído)
    centroid_matrix = np.array(list(centroids.values()), dtype=np.float32) if centroids else np.empty((0, n_pca))
    noise_mask = labels_global == -1
    if noise_mask.any() and len(centroids) > 0:
        for idx in np.where(noise_mask)[0]:
            dists_to_all = np.sqrt(np.sum((centroid_matrix - X_pca[idx]) ** 2, axis=1))
            distances[idx] = float(np.min(dists_to_all))

    # ── 9. Validação por ciclo com approximate_predict ───────────────
    cycle_col = None
    for candidate in ["cycle", "cycle_code"]:
        if candidate in df_alt.columns:
            cycle_col = candidate
            break
    if cycle_col is None:
        raise ValueError("Coluna de ciclo ausente")

    cycles = sorted(df_alt[cycle_col].unique())
    ari_results = []

    log.info(f"Iniciando validação por ciclo ({len(cycles)} ciclos)...")
    for test_cycle in cycles:
        test_mask_arr = (df_alt[cycle_col].values == test_cycle)
        n_test = int(test_mask_arr.sum())
        if n_test < MIN_FOLD_SIZE:
            log.info(f"  Ciclo {test_cycle}: n={n_test} < {MIN_FOLD_SIZE}, pulando")
            continue

        train_mask_arr = ~test_mask_arr
        X_train = X_pca[train_mask_arr]
        X_test = X_pca[test_mask_arr]

        # HDBSCAN no train com prediction_data
        c_train = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE_TRAIN,
            min_samples=HDBSCAN_MIN_SAMPLES_TRAIN,
            metric="euclidean",
            prediction_data=True,
            core_dist_n_jobs=-1,
        )
        c_train.fit(X_train)

        # Predição aproximada no test
        test_labels_pred, strengths = approximate_predict(c_train, X_test)

        # Clustering independente do test como referência
        c_ref = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE_REF,
            min_samples=HDBSCAN_MIN_SAMPLES_REF,
            metric="euclidean",
            core_dist_n_jobs=-1,
        )
        test_labels_ref = c_ref.fit_predict(X_test)

        # ARI entre projeção e referência
        ari = float(adjusted_rand_score(test_labels_ref, test_labels_pred))
        n_pred_clusters = int(len(set(test_labels_pred) - {-1}))
        n_ref_clusters = int(len(set(test_labels_ref) - {-1}))
        n_pred_noise = int(np.sum(test_labels_pred == -1))
        n_ref_noise = int(np.sum(test_labels_ref == -1))

        ari_results.append({
            "cycle": str(test_cycle),
            "n_test": n_test,
            "ari": round(ari, 6),
            "n_pred_clusters": n_pred_clusters,
            "n_ref_clusters": n_ref_clusters,
            "n_pred_noise": n_pred_noise,
            "n_ref_noise": n_ref_noise,
            "mean_strength": round(float(np.mean(strengths)), 6) if len(strengths) > 0 else None,
        })
        log.info(f"  Ciclo {test_cycle}: ARI={ari:.4f}, n={n_test}, pred_c={n_pred_clusters}, ref_c={n_ref_clusters}")

    ari_values = [r["ari"] for r in ari_results]
    mean_ari = float(np.mean(ari_values)) if ari_values else None
    std_ari = float(np.std(ari_values)) if ari_values else None
    log.info(f"ARI médio inter-ciclos: {mean_ari} (±{std_ari})")

    # ── 10. Salvar assignments + distâncias ───────────────────────────
    assign_df = pd.DataFrame({
        "SEQN": df_alt["SEQN"].astype("int64"),
        "cycle": df_alt[cycle_col],
        "RIDAGEYR": pd.to_numeric(df_alt["RIDAGEYR"], errors="coerce").astype("float32"),
        "cluster_id": labels_global.astype("int32"),
        "membership_probability": membership_probs,
        "outlier_score": outlier_scores,
        "distance_to_centroid": distances,
    })
    ASSIGN_PATH.parent.mkdir(parents=True, exist_ok=True)
    assign_df.to_csv(ASSIGN_PATH, index=False)
    log.info(f"Assignments salvos: {ASSIGN_PATH}")

    # Salvar centroides
    CENTROID_PATH.write_text(json.dumps({
        "centroids": centroids,
        "pca_n_components": n_pca,
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Centroides salvos: {CENTROID_PATH}")

    # ── 11. Resumo das features mais discriminantes por cluster ──────
    cluster_feature_summary = {}
    if n_clusters > 0:
        for cid in sorted(cluster_sizes.keys()):
            mask_c = labels_global == cid
            centroid_orig = scaler.inverse_transform(
                pca.inverse_transform(X_pca[mask_c].mean(axis=0, keepdims=True))
            )[0]
            cluster_feature_summary[int(cid)] = {
                "n": cluster_sizes[cid],
                "centroid_thresholds_db": {col: round(float(v), 2) for col, v in zip(FREQ_COLS, centroid_orig)},
                "mean_distance_to_centroid": round(float(distances[mask_c].mean()), 4),
                "median_outlier_score": round(float(outlier_scores[mask_c].median()), 4),
            }

    # ── 12. Salvar JSON final ────────────────────────────────────────
    result = {
        "script": "18_session4_shape_unblock.py",
        "random_state": RANDOM_STATE,
        "filters": {
            "age_range": [AGE_MIN, AGE_MAX],
            "min_completeness": MIN_COMPLETENESS,
            "any25_threshold_db": ANY25_THRESHOLD_DB,
        },
        "sizes": {
            "original": n_original,
            "after_age_filter": n_before_age,
            "after_completeness": n_before_comp,
            "any25_subset": n_total,
            "altered_subset": int(len(df_alt)),
        },
        "any25_stats": {
            "n_any25": n_any25,
            "pct_any25": round(n_any25 / n_total * 100, 1),
            "n_saudavel_removido": n_total - n_any25,
            "pct_saudavel_removido": round((n_total - n_any25) / n_total * 100, 1),
            "n_666_in_any25": n_666_in_any25,
        },
        "row_centering": {
            "method": "subtract row mean of 14 thresholds (skipna)",
            "n_nan_imputed_as_zero": n_imputed,
        },
        "pca": {
            "n_components": n_pca,
            "variance_explained": round(var_sum, 4),
            "per_component": [round(float(v), 6) for v in pca.explained_variance_ratio_],
        },
        "hdbscan_global": {
            "min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE_GLOBAL,
            "min_samples": HDBSCAN_MIN_SAMPLES_GLOBAL,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "noise_fraction": round(noise_frac, 4),
            "cluster_sizes": cluster_sizes,
        },
        "cluster_centroids_thresholds_db": cluster_feature_summary,
        "cross_cycle_validation": {
            "method": "approximate_predict (train) vs independent HDBSCAN (test); ARI",
            "train_min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE_TRAIN,
            "train_min_samples": HDBSCAN_MIN_SAMPLES_TRAIN,
            "ref_min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE_REF,
            "ref_min_samples": HDBSCAN_MIN_SAMPLES_REF,
            "n_folds_tested": len(ari_results),
            "mean_ari": round(mean_ari, 6) if mean_ari is not None else None,
            "std_ari": round(std_ari, 6) if std_ari is not None else None,
            "per_cycle": ari_results,
        },
        "outputs": {
            "assignments_csv": str(ASSIGN_PATH),
            "centroids_json": str(CENTROID_PATH),
            "scaler_pca_params": str(SCALER_PCA_PATH),
        },
        "status": "EXECUTED — sem rótulo clínico",
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    log.info("=" * 60)
    log.info("SESSÃO 4 CONCLUÍDA")
    log.info(f"Clusters: {n_clusters}, Ruído: {noise_frac:.4f}")
    log.info(f"ARI médio inter-ciclos: {mean_ari}")
    log.info(f"Output: {OUTPUT_PATH}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
