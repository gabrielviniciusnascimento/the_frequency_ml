#!/usr/bin/env python3
"""
Nome: 12_hdbscan_pca_grid.py
Tarefa: Rodar HDBSCAN no espaço PCA-15, não nas 95 features brutas, para duas políticas H11.
Input: data/processed/frequencia_model_ready_v1.parquet; data/processed/frequencia_model_ready_v1_666cap125.parquet; outputs/json/model_ready_feature_columns_v1.json.
Output: outputs/json/hdbscan_pca_grid_v2.json; outputs/json/pca15_embeddings_*.csv; outputs/json/hdbscan_pca_assignments_best_*.csv.
Dependências: 06_model_ready.py.
"""

import logging
import json
from pathlib import Path

# Núcleo científico — sempre presente
import numpy as np
import pandas as pd
from scipy import stats, spatial, linalg
from scipy.spatial.distance import jensenshannon

# ML — scikit-learn para tudo modelável
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
try:
    import shap
except ImportError:
    shap = None

# Clustering
from sklearn.cluster import KMeans, DBSCAN
try:
    import hdbscan
    from hdbscan.validity import validity_index
except ImportError:
    hdbscan = None
    validity_index = None

# Paralelismo
from joblib import Parallel, delayed
import multiprocessing as mp
N_JOBS = max(mp.cpu_count() - 1, 1)

# ── Constantes ───────────────────────────────────────────────────────
RANDOM_STATE = 42
PCA_N_COMPONENTS = 15
MIN_CLUSTER_SIZES = [30, 50, 100, 200]
MIN_SAMPLES = [5, 10, 20]
DBCV_SAMPLE_SIZE = 5000
NOISE_THRESHOLD_SELECTION = 0.40
POLICIES = {
    "nan": Path("data/processed/frequencia_model_ready_v1.parquet"),
    "cap125": Path("data/processed/frequencia_model_ready_v1_666cap125.parquet"),
}
FEATURE_COLUMNS_JSON = Path("outputs/json/model_ready_feature_columns_v1.json")
OUTPUT_PATH = Path("outputs/json/hdbscan_pca_grid_v2.json")
LOG_PATH = Path("outputs/logs/12_hdbscan_pca_grid.log")
PCA_EMBED_TEMPLATE = "outputs/json/pca15_embeddings_{policy}.csv"
PCA_META_TEMPLATE = "outputs/json/pca15_meta_{policy}.json"
ASSIGN_TEMPLATE = "outputs/json/hdbscan_pca_assignments_best_{policy}.csv"

# ── Logging padronizado ──────────────────────────────────────────────
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── Checkpointing ────────────────────────────────────────────────────
if OUTPUT_PATH.exists():
    log.info(f"Output já existe em {OUTPUT_PATH}. Pulando.")
    raise SystemExit(0)


def qa_matrix(X: np.ndarray, nome: str) -> None:
    log.info(f"QA matrix {nome}: shape={X.shape}, dtype={X.dtype}")
    assert not np.isnan(X).any(), f"ERRO: NaN em {nome}"
    assert np.isfinite(X).all(), f"ERRO: inf em {nome}"


def cluster_stats(labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    clusters = sorted([int(x) for x in np.unique(labels) if int(x) != -1])
    n_clusters = len(clusters)
    n_noise = int(np.sum(labels == -1))
    n = int(labels.shape[0])
    return {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": float(n_noise / max(n, 1)),
        "cluster_sizes": {str(c): int(np.sum(labels == c)) for c in clusters},
    }


def dbcv_sample(X_pca: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> float | None:
    if validity_index is None:
        return None
    non_noise = [x for x in np.unique(labels) if int(x) != -1]
    if len(non_noise) < 2:
        return None
    n = X_pca.shape[0]
    sample_n = min(DBCV_SAMPLE_SIZE, n)
    idx = rng.choice(n, size=sample_n, replace=False)
    labels_s = labels[idx]
    if len([x for x in np.unique(labels_s) if int(x) != -1]) < 2:
        return None
    try:
        return float(validity_index(X_pca[idx].astype(np.float64), labels_s.astype(np.int64), metric="euclidean"))
    except Exception as exc:
        log.warning(f"ANOMALIA DETECTADA — DBCV PCA falhou: {exc}")
        return None


def choose_best(rows: list[dict]) -> tuple[dict, str]:
    eligible = [r for r in rows if r["noise_fraction"] < NOISE_THRESHOLD_SELECTION and r["n_clusters"] >= 2]
    if eligible:
        # Critério pedido: noise_fraction < 0.40 com maior DBCV. Empate: menor ruído.
        best = sorted(eligible, key=lambda r: (-(r["dbcv_sample_estimate"] if r["dbcv_sample_estimate"] is not None else -999), r["noise_fraction"]))[0]
        criterion = "noise_fraction < 0.40 com maior DBCV amostral; empate por menor ruído"
    else:
        # Fallback explícito porque pode não haver run <40% ruído.
        best = sorted(rows, key=lambda r: (r["noise_fraction"], -(r["dbcv_sample_estimate"] if r["dbcv_sample_estimate"] is not None else -999), r["n_clusters"]))[0]
        criterion = "FALLBACK: nenhum run com noise_fraction < 0.40; escolhido menor ruído, empate por maior DBCV e menos clusters"
    return best, criterion


def make_pca(policy: str, path: Path, shape_features: list[str], cycle_map: dict[str, int]) -> tuple[pd.DataFrame, np.ndarray, dict]:
    df = pd.read_parquet(path)
    log.info(f"{policy}: model_ready shape={df.shape}")
    X = df[shape_features].astype("float32").to_numpy(dtype=np.float32, copy=True)
    qa_matrix(X, f"X_{policy}_raw_shape95")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X).astype("float32")
    qa_matrix(X_scaled, f"X_{policy}_scaled")
    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled).astype("float32")
    qa_matrix(X_pca, f"X_{policy}_pca15")

    inverse_cycle = {v: k for k, v in cycle_map.items()}
    emb = pd.DataFrame({
        "SEQN": df["SEQN"].astype("int64"),
        "cycle_code": df["cycle_code"].astype("int16"),
        "cycle": df["cycle_code"].map(inverse_cycle).astype("string"),
        "RIDAGEYR": df["RIDAGEYR"].astype("float32") if "RIDAGEYR" in df.columns else np.nan,
        "RIAGENDR": df["RIAGENDR"].astype("float32") if "RIAGENDR" in df.columns else np.nan,
    })
    for j in range(PCA_N_COMPONENTS):
        emb[f"PC{j+1}"] = X_pca[:, j]
    emb_path = Path(PCA_EMBED_TEMPLATE.format(policy=policy))
    emb.to_csv(emb_path, index=False)

    meta = {
        "policy": policy,
        "n_samples": int(df.shape[0]),
        "n_shape_features": len(shape_features),
        "pca_n_components": PCA_N_COMPONENTS,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
        "pca_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
        "embedding_csv": str(emb_path),
    }
    meta_path = Path(PCA_META_TEMPLATE.format(policy=policy))
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"{policy}: PCA15 salvo {emb_path}; explained_sum={meta['pca_explained_variance_sum']:.6f}")
    return df, X_pca, meta


def process_policy(policy: str, path: Path, shape_features: list[str], cycle_map: dict[str, int]) -> dict:
    if hdbscan is None:
        raise ImportError("hdbscan indisponível")
    df, X_pca, pca_meta = make_pca(policy, path, shape_features, cycle_map)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    fits = {}
    for min_cluster_size in MIN_CLUSTER_SIZES:
        for min_samples in MIN_SAMPLES:
            log.info(f"HDBSCAN-PCA policy={policy} min_cluster_size={min_cluster_size} min_samples={min_samples}")
            model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=False,
                core_dist_n_jobs=N_JOBS,
            )
            labels = model.fit_predict(X_pca)
            st = cluster_stats(labels)
            dbcv = dbcv_sample(X_pca, labels, rng)
            row = {
                "policy": policy,
                "space": "PCA15",
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(min_samples),
                **st,
                "dbcv_sample_estimate": dbcv,
                "dbcv_sample_size": min(DBCV_SAMPLE_SIZE, X_pca.shape[0]) if dbcv is not None else None,
            }
            log.info(f"HDBSCAN-PCA run: {row}")
            rows.append(row)
            fits[(min_cluster_size, min_samples)] = {
                "labels": labels.astype("int32"),
                "probabilities": getattr(model, "probabilities_", np.full(labels.shape, np.nan)).astype("float32"),
                "outlier_scores": getattr(model, "outlier_scores_", np.full(labels.shape, np.nan)).astype("float32"),
            }

    best, criterion = choose_best(rows)
    best_fit = fits[(best["min_cluster_size"], best["min_samples"])]
    assignments = pd.DataFrame({
        "SEQN": df["SEQN"].astype("int64"),
        "cycle_code": df["cycle_code"].astype("int16"),
        "cluster_id": best_fit["labels"],
        "membership_probability": best_fit["probabilities"],
        "outlier_score": best_fit["outlier_scores"],
    })
    assign_path = Path(ASSIGN_TEMPLATE.format(policy=policy))
    assignments.to_csv(assign_path, index=False)

    status = "PRELIMINAR — HDBSCAN no espaço PCA, sem interpretação clínica"
    log.info(f"""
FINDING #HDBSCAN-PCA-{policy}
Descrição: HDBSCAN rodado em 15 componentes PCA para política {policy}; melhor configuração selecionada por critério explícito.
Métrica: noise_fraction = {best['noise_fraction']:.4f}; n_clusters = {best['n_clusters']}; DBCV_amostral = {best['dbcv_sample_estimate']}
N: {df.shape[0]}
Output salvo: {OUTPUT_PATH} e {assign_path}
Status: {status}
""")
    return {
        "policy": policy,
        "pca_meta": pca_meta,
        "grid_results": rows,
        "best_config": best,
        "selection_criterion": criterion,
        "assignments_csv": str(assign_path),
    }


def main():
    log.info("Iniciando HDBSCAN no espaço PCA15 para duas políticas H11...")
    meta = json.loads(FEATURE_COLUMNS_JSON.read_text(encoding="utf-8"))
    shape_features = meta["shape_only_intersection"]
    model_ready_json = json.loads(Path("outputs/json/06_model_ready.json").read_text(encoding="utf-8"))
    cycle_map = model_ready_json["policies"][0]["cycle_map"]
    results = []
    for policy, path in POLICIES.items():
        results.append(process_policy(policy, path, shape_features, cycle_map))
    output = {
        "script": "12_hdbscan_pca_grid.py",
        "random_state": RANDOM_STATE,
        "pca_n_components": PCA_N_COMPONENTS,
        "min_cluster_sizes": MIN_CLUSTER_SIZES,
        "min_samples": MIN_SAMPLES,
        "selection_rule_requested": "noise_fraction < 0.40 com maior DBCV",
        "policies": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Concluído. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
