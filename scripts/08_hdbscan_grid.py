#!/usr/bin/env python3
"""
Nome: 08_hdbscan_grid.py
Tarefa: Rodar grid HDBSCAN 6 combinações × 2 políticas H11 sobre FEATS_SHAPE_ONLY.
Input: data/processed/frequencia_model_ready_v1.parquet; data/processed/frequencia_model_ready_v1_666cap125.parquet; outputs/json/model_ready_feature_columns_v1.json.
Output: outputs/json/hdbscan_grid_results.json; outputs/json/hdbscan_assignments_best_*.csv.
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
MIN_CLUSTER_SIZES = [20, 40, 80]
MIN_SAMPLES = [5, 10]
DBCV_SAMPLE_SIZE = 5000
MAX_ACCEPTABLE_CLUSTERS = 25
MIN_ACCEPTABLE_CLUSTERS = 2
POLICIES = {
    "nan": Path("data/processed/frequencia_model_ready_v1.parquet"),
    "cap125": Path("data/processed/frequencia_model_ready_v1_666cap125.parquet"),
}
FEATURE_COLUMNS_JSON = Path("outputs/json/model_ready_feature_columns_v1.json")
OUTPUT_PATH = Path("outputs/json/hdbscan_grid_results.json")
LOG_PATH = Path("outputs/logs/08_hdbscan_grid.log")
ASSIGN_TEMPLATE = "outputs/json/hdbscan_assignments_best_{policy}.csv"
SCALER_INFO_TEMPLATE = "outputs/json/hdbscan_scaler_info_{policy}.json"

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
    cluster_labels = sorted([int(x) for x in np.unique(labels) if int(x) != -1])
    n_clusters = len(cluster_labels)
    n_noise = int(np.sum(labels == -1))
    n = int(labels.shape[0])
    sizes = {str(c): int(np.sum(labels == c)) for c in cluster_labels}
    return {"n_clusters": n_clusters, "n_noise": n_noise, "noise_fraction": float(n_noise / max(n, 1)), "cluster_sizes": sizes}


def dbcv_sample(X_scaled: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> float | None:
    if validity_index is None:
        return None
    labels = np.asarray(labels)
    non_noise_clusters = [x for x in np.unique(labels) if int(x) != -1]
    if len(non_noise_clusters) < 2:
        return None
    n = X_scaled.shape[0]
    sample_n = min(DBCV_SAMPLE_SIZE, n)
    idx = rng.choice(n, size=sample_n, replace=False)
    labels_s = labels[idx]
    if len([x for x in np.unique(labels_s) if int(x) != -1]) < 2:
        return None
    try:
        return float(validity_index(X_scaled[idx].astype(np.float64), labels_s.astype(np.int64), metric="euclidean"))
    except Exception as exc:
        log.warning(f"ANOMALIA DETECTADA — DBCV falhou em amostra: {exc}")
        return None


def selection_score(row: dict) -> float:
    n_clusters = row["n_clusters"]
    noise_fraction = row["noise_fraction"]
    fragmentation_penalty = max(0, n_clusters - MAX_ACCEPTABLE_CLUSTERS) * 0.02
    too_few_penalty = 1.0 if n_clusters < MIN_ACCEPTABLE_CLUSTERS else 0.0
    return float(noise_fraction + fragmentation_penalty + too_few_penalty)


def process_policy(policy: str, path: Path, shape_features: list[str], cycle_map: dict[str, int]) -> dict:
    if hdbscan is None:
        raise ImportError("hdbscan não disponível")
    log.info(f"HDBSCAN grid política={policy}; input={path}")
    df = pd.read_parquet(path)
    log.info(f"{policy}: model_ready shape={df.shape}")
    X = df[shape_features].astype("float32").to_numpy(dtype=np.float32, copy=True)
    qa_matrix(X, f"X_hdbscan_{policy}_pre_scaler")
    scaler = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=False)
    X_scaled = scaler.fit_transform(X).astype("float32")
    qa_matrix(X_scaled, f"X_hdbscan_{policy}_scaled")

    rng = np.random.default_rng(RANDOM_STATE)
    run_rows = []
    fitted = {}
    for min_cluster_size in MIN_CLUSTER_SIZES:
        for min_samples in MIN_SAMPLES:
            log.info(f"Rodando HDBSCAN policy={policy} min_cluster_size={min_cluster_size} min_samples={min_samples}")
            model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=False,
                core_dist_n_jobs=N_JOBS,
            )
            labels = model.fit_predict(X_scaled)
            stats_row = cluster_stats(labels)
            dbcv = dbcv_sample(X_scaled, labels, rng)
            row = {
                "policy": policy,
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(min_samples),
                **stats_row,
                "dbcv_sample_estimate": dbcv,
                "dbcv_sample_size": min(DBCV_SAMPLE_SIZE, X_scaled.shape[0]) if dbcv is not None else None,
            }
            row["selection_score"] = selection_score(row)
            log.info(f"HDBSCAN run: {row}")
            run_rows.append(row)
            fitted[(min_cluster_size, min_samples)] = {
                "labels": labels.astype("int32"),
                "probabilities": getattr(model, "probabilities_", np.full(labels.shape, np.nan)).astype("float32"),
                "outlier_scores": getattr(model, "outlier_scores_", np.full(labels.shape, np.nan)).astype("float32"),
            }

    acceptable = [r for r in run_rows if MIN_ACCEPTABLE_CLUSTERS <= r["n_clusters"] <= MAX_ACCEPTABLE_CLUSTERS]
    if acceptable:
        # Critério explícito: menor ruído entre soluções não fragmentadas; empate por maior DBCV; empate por menos clusters.
        best = sorted(
            acceptable,
            key=lambda r: (r["noise_fraction"], -(r["dbcv_sample_estimate"] if r["dbcv_sample_estimate"] is not None else -999), r["n_clusters"])
        )[0]
        criterion = f"menor noise_fraction entre runs com {MIN_ACCEPTABLE_CLUSTERS}-{MAX_ACCEPTABLE_CLUSTERS} clusters; empate por maior DBCV amostral e menos clusters"
    else:
        best = sorted(run_rows, key=lambda r: (r["selection_score"], -(r["dbcv_sample_estimate"] if r["dbcv_sample_estimate"] is not None else -999)))[0]
        criterion = "nenhum run dentro do intervalo não-fragmentado; usado menor score = noise_fraction + penalidade de fragmentação/cluster único"

    best_key = (best["min_cluster_size"], best["min_samples"])
    best_fit = fitted[best_key]
    inverse_cycle = {v: k for k, v in cycle_map.items()}
    assignments = pd.DataFrame({
        "SEQN": df["SEQN"].astype("int64"),
        "cycle_code": df["cycle_code"].astype("int16"),
        "cycle": df["cycle_code"].map(inverse_cycle).astype("string"),
        "cluster_id": best_fit["labels"],
        "membership_probability": best_fit["probabilities"],
        "outlier_score": best_fit["outlier_scores"],
    })
    assign_path = Path(ASSIGN_TEMPLATE.format(policy=policy))
    assignments.to_csv(assign_path, index=False)
    log.info(f"{policy}: best assignments salvos em {assign_path}; shape={assignments.shape}")

    scaler_info = {
        "policy": policy,
        "feature_set": "shape_only_intersection",
        "feature_count": len(shape_features),
        "features": shape_features,
        "scaler": "RobustScaler(quantile_range=(25,75), unit_variance=False)",
    }
    scaler_path = Path(SCALER_INFO_TEMPLATE.format(policy=policy))
    scaler_path.write_text(json.dumps(scaler_info, indent=2, ensure_ascii=False), encoding="utf-8")

    log.info(f"""
FINDING #HDBSCAN-GRID-{policy}
Descrição: Melhor configuração geométrica HDBSCAN selecionada para política {policy} sem rótulo clínico.
Métrica: noise_fraction = {best['noise_fraction']:.4f}; n_clusters = {best['n_clusters']}; DBCV_amostral = {best['dbcv_sample_estimate']}
N: {df.shape[0]}
Output salvo: {OUTPUT_PATH} e {assign_path}
Status: PRELIMINAR — clustering não interpretado clinicamente
""")

    return {
        "policy": policy,
        "input": str(path),
        "n_samples": int(df.shape[0]),
        "n_shape_features": int(len(shape_features)),
        "grid_results": run_rows,
        "best_config": best,
        "selection_criterion": criterion,
        "assignments_csv": str(assign_path),
        "scaler_info_json": str(scaler_path),
    }


def main():
    log.info("Iniciando grid HDBSCAN para políticas H11...")
    meta = json.loads(FEATURE_COLUMNS_JSON.read_text(encoding="utf-8"))
    shape_features = meta["shape_only_intersection"]
    model_ready_json = json.loads(Path("outputs/json/06_model_ready.json").read_text(encoding="utf-8"))
    cycle_map = model_ready_json["policies"][0]["cycle_map"]

    results = []
    for policy, path in POLICIES.items():
        results.append(process_policy(policy, path, shape_features, cycle_map))

    output = {
        "script": "08_hdbscan_grid.py",
        "random_state": RANDOM_STATE,
        "min_cluster_sizes": MIN_CLUSTER_SIZES,
        "min_samples": MIN_SAMPLES,
        "dbcv_note": f"DBCV calculado em amostra aleatória máxima de {DBCV_SAMPLE_SIZE} por run, se disponível; não é validação clínica.",
        "selection_rule": f"Preferir menor ruído sem fragmentar demais: aceitável {MIN_ACCEPTABLE_CLUSTERS}-{MAX_ACCEPTABLE_CLUSTERS} clusters; empates por DBCV amostral e menos clusters.",
        "policies": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Concluído. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
