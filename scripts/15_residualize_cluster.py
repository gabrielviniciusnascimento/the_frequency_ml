#!/usr/bin/env python3
"""
Nome: 15_residualize_cluster.py
Tarefa: Residualizar features audiométricas por idade + idade² + sexo e rodar HDBSCAN/KMeans nos resíduos.
Input: data/processed/frequencia_model_ready*.parquet; outputs/json/model_ready_feature_columns_v1.json.
Output: data/processed/frequencia_residualizado_v1*.parquet; outputs/json/clustering_residualizado_v1.json.
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
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, silhouette_score, davies_bouldin_score
from sklearn.linear_model import LinearRegression
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
HDBSCAN_MIN_CLUSTER_SIZE = 50
HDBSCAN_MIN_SAMPLES = 10
K_VALUES = [3, 4, 5, 6, 8, 10]
POLICIES = {
    "nan": {"input": Path("data/processed/frequencia_model_ready_v1.parquet"), "resid": Path("data/processed/frequencia_residualizado_v1.parquet")},
    "cap125": {"input": Path("data/processed/frequencia_model_ready_v1_666cap125.parquet"), "resid": Path("data/processed/frequencia_residualizado_v1_666cap125.parquet")},
}
FEATURE_COLUMNS_JSON = Path("outputs/json/model_ready_feature_columns_v1.json")
OUTPUT_PATH = Path("outputs/json/clustering_residualizado_v1.json")
LOG_PATH = Path("outputs/logs/15_residualize_cluster.log")
RESID_ASSIGN_TEMPLATE = "outputs/json/residualized_assignments_{policy}.csv"

# ── Logging padronizado ──────────────────────────────────────────────
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

if OUTPUT_PATH.exists():
    log.info(f"Output já existe em {OUTPUT_PATH}. Pulando.")
    raise SystemExit(0)


def residualize(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[dict]]:
    if "RIDAGEYR" not in df.columns or "RIAGENDR" not in df.columns:
        raise ValueError("RIDAGEYR/RIAGENDR necessários para residualização")
    age = df["RIDAGEYR"].astype("float64").to_numpy()
    sex = df["RIAGENDR"].astype("float64").to_numpy()
    design = np.column_stack([age, age ** 2, sex]).astype("float64")
    resid = pd.DataFrame({"SEQN": df["SEQN"].astype("int64"), "cycle_code": df["cycle_code"].astype("int16"), "RIDAGEYR": df["RIDAGEYR"].astype("float32"), "RIAGENDR": df["RIAGENDR"].astype("float32")})
    stats_rows = []
    for feat in features:
        y = df[feat].astype("float64").to_numpy()
        lr = LinearRegression()
        lr.fit(design, y)
        pred = lr.predict(design)
        r = y - pred
        resid[feat] = r.astype("float32")
        stats_rows.append({"feature": feat, "r2_age_sex_model": float(lr.score(design, y)), "coef_age": float(lr.coef_[0]), "coef_age2": float(lr.coef_[1]), "coef_sex": float(lr.coef_[2])})
    return resid, stats_rows


def cluster_resid(policy: str, resid: pd.DataFrame, features: list[str]) -> dict:
    X = resid[features].astype("float32").to_numpy(dtype=np.float32, copy=True)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X).astype("float32")
    pca = PCA(n_components=min(PCA_N_COMPONENTS, X_scaled.shape[1]), random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled).astype("float32")
    # HDBSCAN fixo nos resíduos
    if hdbscan is not None:
        h = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, min_samples=HDBSCAN_MIN_SAMPLES, metric="euclidean")
        labels_h = h.fit_predict(X_pca).astype("int32")
        h_stats = {"min_cluster_size": HDBSCAN_MIN_CLUSTER_SIZE, "min_samples": HDBSCAN_MIN_SAMPLES, "n_clusters": int(len([x for x in np.unique(labels_h) if x != -1])), "n_noise": int(np.sum(labels_h == -1)), "noise_fraction": float(np.mean(labels_h == -1))}
    else:
        labels_h = np.full(X_pca.shape[0], -1, dtype="int32")
        h_stats = None
    # KMeans grid nos resíduos
    km_rows = []
    labels_best = None
    for k in K_VALUES:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X_pca)
        sil = float(silhouette_score(X_pca, labels, sample_size=min(5000, X_pca.shape[0]), random_state=RANDOM_STATE))
        db = float(davies_bouldin_score(X_pca, labels))
        row = {"k": int(k), "silhouette_sample": sil, "davies_bouldin": db, "inertia": float(km.inertia_)}
        km_rows.append(row)
    max_sil = max(r["silhouette_sample"] for r in km_rows)
    km_best = sorted([r for r in km_rows if r["silhouette_sample"] >= 0.95 * max_sil], key=lambda r: (r["k"], r["davies_bouldin"]))[0]
    km_final = KMeans(n_clusters=km_best["k"], random_state=RANDOM_STATE, n_init=20)
    labels_k = km_final.fit_predict(X_pca).astype("int32")
    assign = resid[["SEQN", "cycle_code", "RIDAGEYR", "RIAGENDR"]].copy()
    assign["hdbscan_resid_cluster"] = labels_h
    assign["kmeans_resid_cluster"] = labels_k
    assign_path = Path(RESID_ASSIGN_TEMPLATE.format(policy=policy))
    assign.to_csv(assign_path, index=False)
    log.info(f"""
FINDING #RESID-CLUSTER-{policy}
Descrição: Clustering nos resíduos após remover efeito linear/quadrático de idade e sexo, sem rótulo clínico.
Métrica: HDBSCAN_noise_fraction = {h_stats['noise_fraction'] if h_stats else None}; KMeans_best_silhouette = {km_best['silhouette_sample']:.4f}; KMeans_k = {km_best['k']}
N: {X_pca.shape[0]}
Output salvo: {OUTPUT_PATH} e {assign_path}
Status: PRELIMINAR — estrutura residual, sem interpretação clínica
""")
    return {"policy": policy, "residualized_matrix": None, "pca_explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)), "hdbscan_fixed": h_stats, "kmeans_grid": km_rows, "kmeans_best": km_best, "assignments_csv": str(assign_path)}


def main():
    log.info("Iniciando residualização por idade/sexo e clustering nos resíduos...")
    meta = json.loads(FEATURE_COLUMNS_JSON.read_text(encoding="utf-8"))
    features = meta["shape_only_intersection"]
    results = []
    resid_stats = {}
    for policy, spec in POLICIES.items():
        df = pd.read_parquet(spec["input"])
        resid, stats_rows = residualize(df, features)
        spec["resid"].parent.mkdir(parents=True, exist_ok=True)
        resid.to_parquet(spec["resid"], index=False)
        resid.to_csv(str(spec["resid"]).replace(".parquet", ".csv"), index=False)
        r = cluster_resid(policy, resid, features)
        r["residualized_matrix"] = str(spec["resid"])
        results.append(r)
        resid_stats[policy] = stats_rows
    output = {"script": "15_residualize_cluster.py", "random_state": RANDOM_STATE, "residualization_model": "feature ~ age + age^2 + sex", "policies": results, "residualization_feature_stats": resid_stats}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Concluído. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
