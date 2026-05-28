#!/usr/bin/env python3
"""
Nome: 13_kmeans_baseline.py
Tarefa: Rodar KMeans baseline no espaço PCA15 para comparar geometria com HDBSCAN.
Input: outputs/json/pca15_embeddings_*.csv.
Output: outputs/json/kmeans_grid_v1.json; outputs/json/kmeans_assignments_best_*.csv.
Dependências: 12_hdbscan_pca_grid.py.
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
try:
    import shap
except ImportError:
    shap = None

# Clustering
from sklearn.cluster import KMeans, DBSCAN
try:
    import hdbscan
except ImportError:
    hdbscan = None

# Paralelismo
from joblib import Parallel, delayed
import multiprocessing as mp
N_JOBS = max(mp.cpu_count() - 1, 1)

# ── Constantes ───────────────────────────────────────────────────────
RANDOM_STATE = 42
K_VALUES = [3, 4, 5, 6, 8, 10]
SILHOUETTE_SAMPLE_SIZE = 5000
POLICIES = {
    "nan": Path("outputs/json/pca15_embeddings_nan.csv"),
    "cap125": Path("outputs/json/pca15_embeddings_cap125.csv"),
}
OUTPUT_PATH = Path("outputs/json/kmeans_grid_v1.json")
LOG_PATH = Path("outputs/logs/13_kmeans_baseline.log")
ASSIGN_TEMPLATE = "outputs/json/kmeans_assignments_best_{policy}.csv"
PC_COLS = [f"PC{i}" for i in range(1, 16)]

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


def choose_best(rows: list[dict]) -> tuple[dict, str]:
    # "Elbow de silhouette" operacional: escolher menor k cujo silhouette esteja até 95% do máximo.
    max_sil = max(r["silhouette_sample"] for r in rows)
    candidates = [r for r in rows if r["silhouette_sample"] >= 0.95 * max_sil]
    best = sorted(candidates, key=lambda r: (r["k"], r["davies_bouldin"]))[0]
    criterion = "menor k com silhouette >= 95% do máximo; empate por menor Davies-Bouldin"
    return best, criterion


def process_policy(policy: str, path: Path) -> dict:
    log.info(f"KMeans baseline policy={policy}; input={path}")
    emb = pd.read_csv(path)
    X = emb[PC_COLS].astype("float32").to_numpy(dtype=np.float32, copy=True)
    if np.isnan(X).any() or not np.isfinite(X).all():
        raise ValueError(f"NaN/inf em PCA embeddings {policy}")
    rows = []
    labels_by_k = {}
    for k in K_VALUES:
        log.info(f"KMeans policy={policy} k={k}")
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X)
        sil = float(silhouette_score(X, labels, sample_size=min(SILHOUETTE_SAMPLE_SIZE, X.shape[0]), random_state=RANDOM_STATE))
        db = float(davies_bouldin_score(X, labels))
        row = {"policy": policy, "k": int(k), "silhouette_sample": sil, "silhouette_sample_size": min(SILHOUETTE_SAMPLE_SIZE, X.shape[0]), "davies_bouldin": db, "inertia": float(km.inertia_)}
        log.info(f"KMeans run: {row}")
        rows.append(row)
        labels_by_k[k] = labels.astype("int32")
    best, criterion = choose_best(rows)
    best_labels = labels_by_k[best["k"]]
    assign = emb[["SEQN", "cycle_code", "cycle", "RIDAGEYR", "RIAGENDR"]].copy()
    assign["cluster_id"] = best_labels
    assign_path = Path(ASSIGN_TEMPLATE.format(policy=policy))
    assign.to_csv(assign_path, index=False)
    log.info(f"""
FINDING #KMEANS-PCA-{policy}
Descrição: KMeans baseline no espaço PCA15 selecionado por critério de silhouette/elbow operacional, sem rótulo clínico.
Métrica: silhouette_sample = {best['silhouette_sample']:.4f}; davies_bouldin = {best['davies_bouldin']:.4f}; k = {best['k']}
N: {X.shape[0]}
Output salvo: {OUTPUT_PATH} e {assign_path}
Status: PRELIMINAR — baseline geométrico forçado, sem ruído
""")
    return {"policy": policy, "grid_results": rows, "best_config": best, "selection_criterion": criterion, "assignments_csv": str(assign_path)}


def main():
    log.info("Iniciando KMeans baseline...")
    results = [process_policy(policy, path) for policy, path in POLICIES.items()]
    output = {"script": "13_kmeans_baseline.py", "random_state": RANDOM_STATE, "k_values": K_VALUES, "policies": results}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Concluído. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
