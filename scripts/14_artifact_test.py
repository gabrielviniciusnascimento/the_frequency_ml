#!/usr/bin/env python3
"""
Nome: 14_artifact_test.py
Tarefa: Testar artefatos de idade/ciclo/sexo e rodar clustering estratificado por faixa etária no espaço PCA15.
Input: outputs/json/hdbscan_pca_grid_v2.json; outputs/json/kmeans_grid_v1.json; outputs/json/*assignments*.csv; outputs/json/pca15_embeddings_*.csv.
Output: outputs/json/artifact_test_v1.json.
Dependências: 12_hdbscan_pca_grid.py; 13_kmeans_baseline.py.
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
NOISE_THRESHOLD_FOR_HDBSCAN = 0.40
AGE_BINS = ["lt30", "30_60", "gt60"]
AGE_RULES = {"lt30": lambda s: s < 30, "30_60": lambda s: (s >= 30) & (s <= 60), "gt60": lambda s: s > 60}
STRAT_HDBSCAN_MIN_CLUSTER_SIZE = 30
STRAT_HDBSCAN_MIN_SAMPLES = 5
STRAT_KMEANS_K_VALUES = [3, 4, 5, 6]
POLICIES = ["nan", "cap125"]
PC_COLS = [f"PC{i}" for i in range(1, 16)]
HDBSCAN_JSON = Path("outputs/json/hdbscan_pca_grid_v2.json")
KMEANS_JSON = Path("outputs/json/kmeans_grid_v1.json")
OUTPUT_PATH = Path("outputs/json/artifact_test_v1.json")
LOG_PATH = Path("outputs/logs/14_artifact_test.log")

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


def cramers_v(confusion: pd.DataFrame) -> float:
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    n = confusion.to_numpy().sum()
    if n == 0:
        return np.nan
    r, k = confusion.shape
    return float(np.sqrt((chi2 / n) / max(min(k - 1, r - 1), 1)))


def choose_source(policy: str, hdb: dict, km: dict) -> dict:
    hdb_pol = next(p for p in hdb["policies"] if p["policy"] == policy)
    km_pol = next(p for p in km["policies"] if p["policy"] == policy)
    hdb_best = hdb_pol["best_config"]
    if hdb_best["noise_fraction"] < NOISE_THRESHOLD_FOR_HDBSCAN:
        return {"source": "hdbscan_pca", "assignments": f"outputs/json/hdbscan_pca_assignments_best_{policy}.csv", "best_config": hdb_best, "reason": "HDBSCAN PCA noise_fraction < 0.40"}
    return {"source": "kmeans_pca", "assignments": km_pol["assignments_csv"], "best_config": km_pol["best_config"], "reason": "HDBSCAN PCA não atingiu noise_fraction < 0.40; usando KMeans baseline conforme regra da tarefa"}


def cluster_distributions(assign: pd.DataFrame, source: str) -> tuple[list[dict], dict]:
    work = assign.copy()
    if "cluster_id" not in work.columns:
        raise ValueError("cluster_id ausente")
    age = pd.to_numeric(work["RIDAGEYR"], errors="coerce")
    work["age"] = age
    work["sex"] = pd.to_numeric(work["RIAGENDR"], errors="coerce")
    rows = []
    for cid, g in work.groupby("cluster_id", dropna=False):
        ga = pd.to_numeric(g["age"], errors="coerce")
        cycle_counts = g["cycle"].astype(str).value_counts().sort_index()
        sex_counts = g["sex"].value_counts(dropna=False).sort_index()
        rows.append({
            "cluster_id": int(cid),
            "is_noise": bool(int(cid) == -1) if source.startswith("hdbscan") else False,
            "n": int(g.shape[0]),
            "age_median": float(ga.median()),
            "age_iqr": float(ga.quantile(0.75) - ga.quantile(0.25)),
            "pct_lt30": float((ga < 30).mean()),
            "pct_gt60": float((ga > 60).mean()),
            "pct_gt65": float((ga > 65).mean()),
            "flag_gt80pct_gt65": bool((ga > 65).mean() > 0.80),
            "cycle_counts": {str(k): int(v) for k, v in cycle_counts.items()},
            "sex_counts_RIAGENDR": {str(k): int(v) for k, v in sex_counts.items()},
        })
    # Testes globais: cluster vs ciclo/sexo; Kruskal idade vs cluster.
    non_noise = work.loc[work["cluster_id"] != -1].copy() if source.startswith("hdbscan") else work.copy()
    cycle_tab = pd.crosstab(non_noise["cluster_id"], non_noise["cycle"])
    sex_tab = pd.crosstab(non_noise["cluster_id"], non_noise["sex"])
    cycle_chi = stats.chi2_contingency(cycle_tab, correction=False) if cycle_tab.shape[0] > 1 and cycle_tab.shape[1] > 1 else None
    sex_chi = stats.chi2_contingency(sex_tab, correction=False) if sex_tab.shape[0] > 1 and sex_tab.shape[1] > 1 else None
    age_groups = [pd.to_numeric(g["age"], errors="coerce").dropna().to_numpy() for _, g in non_noise.groupby("cluster_id")]
    age_groups = [g for g in age_groups if len(g) > 0]
    age_kruskal = stats.kruskal(*age_groups) if len(age_groups) > 1 else None
    tests = {
        "cycle_chi_square_global_non_noise": None if cycle_chi is None else {"chi2": float(cycle_chi.statistic), "p": float(cycle_chi.pvalue), "dof": int(cycle_chi.dof), "cramers_v": cramers_v(cycle_tab)},
        "sex_chi_square_global_non_noise": None if sex_chi is None else {"chi2": float(sex_chi.statistic), "p": float(sex_chi.pvalue), "dof": int(sex_chi.dof), "cramers_v": cramers_v(sex_tab)},
        "age_kruskal_global_non_noise": None if age_kruskal is None else {"statistic": float(age_kruskal.statistic), "p": float(age_kruskal.pvalue)},
    }
    return rows, tests


def stratified_clustering(policy: str) -> list[dict]:
    emb = pd.read_csv(f"outputs/json/pca15_embeddings_{policy}.csv")
    results = []
    age = pd.to_numeric(emb["RIDAGEYR"], errors="coerce")
    for group_name, rule in AGE_RULES.items():
        sub = emb.loc[rule(age)].copy()
        X = sub[PC_COLS].astype("float32").to_numpy(dtype=np.float32, copy=True)
        if sub.shape[0] < 100:
            results.append({"policy": policy, "age_group": group_name, "n": int(sub.shape[0]), "status": "SKIPPED_n_lt_100"})
            continue
        # HDBSCAN fixo simples para limpeza intra-idade
        if hdbscan is not None:
            h = hdbscan.HDBSCAN(min_cluster_size=min(STRAT_HDBSCAN_MIN_CLUSTER_SIZE, max(10, X.shape[0] // 20)), min_samples=STRAT_HDBSCAN_MIN_SAMPLES, metric="euclidean")
            labels_h = h.fit_predict(X)
            n_clusters_h = len([x for x in np.unique(labels_h) if int(x) != -1])
            noise_h = int(np.sum(labels_h == -1))
            noise_frac_h = float(noise_h / X.shape[0])
        else:
            n_clusters_h, noise_h, noise_frac_h = None, None, None
        # KMeans melhor por silhouette dentro do estrato
        km_rows = []
        for k in STRAT_KMEANS_K_VALUES:
            if X.shape[0] <= k:
                continue
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
            lab = km.fit_predict(X)
            sil = float(silhouette_score(X, lab, sample_size=min(3000, X.shape[0]), random_state=RANDOM_STATE))
            db = float(davies_bouldin_score(X, lab))
            km_rows.append({"k": k, "silhouette": sil, "davies_bouldin": db})
        best_km = sorted(km_rows, key=lambda r: (-r["silhouette"], r["davies_bouldin"]))[0] if km_rows else None
        results.append({
            "policy": policy,
            "age_group": group_name,
            "n": int(sub.shape[0]),
            "hdbscan_fixed": {"n_clusters": n_clusters_h, "n_noise": noise_h, "noise_fraction": noise_frac_h},
            "kmeans_grid": km_rows,
            "kmeans_best": best_km,
        })
    return results


def main():
    log.info("Iniciando testes de artefato idade/ciclo/sexo...")
    hdb = json.loads(HDBSCAN_JSON.read_text(encoding="utf-8"))
    km = json.loads(KMEANS_JSON.read_text(encoding="utf-8"))
    policy_results = []
    strat_results = []
    for policy in POLICIES:
        source = choose_source(policy, hdb, km)
        assign = pd.read_csv(source["assignments"])
        if "cycle" not in assign.columns:
            emb = pd.read_csv(f"outputs/json/pca15_embeddings_{policy}.csv", usecols=["SEQN", "cycle", "RIDAGEYR", "RIAGENDR"])
            assign = assign.merge(emb, on="SEQN", how="left", validate="one_to_one")
        rows, tests = cluster_distributions(assign, source["source"])
        strat = stratified_clustering(policy)
        strat_results.extend(strat)
        policy_results.append({"policy": policy, "cluster_source_used": source, "cluster_distributions": rows, "artifact_tests": tests})
        log.info(f"""
FINDING #ARTIFACT-{policy}
Descrição: Testes globais de artefato cluster×ciclo, cluster×sexo e cluster×idade computados para fonte {source['source']}.
Métrica: cycle_cramers_v = {tests['cycle_chi_square_global_non_noise']['cramers_v'] if tests['cycle_chi_square_global_non_noise'] else None}; age_kruskal_p = {tests['age_kruskal_global_non_noise']['p'] if tests['age_kruskal_global_non_noise'] else None}
N: {assign.shape[0]}
Output salvo: {OUTPUT_PATH}
Status: PRELIMINAR — teste de artefato, sem rótulo clínico
""")
    output = {"script": "14_artifact_test.py", "random_state": RANDOM_STATE, "policies": policy_results, "stratified_by_age": strat_results}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Concluído. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
