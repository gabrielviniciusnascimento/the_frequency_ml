#!/usr/bin/env python3
"""
Nome: 09_cluster_profiles.py
Tarefa: Descrever geometricamente os clusters melhores por política H11 e comparar estabilidade ARI.
Input: outputs/json/hdbscan_grid_results.json; outputs/json/hdbscan_assignments_best_*.csv; data/processed/frequencia_model_ready*.parquet; data/processed/frequencia_bruto.csv.
Output: outputs/json/cluster_profiles_v1.json.
Dependências: 08_hdbscan_grid.py.
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
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, adjusted_rand_score
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
POLICIES = {
    "nan": {
        "model_ready": Path("data/processed/frequencia_model_ready_v1.parquet"),
        "assignments": Path("outputs/json/hdbscan_assignments_best_nan.csv"),
    },
    "cap125": {
        "model_ready": Path("data/processed/frequencia_model_ready_v1_666cap125.parquet"),
        "assignments": Path("outputs/json/hdbscan_assignments_best_cap125.csv"),
    },
}
BRUTO_CSV = Path("data/processed/frequencia_bruto.csv")
HDBSCAN_GRID_JSON = Path("outputs/json/hdbscan_grid_results.json")
OUTPUT_PATH = Path("outputs/json/cluster_profiles_v1.json")
LOG_PATH = Path("outputs/logs/09_cluster_profiles.log")
SELECTED_GEOMETRY_FEATURES = [
    "hf_lf_contrast_mean", "hf_lf_contrast_R", "hf_lf_contrast_L",
    "slope_500_8000_R", "slope_500_8000_L", "slope_2000_8000_R", "slope_2000_8000_L",
    "asym_mean", "asym_high", "pta_high_mean_binaural", "pta_high_R", "pta_high_L",
    "notch_4k_R", "notch_4k_L", "recovery_8k_R", "recovery_8k_L"
]

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


def tinnitus_any_from_bruto() -> pd.DataFrame | None:
    if not BRUTO_CSV.exists():
        return None
    usecols = None
    head = pd.read_csv(BRUTO_CSV, nrows=1)
    if "AUQ191" not in head.columns:
        return None
    usecols = ["SEQN", "cycle", "AUQ191"]
    bruto = pd.read_csv(BRUTO_CSV, usecols=usecols, low_memory=False)
    auq = pd.to_numeric(bruto["AUQ191"], errors="coerce")
    # NHANES AUQ191: 1 = Yes, 2 = No; 7/9/missing não codificados como ausência.
    bruto["tinnitus_any"] = np.select([auq == 1, auq == 2], [1.0, 0.0], default=np.nan).astype("float32")
    return bruto[["SEQN", "cycle", "tinnitus_any"]]


def describe_policy(policy: str, spec: dict, tinnitus_df: pd.DataFrame | None) -> dict:
    log.info(f"Descrevendo clusters policy={policy}")
    df = pd.read_parquet(spec["model_ready"])
    assign = pd.read_csv(spec["assignments"])
    if df.shape[0] != assign.shape[0] or not df["SEQN"].astype("int64").equals(assign["SEQN"].astype("int64")):
        raise ValueError(f"Ordem/chaves não coincidem em policy={policy}")
    work = df.copy()
    work["cluster_id"] = assign["cluster_id"].astype("int32")
    work["membership_probability"] = assign["membership_probability"].astype("float32")
    work["outlier_score"] = assign["outlier_score"].astype("float32")
    work["cycle_label"] = assign["cycle"].astype("string")

    if tinnitus_df is not None:
        work = work.merge(tinnitus_df, left_on=["SEQN", "cycle_label"], right_on=["SEQN", "cycle"], how="left", suffixes=("", "_raw"))
    else:
        work["tinnitus_any"] = np.nan

    clusters = sorted([int(c) for c in work["cluster_id"].unique()])
    feature_cols = [c for c in df.columns if c not in ["SEQN", "cycle_code"]]
    selected_cols = [c for c in SELECTED_GEOMETRY_FEATURES if c in work.columns]
    cluster_profiles = {}
    for cid in clusters:
        g = work.loc[work["cluster_id"] == cid]
        medians = g[feature_cols].median(axis=0, skipna=True).astype(float).to_dict()
        selected = g[selected_cols].median(axis=0, skipna=True).astype(float).to_dict() if selected_cols else {}
        age_summary = {}
        if "RIDAGEYR" in g.columns:
            age = pd.to_numeric(g["RIDAGEYR"], errors="coerce")
            age_summary = {"median": float(age.median()), "mean": float(age.mean()), "p25": float(age.quantile(0.25)), "p75": float(age.quantile(0.75))}
        sex_dist = g["RIAGENDR"].value_counts(dropna=False, normalize=False).astype(int).to_dict() if "RIAGENDR" in g.columns else {}
        cycle_dist = g["cycle_label"].value_counts(dropna=False, normalize=False).astype(int).to_dict()
        tinnitus = pd.to_numeric(g["tinnitus_any"], errors="coerce")
        tinnitus_summary = {
            "n_non_missing": int(tinnitus.notna().sum()),
            "rate": float(tinnitus.mean(skipna=True)) if tinnitus.notna().sum() > 0 else None,
        }
        cluster_profiles[str(cid)] = {
            "n": int(g.shape[0]),
            "is_noise": bool(cid == -1),
            "membership_probability_median": float(g["membership_probability"].median(skipna=True)),
            "outlier_score_median": float(g["outlier_score"].median(skipna=True)),
            "age_summary": age_summary,
            "sex_distribution_RIAGENDR_codes": {str(k): int(v) for k, v in sex_dist.items()},
            "cycle_distribution": {str(k): int(v) for k, v in cycle_dist.items()},
            "tinnitus_any_summary_if_available": tinnitus_summary,
            "selected_geometry_medians": selected,
            "all_feature_medians": medians,
        }
    log.info(f"{policy}: clusters descritos={len(cluster_profiles)}; n={work.shape[0]}")
    return {"policy": policy, "n_samples": int(work.shape[0]), "clusters": cluster_profiles}


def compute_ari() -> dict:
    a = pd.read_csv(POLICIES["nan"]["assignments"])
    b = pd.read_csv(POLICIES["cap125"]["assignments"])
    merged = a[["SEQN", "cycle", "cluster_id"]].merge(
        b[["SEQN", "cycle", "cluster_id"]], on=["SEQN", "cycle"], suffixes=("_nan", "_cap125"), validate="one_to_one"
    )
    ari_all = float(adjusted_rand_score(merged["cluster_id_nan"], merged["cluster_id_cap125"]))
    mask = (merged["cluster_id_nan"] != -1) & (merged["cluster_id_cap125"] != -1)
    ari_non_noise = float(adjusted_rand_score(merged.loc[mask, "cluster_id_nan"], merged.loc[mask, "cluster_id_cap125"])) if int(mask.sum()) >= 2 else None
    return {"n_all": int(merged.shape[0]), "ari_all_including_noise": ari_all, "n_non_noise_intersection": int(mask.sum()), "ari_non_noise_intersection": ari_non_noise}


def main():
    log.info("Iniciando perfis de cluster sem rótulo clínico...")
    grid = json.loads(HDBSCAN_GRID_JSON.read_text(encoding="utf-8"))
    tinnitus_df = tinnitus_any_from_bruto()
    policies = []
    for policy, spec in POLICIES.items():
        policies.append(describe_policy(policy, spec, tinnitus_df))
    ari = compute_ari()

    result = {
        "script": "09_cluster_profiles.py",
        "random_state": RANDOM_STATE,
        "no_clinical_labels": True,
        "hdbscan_best_configs": {p["policy"]: next(x for x in grid["policies"] if x["policy"] == p["policy"])["best_config"] for p in policies},
        "h11_policy_stability": ari,
        "policies": policies,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    log.info(f"""
FINDING #H11-ARI
Descrição: Estabilidade geométrica entre políticas H11 computada por Adjusted Rand Index, sem rótulo clínico.
Métrica: ARI_all_including_noise = {ari['ari_all_including_noise']:.4f}; ARI_non_noise_intersection = {ari['ari_non_noise_intersection']}
N: {ari['n_all']}
Output salvo: {OUTPUT_PATH}
Status: PRELIMINAR — estabilidade de cluster, sem interpretação clínica
""")
    log.info(f"Concluído. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
