#!/usr/bin/env python3
"""
Nome: 10_rf_surrogate.py
Tarefa: Treinar Random Forest surrogate para explicar clusters geométricos HDBSCAN sem rótulo clínico.
Input: data/processed/frequencia_model_ready*.parquet; outputs/json/hdbscan_assignments_best_*.csv; outputs/json/model_ready_feature_columns_v1.json.
Output: outputs/json/rf_surrogate_v1.json.
Dependências: 08_hdbscan_grid.py; 09_cluster_profiles.py recomendado.
"""

import logging
import json
from pathlib import Path
from collections import Counter, defaultdict

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
from sklearn.inspection import permutation_importance
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
RF_N_ESTIMATORS = 200
PERMUTATION_N_REPEATS = 5
PERMUTATION_SAMPLE_SIZE = 5000
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
FEATURE_COLUMNS_JSON = Path("outputs/json/model_ready_feature_columns_v1.json")
OUTPUT_PATH = Path("outputs/json/rf_surrogate_v1.json")
LOG_PATH = Path("outputs/logs/10_rf_surrogate.log")

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


def qa_dataframe(df: pd.DataFrame, nome: str = "df", target: str | None = None) -> None:
    """Roda antes de qualquer fit()."""
    log.info(f"\n{'='*50}\nQA: {nome}\n{'='*50}")
    log.info(f"Shape: {df.shape}")
    nulls = df.isnull().sum()
    log.info(f"Nulos:\n{nulls[nulls > 0]}")
    log.info(f"Dtypes problemáticos:\n{df.dtypes[df.dtypes == 'object']}")
    if target and target in df.columns:
        log.info(f"Target ({target}) — distribuição:\n{df[target].describe()}")
    assert df.isnull().sum().sum() == 0, "ERRO: Nulos no dataset antes do modelo"
    assert not (df.dtypes == "object").any(), "ERRO: object dtype no dataset antes do modelo"
    log.info("QA passou ✓")


def top_split_features(rf: RandomForestClassifier, features: list[str]) -> list[dict]:
    counts = Counter()
    thresholds = defaultdict(list)
    for estimator in rf.estimators_:
        tree = estimator.tree_
        used = tree.feature
        thresh = tree.threshold
        mask = used >= 0
        for idx, thr in zip(used[mask], thresh[mask]):
            fname = features[int(idx)]
            counts[fname] += 1
            thresholds[fname].append(float(thr))
    rows = []
    for fname, count in counts.most_common(10):
        th = np.array(thresholds[fname], dtype=float)
        rows.append({"feature": fname, "split_count": int(count), "threshold_median_scaled": float(np.median(th)), "threshold_p25_scaled": float(np.quantile(th, 0.25)), "threshold_p75_scaled": float(np.quantile(th, 0.75))})
    return rows


def process_policy(policy: str, spec: dict, shape_features: list[str]) -> dict:
    log.info(f"RF surrogate policy={policy}")
    df = pd.read_parquet(spec["model_ready"])
    assignments = pd.read_csv(spec["assignments"])
    if df.shape[0] != assignments.shape[0] or not df["SEQN"].astype("int64").equals(assignments["SEQN"].astype("int64")):
        raise ValueError(f"Chaves/ordem não coincidem em {policy}")
    work = df[["SEQN", "cycle_code"] + shape_features].copy()
    work["cluster_id"] = assignments["cluster_id"].astype("int32")
    work = work.loc[work["cluster_id"] != -1].copy()
    if work["cluster_id"].nunique() < 2:
        raise RuntimeError(f"Menos de 2 clusters não-ruído em {policy}; surrogate sem sentido")

    qa_dataframe(work.drop(columns=["SEQN"]), nome=f"rf_surrogate_input_{policy}", target="cluster_id")
    X_raw = work[shape_features].astype("float32").to_numpy(dtype=np.float32, copy=True)
    y_raw = work["cluster_id"].astype("int32").to_numpy()
    groups = work["cycle_code"].astype("int16").to_numpy()

    scaler = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=False)
    X = scaler.fit_transform(X_raw).astype("float32")
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced_subsample",
        max_features="sqrt",
    )
    n_groups = len(np.unique(groups))
    cv_scores = None
    if n_groups >= 2:
        n_splits = min(5, n_groups)
        try:
            cv = GroupKFold(n_splits=n_splits)
            cv_scores = cross_val_score(rf, X, y, cv=cv, groups=groups, scoring="accuracy", n_jobs=N_JOBS).astype(float).tolist()
            log.info(f"{policy}: GroupKFold accuracy={cv_scores}")
        except Exception as exc:
            log.warning(f"ANOMALIA DETECTADA — cross_val_score falhou em {policy}: {exc}")
            cv_scores = None

    rf.fit(X, y)
    pred = rf.predict(X)
    train_accuracy = float(accuracy_score(y, pred))
    gini = pd.DataFrame({"feature": shape_features, "importance": rf.feature_importances_.astype(float)}).sort_values("importance", ascending=False)

    rng = np.random.default_rng(RANDOM_STATE)
    sample_n = min(PERMUTATION_SAMPLE_SIZE, X.shape[0])
    idx = rng.choice(X.shape[0], size=sample_n, replace=False)
    perm = permutation_importance(
        rf, X[idx], y[idx], n_repeats=PERMUTATION_N_REPEATS, random_state=RANDOM_STATE, n_jobs=N_JOBS, scoring="accuracy"
    )
    perm_df = pd.DataFrame({
        "feature": shape_features,
        "importance_mean": perm.importances_mean.astype(float),
        "importance_std": perm.importances_std.astype(float),
    }).sort_values("importance_mean", ascending=False)

    split_rows = top_split_features(rf, shape_features)
    class_counts = pd.Series(y_raw).value_counts().sort_index().astype(int).to_dict()

    log.info(f"""
FINDING #RF-SURROGATE-{policy}
Descrição: Random Forest surrogate treinado para reproduzir cluster_id geométrico HDBSCAN, excluindo ruído -1.
Métrica: train_accuracy = {train_accuracy:.4f}; cv_accuracy_mean = {float(np.mean(cv_scores)) if cv_scores else None}
N: {X.shape[0]}
Output salvo: {OUTPUT_PATH}
Status: PRELIMINAR — abertura de caixa preta do clustering, sem rótulo clínico
""")

    return {
        "policy": policy,
        "n_samples_non_noise": int(X.shape[0]),
        "n_shape_features": int(len(shape_features)),
        "n_classes": int(len(le.classes_)),
        "cluster_class_counts_original_ids": {str(k): int(v) for k, v in class_counts.items()},
        "rf_params": {"n_estimators": RF_N_ESTIMATORS, "n_jobs": -1, "random_state": RANDOM_STATE, "class_weight": "balanced_subsample", "max_features": "sqrt"},
        "train_accuracy": train_accuracy,
        "groupkfold_accuracy_scores": cv_scores,
        "groupkfold_accuracy_mean": float(np.mean(cv_scores)) if cv_scores else None,
        "gini_importance_top20": gini.head(20).to_dict(orient="records"),
        "permutation_importance_top20": perm_df.head(20).to_dict(orient="records"),
        "permutation_sample_size": int(sample_n),
        "permutation_n_repeats": PERMUTATION_N_REPEATS,
        "top10_split_features": split_rows,
    }


def main():
    log.info("Iniciando RF surrogate para duas políticas H11...")
    meta = json.loads(FEATURE_COLUMNS_JSON.read_text(encoding="utf-8"))
    shape_features = meta["shape_only_intersection"]
    results = []
    for policy, spec in POLICIES.items():
        results.append(process_policy(policy, spec, shape_features))

    output = {"script": "10_rf_surrogate.py", "random_state": RANDOM_STATE, "feature_set": "shape_only_intersection", "policies": results}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Concluído. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
