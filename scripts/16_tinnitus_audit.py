#!/usr/bin/env python3
"""
Nome: 16_tinnitus_audit.py
Tarefa: Auditar disponibilidade/distribuição de tinnitus AUQ191 e correlação com pta_high/hf_lf_contrast.
Input: data/processed/frequencia_bruto.csv; data/processed/frequencia_feature_matrix_v1.csv.
Output: outputs/json/tinnitus_audit_v1.json.
Dependências: 03_features_v1.py.
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
except ImportError:
    hdbscan = None

# Paralelismo
from joblib import Parallel, delayed
import multiprocessing as mp
N_JOBS = max(mp.cpu_count() - 1, 1)

RANDOM_STATE = 42
BRUTO_CSV = Path("data/processed/frequencia_bruto.csv")
FEATURE_CSV = Path("data/processed/frequencia_feature_matrix_v1.csv")
OUTPUT_PATH = Path("outputs/json/tinnitus_audit_v1.json")
LOG_PATH = Path("outputs/logs/16_tinnitus_audit.log")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()])
log = logging.getLogger(__name__)

if OUTPUT_PATH.exists():
    log.info(f"Output já existe em {OUTPUT_PATH}. Pulando.")
    raise SystemExit(0)


def corr_pair(df: pd.DataFrame, x: str, y: str) -> dict:
    sub = df[[x, y]].dropna()
    if sub.shape[0] < 3:
        return {"x": x, "y": y, "n": int(sub.shape[0]), "pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pr = stats.pearsonr(sub[x], sub[y])
    sr = stats.spearmanr(sub[x], sub[y])
    return {"x": x, "y": y, "n": int(sub.shape[0]), "pearson_r": float(pr.statistic), "pearson_p": float(pr.pvalue), "spearman_r": float(sr.statistic), "spearman_p": float(sr.pvalue)}


def main():
    log.info("Iniciando auditoria tinnitus...")
    bruto_cols = pd.read_csv(BRUTO_CSV, nrows=1).columns.tolist()
    has_auq191 = "AUQ191" in bruto_cols
    has_auq500 = "AUQ500" in bruto_cols
    usecols = ["SEQN", "cycle"] + [c for c in ["AUQ191", "AUQ500"] if c in bruto_cols]
    bruto = pd.read_csv(BRUTO_CSV, usecols=usecols, low_memory=False)
    feat = pd.read_csv(FEATURE_CSV, usecols=["SEQN", "cycle", "pta_high_mean_binaural", "hf_lf_contrast_mean"], low_memory=False)
    df = bruto.merge(feat, on=["SEQN", "cycle"], how="left", validate="one_to_one")
    if has_auq191:
        auq = pd.to_numeric(df["AUQ191"], errors="coerce")
        df["tinnitus_any"] = np.select([auq == 1, auq == 2], [1.0, 0.0], default=np.nan).astype("float32")
    else:
        df["tinnitus_any"] = np.nan

    by_cycle = []
    for cycle, g in df.groupby("cycle", dropna=False):
        raw_counts = g["AUQ191"].value_counts(dropna=False).to_dict() if has_auq191 else {}
        t = g["tinnitus_any"]
        by_cycle.append({
            "cycle": str(cycle),
            "n": int(g.shape[0]),
            "AUQ191_exists_in_merged_file": bool(has_auq191),
            "AUQ500_exists_in_merged_file": bool(has_auq500),
            "AUQ191_raw_counts": {str(k): int(v) for k, v in raw_counts.items()},
            "tinnitus_nonmissing_n": int(t.notna().sum()),
            "tinnitus_yes_n": int((t == 1).sum()),
            "tinnitus_no_n": int((t == 0).sum()),
            "tinnitus_yes_rate_among_nonmissing": float(t.mean(skipna=True)) if t.notna().sum() else None,
        })

    corrs = [
        corr_pair(df, "tinnitus_any", "pta_high_mean_binaural"),
        corr_pair(df, "tinnitus_any", "hf_lf_contrast_mean"),
    ]
    log.info(f"""
FINDING #TINNITUS-AUDIT
Descrição: Disponibilidade e correlação bruta de AUQ191/tinnitus_any auditadas antes de uso como descritor de cluster.
Métrica: n_tinnitus_nonmissing_total = {int(df['tinnitus_any'].notna().sum())}; tinnitus_yes_rate = {float(df['tinnitus_any'].mean(skipna=True)) if df['tinnitus_any'].notna().sum() else None}
N: {df.shape[0]}
Output salvo: {OUTPUT_PATH}
Status: PRELIMINAR — auditoria de variável, sem interpretação clínica
""")
    output = {"script": "16_tinnitus_audit.py", "random_state": RANDOM_STATE, "has_AUQ191": has_auq191, "has_AUQ500": has_auq500, "by_cycle": by_cycle, "correlations": corrs}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Concluído. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
