#!/usr/bin/env python3
"""
Nome: 04_qa_report.py
Tarefa: Gerar relatório QA antes de qualquer modelo ou interpretação de cluster.
Input: data/processed/frequencia_bruto.csv; data/processed/frequencia_feature_matrix_v1.csv.
Output: outputs/logs/qa_report_the_frequency_v1.md; outputs/json/04_qa_report.json.
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
try:
    import shap  # se disponível
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
BRUTO_CSV = Path("data/processed/frequencia_bruto.csv")
FEATURE_CSV = Path("data/processed/frequencia_feature_matrix_v1.csv")
OUTPUT_PATH = Path("outputs/json/04_qa_report.json")
LOG_PATH = Path("outputs/logs/04_qa_report.log")
REPORT_MD = Path("outputs/logs/qa_report_the_frequency_v1.md")
FREQS = np.array([500, 1000, 2000, 3000, 4000, 6000, 8000], dtype=np.int32)

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


def null_summary(df: pd.DataFrame, top_n: int = 40) -> pd.DataFrame:
    n = df.isna().sum()
    pct = n / max(df.shape[0], 1)
    out = pd.DataFrame({"n_null": n.astype(int), "pct_null": pct})
    return out.loc[out["n_null"] > 0].sort_values("n_null", ascending=False).head(top_n)


def threshold_summary(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ear in ["R", "L"]:
        for freq in FREQS:
            col = f"thr_{ear}_{int(freq)}"
            if col not in features.columns:
                continue
            s = pd.to_numeric(features[col], errors="coerce")
            rows.append({
                "ear": ear,
                "frequency_hz": int(freq),
                "n": int(s.notna().sum()),
                "mean": float(s.mean(skipna=True)),
                "std": float(s.std(skipna=True)),
                "p05": float(s.quantile(0.05)),
                "p50": float(s.quantile(0.50)),
                "p95": float(s.quantile(0.95)),
                "n_null": int(s.isna().sum()),
            })
    return pd.DataFrame(rows)


def cycle_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cycle, g in df.groupby("cycle", dropna=False):
        row = {"cycle": str(cycle), "n": int(g.shape[0])}
        if "RIDAGEYR" in g.columns:
            age = pd.to_numeric(g["RIDAGEYR"], errors="coerce")
            row.update({"age_mean": float(age.mean()), "age_min": float(age.min()), "age_max": float(age.max())})
        if "RIAGENDR" in g.columns:
            row["sex_counts"] = g["RIAGENDR"].value_counts(dropna=False).to_dict()
        if "n_no_response_666_thresholds" in g.columns:
            row["n_rows_any_666"] = int((g["n_no_response_666_thresholds"] > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def df_to_md(df: pd.DataFrame, index: bool = False) -> str:
    if df.empty:
        return "_Vazio._\n"
    return df.to_markdown(index=index) + "\n"


def main():
    log.info("Iniciando QA report...")
    if not BRUTO_CSV.exists() or not FEATURE_CSV.exists():
        raise FileNotFoundError("Dependências ausentes: frequencia_bruto.csv e/ou feature_matrix_v1.csv")
    bruto = pd.read_csv(BRUTO_CSV, low_memory=False)
    features = pd.read_csv(FEATURE_CSV, low_memory=False)
    log.info(f"Shape bruto: {bruto.shape}")
    log.info(f"Shape features: {features.shape}")

    nulls_bruto = null_summary(bruto)
    nulls_features = null_summary(features)
    thr = threshold_summary(features)
    cycles = cycle_summary(features)
    object_cols = features.dtypes[features.dtypes == "object"].astype(str).reset_index()
    object_cols.columns = ["column", "dtype"]

    any_666 = int((features.get("n_no_response_666_thresholds", pd.Series(0, index=features.index)) > 0).sum()) if "n_no_response_666_thresholds" in features.columns else 0
    any_888 = int((features.get("n_could_not_obtain_888_thresholds", pd.Series(0, index=features.index)) > 0).sum()) if "n_could_not_obtain_888_thresholds" in features.columns else 0

    report = []
    report.append("# QA_REPORT_THE_FREQUENCY_V1\n")
    report.append(f"- Random state: `{RANDOM_STATE}`\n")
    report.append(f"- Shape `frequencia_bruto`: `{bruto.shape}`\n")
    report.append(f"- Shape `feature_matrix_v1`: `{features.shape}`\n")
    report.append(f"- Linhas com ao menos um `666/no response`: `{any_666}`\n")
    report.append(f"- Linhas com ao menos um `888/could not obtain`: `{any_888}`\n")
    report.append("\n> **Bloqueio metodológico H11:** qualquer interpretação de cluster de alta perda depende de rodar `05_h11_sensitivity_666.py`.\n")
    report.append("\n## Resumo por ciclo\n\n")
    report.append(df_to_md(cycles))
    report.append("\n## Threshold summary\n\n")
    report.append(df_to_md(thr))
    report.append("\n## Top nulos — bruto\n\n")
    report.append(df_to_md(nulls_bruto, index=True))
    report.append("\n## Top nulos — features\n\n")
    report.append(df_to_md(nulls_features, index=True))
    report.append("\n## Colunas object em features\n\n")
    report.append(df_to_md(object_cols))

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text("".join(report), encoding="utf-8")

    result = {
        "script": "04_qa_report.py",
        "random_state": RANDOM_STATE,
        "bruto_shape": list(bruto.shape),
        "features_shape": list(features.shape),
        "rows_any_666": any_666,
        "rows_any_888": any_888,
        "report_md": str(REPORT_MD),
        "status": "QA_PRE_MODEL_GENERATED_NO_FIT_RUN",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)

    log.info(f"Concluído. Report: {REPORT_MD}; Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
