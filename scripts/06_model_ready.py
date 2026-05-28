#!/usr/bin/env python3
"""
Nome: 06_model_ready.py
Tarefa: Remover object, remover colunas com >30% NaN, imputar restante com mediana por ciclo para as duas políticas H11.
Input: data/processed/frequencia_feature_matrix_v1.csv; data/processed/frequencia_feature_matrix_v1_666cap125.csv.
Output: data/processed/frequencia_model_ready_v1.parquet; data/processed/frequencia_model_ready_v1_666cap125.parquet; outputs/json/06_model_ready.json.
Dependências: 03_features_v1.py; 05_h11_sensitivity_666.py.
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

# ── Constantes ───────────────────────────────────────────────────────
RANDOM_STATE = 42
MAX_NAN_FRACTION = 0.30
POLICIES = {
    "nan": {
        "input": Path("data/processed/frequencia_feature_matrix_v1.csv"),
        "output_parquet": Path("data/processed/frequencia_model_ready_v1.parquet"),
        "output_csv": Path("data/processed/frequencia_model_ready_v1.csv"),
    },
    "cap125": {
        "input": Path("data/processed/frequencia_feature_matrix_v1_666cap125.csv"),
        "output_parquet": Path("data/processed/frequencia_model_ready_v1_666cap125.parquet"),
        "output_csv": Path("data/processed/frequencia_model_ready_v1_666cap125.csv"),
    },
}
OUTPUT_PATH = Path("outputs/json/06_model_ready.json")
LOG_PATH = Path("outputs/logs/06_model_ready.log")
FEATURE_COLUMNS_JSON = Path("outputs/json/model_ready_feature_columns_v1.json")

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


def shape_only_columns(cols: list[str]) -> list[str]:
    prefixes = (
        "thr_R_", "thr_L_", "better_thr_", "worse_thr_", "asym_thr_",
        "pta_", "hf_lf_contrast", "slope_", "notch_", "recovery_", "curvature_",
        "asym_mean", "asym_high", "retest1000_", "n_no_response_", "n_could_not_"
    )
    banned_exact = {"SEQN", "cycle_code", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "RIDRETH3", "INDFMPIR", "SDMVSTRA", "SDMVPSU", "WTMEC2YR", "WTSAU2YR", "WTMECPRP"}
    out = []
    for c in cols:
        if c in banned_exact:
            continue
        if c.endswith("_raw"):
            continue
        if c.startswith(prefixes):
            out.append(c)
    return out


def process_policy(policy: str, spec: dict) -> tuple[pd.DataFrame, dict]:
    input_path = spec["input"]
    if not input_path.exists():
        raise FileNotFoundError(f"Input ausente para {policy}: {input_path}")
    log.info(f"Carregando política={policy}: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    log.info(f"Shape bruto {policy}: {df.shape}")

    if "cycle" not in df.columns or "SEQN" not in df.columns:
        raise ValueError("Colunas obrigatórias ausentes: cycle/SEQN")

    cycle_labels = sorted(df["cycle"].astype(str).unique().tolist())
    cycle_map = {label: idx for idx, label in enumerate(cycle_labels)}
    out = df.copy()
    out["cycle_code"] = out["cycle"].astype(str).map(cycle_map).astype("int16")

    object_cols = out.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    # cycle é removido; cycle_code preserva a informação de ciclo numericamente para imputação/metadata.
    out_numeric = out.drop(columns=object_cols, errors="ignore")
    log.info(f"{policy}: removidas colunas object/string/category: {len(object_cols)}")

    nan_fraction = out_numeric.isna().mean(axis=0)
    protected = {"SEQN", "cycle_code"}
    too_nan_cols = [c for c in nan_fraction.index if nan_fraction[c] > MAX_NAN_FRACTION and c not in protected]
    out_numeric = out_numeric.drop(columns=too_nan_cols, errors="ignore")
    log.info(f"{policy}: removidas colunas >{MAX_NAN_FRACTION:.0%} NaN: {len(too_nan_cols)}")

    # Tipagem numérica explícita.
    for c in out_numeric.columns:
        if c in ["SEQN"]:
            out_numeric[c] = pd.to_numeric(out_numeric[c], errors="raise").astype("int64")
        elif c in ["cycle_code"]:
            out_numeric[c] = pd.to_numeric(out_numeric[c], errors="raise").astype("int16")
        else:
            out_numeric[c] = pd.to_numeric(out_numeric[c], errors="coerce").astype("float32")

    feature_cols = [c for c in out_numeric.columns if c not in ["SEQN", "cycle_code"]]
    before_nulls = out_numeric[feature_cols].isna().sum()
    imputed_cols = before_nulls[before_nulls > 0].index.tolist()

    # Imputação por mediana por ciclo; fallback global para ciclo com mediana ausente.
    group_medians = out_numeric.groupby("cycle_code", observed=True)[feature_cols].transform("median")
    global_medians = out_numeric[feature_cols].median(axis=0, skipna=True)
    out_numeric.loc[:, feature_cols] = out_numeric[feature_cols].fillna(group_medians)
    out_numeric.loc[:, feature_cols] = out_numeric[feature_cols].fillna(global_medians)

    remaining_all_nan = out_numeric[feature_cols].columns[out_numeric[feature_cols].isna().any()].tolist()
    if remaining_all_nan:
        log.warning(f"ANOMALIA DETECTADA — colunas ainda com NaN após fallback global em {policy}: {remaining_all_nan}")
        out_numeric = out_numeric.drop(columns=remaining_all_nan)
        feature_cols = [c for c in feature_cols if c not in remaining_all_nan]

    out_numeric = out_numeric.copy()
    qa_dataframe(out_numeric, nome=f"model_ready_{policy}")

    spec["output_parquet"].parent.mkdir(parents=True, exist_ok=True)
    out_numeric.to_parquet(spec["output_parquet"], index=False)
    out_numeric.to_csv(spec["output_csv"], index=False)
    log.info(f"{policy}: salvo {spec['output_parquet']} e {spec['output_csv']}; shape={out_numeric.shape}")

    result = {
        "policy": policy,
        "input": str(input_path),
        "output_parquet": str(spec["output_parquet"]),
        "output_csv": str(spec["output_csv"]),
        "input_shape": list(df.shape),
        "output_shape": list(out_numeric.shape),
        "cycle_map": cycle_map,
        "removed_object_cols": object_cols,
        "removed_gt30pct_nan_cols": too_nan_cols,
        "imputed_cols": imputed_cols,
        "n_features_model_ready": len(feature_cols),
        "n_shape_only_features": len(shape_only_columns(out_numeric.columns.tolist())),
    }
    return out_numeric, result


def main():
    log.info("Iniciando limpeza final model-ready para políticas H11...")
    processed = []
    frames = {}
    for policy, spec in POLICIES.items():
        frame, result = process_policy(policy, spec)
        processed.append(result)
        frames[policy] = frame

    # Interseção conservadora de features shape-only para comparabilidade entre políticas.
    shape_sets = {policy: set(shape_only_columns(frame.columns.tolist())) for policy, frame in frames.items()}
    shape_intersection = sorted(set.intersection(*shape_sets.values()))
    all_feature_sets = {policy: set([c for c in frame.columns if c not in ["SEQN", "cycle_code"]]) for policy, frame in frames.items()}
    all_intersection = sorted(set.intersection(*all_feature_sets.values()))

    feature_meta = {
        "shape_only_intersection": shape_intersection,
        "all_model_features_intersection": all_intersection,
        "policies": {p: {"shape_only": sorted(shape_sets[p]), "all_model": sorted(all_feature_sets[p])} for p in frames},
    }
    FEATURE_COLUMNS_JSON.parent.mkdir(parents=True, exist_ok=True)
    with FEATURE_COLUMNS_JSON.open("w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=2, ensure_ascii=False)

    result = {
        "script": "06_model_ready.py",
        "random_state": RANDOM_STATE,
        "max_nan_fraction": MAX_NAN_FRACTION,
        "policies": processed,
        "feature_columns_json": str(FEATURE_COLUMNS_JSON),
        "shape_only_intersection_n": len(shape_intersection),
        "all_model_features_intersection_n": len(all_intersection),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"Concluído. Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
