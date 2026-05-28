#!/usr/bin/env python3
"""
Nome: 23_session5_tinnitus_clusters.py
Tarefa: Cruzar tinnitus AUQ191 com os clusters da Sessão 4.
Input: outputs/json/session4_assignments_any25.csv, data/processed/frequencia_feature_matrix_v1.csv
Output: outputs/json/23_tinnitus_clusters.json
"""

import logging, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

RANDOM_STATE = 42
ASSIGN = Path("outputs/json/session4_assignments_any25.csv")
FEATURE = Path("data/processed/frequencia_feature_matrix_v1.csv")
OUTPUT = Path("outputs/json/23_tinnitus_clusters.json")
LOG = Path("outputs/logs/23_tinnitus_clusters.log")

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger(__name__)

if OUTPUT.exists():
    log.info(f"Output já existe: {OUTPUT}. Pulando.")
    raise SystemExit(0)


def main():
    log.info("=" * 60)
    log.info("SESSÃO 5 — TAREFA 5: Tinnitus × Clusters")
    log.info("=" * 60)

    assign = pd.read_csv(ASSIGN)
    df = pd.read_csv(FEATURE, low_memory=False)

    # Merge assignments com feature matrix
    df = df.merge(assign[["SEQN", "cluster_id"]], on="SEQN", how="inner", validate="one_to_one")

    # Identificar coluna de tinnitus
    tin_col = None
    for candidate in ["AUQ191", "tinnitus_any", "AUQ500"]:
        if candidate in df.columns:
            tin_col = candidate
            break

    if tin_col is None:
        log.warning("Nenhuma coluna de tinnitus encontrada (AUQ191, tinnitus_any, AUQ500)")
        result = {
            "script": "23_session5_tinnitus_clusters.py",
            "random_state": RANDOM_STATE,
            "error": "Nenhuma coluna de tinnitus encontrada",
            "available_columns_with_AUQ": [c for c in df.columns if "AUQ" in c.upper()],
            "status": "SKIPPED — sem variável de tinnitus",
        }
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return

    log.info(f"Coluna de tinnitus: {tin_col}")

    # Codificar tinnitus_any: NHANES AUQ191 → 1=sim, 2=não
    if tin_col == "AUQ191":
        raw = pd.to_numeric(df[tin_col], errors="coerce")
        df["tinnitus_any"] = np.select([raw == 1, raw == 2], [1.0, 0.0], default=np.nan)
    elif tin_col == "tinnitus_any":
        df["tinnitus_any"] = pd.to_numeric(df[tin_col], errors="coerce")
    else:
        raw = pd.to_numeric(df[tin_col], errors="coerce")
        df["tinnitus_any"] = np.select([raw == 1, raw == 2], [1.0, 0.0], default=np.nan)

    # Labels dos grupos
    group_labels = {-1: "outliers", 0: "cluster_0", 1: "cluster_1"}
    df["group"] = df["cluster_id"].map(group_labels)

    # Taxa por grupo
    group_stats = {}
    for cid, label in group_labels.items():
        g = df[df["cluster_id"] == cid]
        tin = pd.to_numeric(g["tinnitus_any"], errors="coerce")
        n_total = len(g)
        n_nonmissing = int(tin.notna().sum())
        n_yes = int((tin == 1).sum())
        n_no = int((tin == 0).sum())
        rate = round(float(tin.mean(skipna=True)), 4) if n_nonmissing > 0 else None

        group_stats[label] = {
            "cluster_id": cid, "n_total": n_total,
            "n_tinnitus_nonmissing": n_nonmissing,
            "n_tinnitus_yes": n_yes, "n_tinnitus_no": n_no,
            "tinnitus_rate": rate,
        }
        log.info(f"  {label}: n={n_total}, tinnitus_não_nulo={n_nonmissing}, sim={n_yes}, taxa={rate}")

    # Chi-quadrado: tinnitus × grupo (excluindo missing)
    df_valid = df[df["tinnitus_any"].notna() & df["cluster_id"].isin([0, 1, -1])].copy()
    if len(df_valid) >= 10:
        contingency = pd.crosstab(df_valid["group"], df_valid["tinnitus_any"].astype(int))
        if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
            chi2, p, dof, expected = stats.chi2_contingency(contingency, correction=False)
            n = contingency.to_numpy().sum()
            r, k = contingency.shape
            cramers_v = float(np.sqrt((chi2 / n) / max(min(k - 1, r - 1), 1)))
            chi_result = {
                "chi2": round(float(chi2), 4), "p_value": float(p),
                "dof": int(dof), "cramers_v": round(cramers_v, 4),
                "contingency_table": contingency.to_dict(),
                "n_total": int(n),
            }
            log.info(f"  Chi²: chi2={chi2:.4f}, p={p:.6f}, Cramér's V={cramers_v:.4f}")
        else:
            chi_result = {"error": "Tabela de contingência insuficiente"}
    else:
        chi_result = {"error": "Dados insuficientes para chi-quadrado"}

    # Correlação tinnitus × pta_high (global no subset ANY25)
    pta_col = "pta_high_mean_binaural"
    corr_result = None
    if pta_col in df.columns:
        sub = df[["tinnitus_any", pta_col]].dropna()
        if len(sub) >= 10:
            pr = stats.pearsonr(sub["tinnitus_any"], sub[pta_col])
            sr = stats.spearmanr(sub["tinnitus_any"], sub[pta_col])
            corr_result = {
                "x": "tinnitus_any", "y": pta_col, "n": int(len(sub)),
                "pearson_r": round(float(pr.statistic), 6), "pearson_p": float(pr.pvalue),
                "spearman_r": round(float(sr.statistic), 6), "spearman_p": float(sr.pvalue),
            }
            log.info(f"  Correlação tinnitus × {pta_col}: Pearson r={pr.statistic:.4f}, Spearman r={sr.statistic:.4f}")

    result = {
        "script": "23_session5_tinnitus_clusters.py",
        "random_state": RANDOM_STATE,
        "tinnitus_column_used": tin_col,
        "group_stats": group_stats,
        "chi_square_test": chi_result,
        "correlation_tinnitus_pta_high": corr_result,
        "status": "EXECUTED — sem rótulo clínico",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    log.info(f"Concluído. Output: {OUTPUT}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
