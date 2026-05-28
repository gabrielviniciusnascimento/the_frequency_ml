#!/usr/bin/env python3
"""
Nome: 11_generate_results_md.py
Tarefa: Gerar RESULTADOS_CLUSTERING_V1.md com grid HDBSCAN, perfis geométricos, ARI, RF surrogate e bloqueios.
Input: outputs/json/hdbscan_grid_results.json; outputs/json/cluster_profiles_v1.json; outputs/json/rf_surrogate_v1.json.
Output: RESULTADOS_CLUSTERING_V1.md; outputs/json/11_generate_results_md.json.
Dependências: 08_hdbscan_grid.py; 09_cluster_profiles.py; 10_rf_surrogate.py.
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
GRID_JSON = Path("outputs/json/hdbscan_grid_results.json")
PROFILES_JSON = Path("outputs/json/cluster_profiles_v1.json")
RF_JSON = Path("outputs/json/rf_surrogate_v1.json")
PCA_JSON = Path("outputs/json/07_pca_umap.json")
MODEL_READY_JSON = Path("outputs/json/06_model_ready.json")
OUTPUT_MD = Path("RESULTADOS_CLUSTERING_V1.md")
OUTPUT_PATH = Path("outputs/json/11_generate_results_md.json")
LOG_PATH = Path("outputs/logs/11_generate_results_md.log")

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


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Vazio._\n"
    return df.to_markdown(index=False) + "\n"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def grid_table(grid: dict) -> pd.DataFrame:
    rows = []
    for p in grid["policies"]:
        for r in p["grid_results"]:
            rows.append({
                "policy": p["policy"],
                "min_cluster_size": r["min_cluster_size"],
                "min_samples": r["min_samples"],
                "n_clusters": r["n_clusters"],
                "n_noise": r["n_noise"],
                "noise_fraction": round(r["noise_fraction"], 4),
                "dbcv_sample_estimate": None if r["dbcv_sample_estimate"] is None else round(r["dbcv_sample_estimate"], 4),
                "selection_score": round(r["selection_score"], 4),
            })
    return pd.DataFrame(rows)


def best_table(grid: dict) -> pd.DataFrame:
    rows = []
    for p in grid["policies"]:
        b = p["best_config"]
        rows.append({
            "policy": p["policy"],
            "min_cluster_size": b["min_cluster_size"],
            "min_samples": b["min_samples"],
            "n_clusters": b["n_clusters"],
            "n_noise": b["n_noise"],
            "noise_fraction": round(b["noise_fraction"], 4),
            "dbcv_sample_estimate": None if b["dbcv_sample_estimate"] is None else round(b["dbcv_sample_estimate"], 4),
            "criterion": p["selection_criterion"],
        })
    return pd.DataFrame(rows)


def profiles_summary(profiles: dict) -> pd.DataFrame:
    rows = []
    for policy in profiles["policies"]:
        for cid, c in policy["clusters"].items():
            sel = c["selected_geometry_medians"]
            rows.append({
                "policy": policy["policy"],
                "cluster_id": cid,
                "noise": c["is_noise"],
                "n": c["n"],
                "age_median": round(c.get("age_summary", {}).get("median", np.nan), 2) if c.get("age_summary") else None,
                "pta_high_mean_binaural_median": round(sel.get("pta_high_mean_binaural", np.nan), 3) if sel else None,
                "hf_lf_contrast_mean_median": round(sel.get("hf_lf_contrast_mean", np.nan), 3) if sel else None,
                "asym_mean_median": round(sel.get("asym_mean", np.nan), 3) if sel else None,
                "tinnitus_rate_available": None if c["tinnitus_any_summary_if_available"]["rate"] is None else round(c["tinnitus_any_summary_if_available"]["rate"], 4),
                "tinnitus_n": c["tinnitus_any_summary_if_available"]["n_non_missing"],
            })
    return pd.DataFrame(rows).sort_values(["policy", "noise", "cluster_id"])


def rf_tables(rf: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gini_rows = []
    perm_rows = []
    split_rows = []
    for p in rf["policies"]:
        for r in p["gini_importance_top20"][:10]:
            gini_rows.append({"policy": p["policy"], "feature": r["feature"], "gini_importance": round(r["importance"], 6)})
        for r in p["permutation_importance_top20"][:10]:
            perm_rows.append({"policy": p["policy"], "feature": r["feature"], "perm_importance_mean": round(r["importance_mean"], 6), "perm_importance_std": round(r["importance_std"], 6)})
        for r in p["top10_split_features"]:
            split_rows.append({"policy": p["policy"], **r})
    return pd.DataFrame(gini_rows), pd.DataFrame(perm_rows), pd.DataFrame(split_rows)


def main():
    log.info("Gerando RESULTADOS_CLUSTERING_V1.md...")
    grid = load_json(GRID_JSON)
    profiles = load_json(PROFILES_JSON)
    rf = load_json(RF_JSON)
    pca = load_json(PCA_JSON)
    model_ready = load_json(MODEL_READY_JSON)

    gt = grid_table(grid)
    bt = best_table(grid)
    ps = profiles_summary(profiles)
    gini, perm, splits = rf_tables(rf)
    ari = profiles["h11_policy_stability"]

    lines = []
    lines.append("# RESULTADOS_CLUSTERING_V1\n\n")
    lines.append("**Projeto:** The Frequency × ML  \n")
    lines.append("**Sessão:** 2 — PCA + UMAP + HDBSCAN + RF surrogate  \n")
    lines.append("**Data:** 2026-05-25  \n")
    lines.append("**Regra metodológica:** nenhum cluster recebe rótulo clínico neste documento. Tudo abaixo é geometria de features audiométricas/operacionais.\n\n")

    lines.append("## 1. Matriz model-ready\n\n")
    lines.append(f"- Política primária `nan`: `data/processed/frequencia_model_ready_v1.parquet`\n")
    lines.append(f"- Política H11 `cap125`: `data/processed/frequencia_model_ready_v1_666cap125.parquet`\n")
    lines.append(f"- Feature set usado para PCA/HDBSCAN/RF: `shape_only_intersection`, n = `{model_ready['shape_only_intersection_n']}`.\n")
    lines.append("- Remoção aplicada: colunas object/string/category e colunas com >30% NaN. Imputação restante: mediana por ciclo, fallback mediana global.\n")
    for p in model_ready["policies"]:
        lines.append(f"  - `{p['policy']}`: shape final `{p['output_shape']}`, features imputadas `{len(p['imputed_cols'])}`, removidas por >30% NaN `{len(p['removed_gt30pct_nan_cols'])}`.\n")
    lines.append("\n")

    lines.append("## 2. PCA + UMAP exploratório\n\n")
    lines.append("UMAP é visualização exploratória, não prova. Dashboard salvo em: `outputs/dashboards/pca_umap_exploratório.html`.\n\n")
    lines.append("| policy | n samples | n shape features | PCA components para 95% | variância explicada |\n|---|---:|---:|---:|---:|\n")
    for p in pca["policies"]:
        lines.append(f"| {p['policy']} | {p['n_samples']} | {p['n_shape_features']} | {p['pca_n_components_95pct']} | {p['pca_explained_variance_sum']:.4f} |\n")
    lines.append("\n")

    lines.append("## 3. Grid HDBSCAN completo\n\n")
    lines.append(f"Critério declarado: {grid['selection_rule']}\n\n")
    lines.append(md_table(gt))
    lines.append("\n### Configurações escolhidas\n\n")
    lines.append(md_table(bt))

    lines.append("\n## 4. Perfis geométricos dos clusters — sem diagnóstico\n\n")
    lines.append("Leitura neutra: `pta_high` = nível alto em altas frequências; `hf_lf_contrast` = diferença altas-baixas; `asym_mean` = assimetria média entre ouvidos. Isso não nomeia etiologia.\n\n")
    lines.append(md_table(ps))

    lines.append("\n## 5. Estabilidade H11\n\n")
    lines.append(f"- ARI incluindo ruído: `{ari['ari_all_including_noise']:.4f}` com n = `{ari['n_all']}`.\n")
    lines.append(f"- ARI apenas interseção não-ruído: `{ari['ari_non_noise_intersection']}` com n = `{ari['n_non_noise_intersection']}`.\n")
    lines.append("- Interpretação permitida: estabilidade/instabilidade geométrica entre políticas de tratamento do `666`.\n")
    lines.append("- Interpretação bloqueada: etiologia clínica de qualquer cluster.\n\n")

    lines.append("## 6. Random Forest surrogate — abertura de caixa preta\n\n")
    lines.append("Target: `cluster_id`, excluindo ruído `-1`. Features: `FEATS_SHAPE_ONLY`. RF com 200 árvores, `n_jobs=-1`, `random_state=42`.\n\n")
    lines.append("| policy | n non-noise | n classes | train accuracy | GroupKFold mean accuracy |\n|---|---:|---:|---:|---:|\n")
    for p in rf["policies"]:
        lines.append(f"| {p['policy']} | {p['n_samples_non_noise']} | {p['n_classes']} | {p['train_accuracy']:.4f} | {p['groupkfold_accuracy_mean']} |\n")
    lines.append("\n### Top 10 — Gini importance\n\n")
    lines.append(md_table(gini))
    lines.append("\n### Top 10 — Permutation importance\n\n")
    lines.append(md_table(perm))
    lines.append("\n### Top 10 split features mais frequentes\n\n")
    lines.append(md_table(splits))

    lines.append("\n## 7. Bloqueios para próxima sessão\n\n")
    lines.append("- Não rotular clusters como cisplatina, presbiacusia, ruído, etc. sem validação pós-hoc.\n")
    lines.append("- Investigar se clusters são artefatos de ciclo/idade/sexo antes de qualquer narrativa clínica.\n")
    lines.append("- Rodar validação por ciclo/holdout e estabilidade bootstrap.\n")
    lines.append("- Auditar features AUQ/tinnitus antes de usá-las como descrição forte; há missingness por elegibilidade/ciclo.\n")
    lines.append("- Decidir formalmente como tratar `666` em modelagem principal após comparar impacto nos clusters.\n\n")

    lines.append("## 8. Para outros LLMs expandirem\n\n")
    lines.append("1. Criar validação por ciclo: treinar em ciclos adultos e projetar em ciclos holdout.\n")
    lines.append("2. Calcular estabilidade por bootstrap/subamostragem, com ARI/NMI.\n")
    lines.append("3. Rodar análise residualizando idade/sexo para separar geometria auditiva de demografia.\n")
    lines.append("4. Formalizar distância audiométrica alternativa: curva log-frequência, cosine shape, Euclidean robust-scaled.\n")
    lines.append("5. Só depois propor rótulos clínicos pós-hoc com evidência externa, nunca por abdução.\n")

    OUTPUT_MD.write_text("".join(lines), encoding="utf-8")
    result = {"script": "11_generate_results_md.py", "random_state": RANDOM_STATE, "output_md": str(OUTPUT_MD), "status": "ok"}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info(f"Concluído. Output: {OUTPUT_MD} e {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
