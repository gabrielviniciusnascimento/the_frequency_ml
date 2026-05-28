#!/usr/bin/env python3
"""
Nome: 17_generate_results_v2_md.py
Tarefa: Gerar RESULTADOS_CLUSTERING_V2.md com HDBSCAN PCA, KMeans, artefatos, residualização e tinnitus audit.
Input: outputs/json/hdbscan_grid_results.json; outputs/json/hdbscan_pca_grid_v2.json; outputs/json/kmeans_grid_v1.json; outputs/json/artifact_test_v1.json; outputs/json/clustering_residualizado_v1.json; outputs/json/tinnitus_audit_v1.json.
Output: RESULTADOS_CLUSTERING_V2.md; outputs/json/17_generate_results_v2_md.json.
Dependências: 12-16.
"""

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, spatial, linalg
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
try:
    import shap
except ImportError:
    shap = None
from sklearn.cluster import KMeans, DBSCAN
try:
    import hdbscan
except ImportError:
    hdbscan = None
from joblib import Parallel, delayed
import multiprocessing as mp
N_JOBS = max(mp.cpu_count() - 1, 1)

RANDOM_STATE = 42
RAW_HDBSCAN_JSON = Path("outputs/json/hdbscan_grid_results.json")
PCA_HDBSCAN_JSON = Path("outputs/json/hdbscan_pca_grid_v2.json")
KMEANS_JSON = Path("outputs/json/kmeans_grid_v1.json")
ARTIFACT_JSON = Path("outputs/json/artifact_test_v1.json")
RESID_JSON = Path("outputs/json/clustering_residualizado_v1.json")
TINNITUS_JSON = Path("outputs/json/tinnitus_audit_v1.json")
OUTPUT_MD = Path("RESULTADOS_CLUSTERING_V2.md")
OUTPUT_PATH = Path("outputs/json/17_generate_results_v2_md.json")
LOG_PATH = Path("outputs/logs/17_generate_results_v2_md.log")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()])
log = logging.getLogger(__name__)

if OUTPUT_PATH.exists():
    log.info(f"Output já existe em {OUTPUT_PATH}. Pulando.")
    raise SystemExit(0)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def md_table(df):
    if df.empty:
        return "_Vazio._\n"
    return df.to_markdown(index=False) + "\n"


def hdb_table(obj):
    rows=[]
    for p in obj['policies']:
        for r in p['grid_results']:
            rows.append({'policy':p['policy'], 'space': r.get('space','raw95'), 'min_cluster_size': r['min_cluster_size'], 'min_samples': r['min_samples'], 'n_clusters': r['n_clusters'], 'n_noise': r['n_noise'], 'noise_fraction': round(r['noise_fraction'],4), 'dbcv': None if r.get('dbcv_sample_estimate') is None else round(r['dbcv_sample_estimate'],4)})
    return pd.DataFrame(rows)


def best_hdb(obj):
    rows=[]
    for p in obj['policies']:
        b=p['best_config']
        rows.append({'policy':p['policy'], 'min_cluster_size':b['min_cluster_size'], 'min_samples':b['min_samples'], 'n_clusters':b['n_clusters'], 'n_noise':b['n_noise'], 'noise_fraction':round(b['noise_fraction'],4), 'dbcv':None if b.get('dbcv_sample_estimate') is None else round(b['dbcv_sample_estimate'],4), 'criterion':p['selection_criterion']})
    return pd.DataFrame(rows)


def kmeans_table(obj):
    rows=[]
    for p in obj['policies']:
        for r in p['grid_results']:
            rows.append({'policy':p['policy'], 'k':r['k'], 'silhouette':round(r['silhouette_sample'],4), 'davies_bouldin':round(r['davies_bouldin'],4), 'inertia':round(r['inertia'],2)})
    return pd.DataFrame(rows)


def kmeans_best(obj):
    rows=[]
    for p in obj['policies']:
        b=p['best_config']
        rows.append({'policy':p['policy'], 'k':b['k'], 'silhouette':round(b['silhouette_sample'],4), 'davies_bouldin':round(b['davies_bouldin'],4), 'criterion':p['selection_criterion']})
    return pd.DataFrame(rows)


def artifact_summary(obj):
    rows=[]
    for p in obj['policies']:
        tests=p['artifact_tests']
        cyc=tests['cycle_chi_square_global_non_noise']; sex=tests['sex_chi_square_global_non_noise']; age=tests['age_kruskal_global_non_noise']
        rows.append({'policy':p['policy'], 'source_used':p['cluster_source_used']['source'], 'reason':p['cluster_source_used']['reason'], 'cycle_chi_p': None if cyc is None else f"{cyc['p']:.3e}", 'cycle_cramers_v': None if cyc is None else round(cyc['cramers_v'],4), 'sex_chi_p':None if sex is None else f"{sex['p']:.3e}", 'sex_cramers_v':None if sex is None else round(sex['cramers_v'],4), 'age_kruskal_p':None if age is None else f"{age['p']:.3e}"})
    return pd.DataFrame(rows)


def age_strat_table(obj):
    rows=[]
    for r in obj['stratified_by_age']:
        hb=r.get('hdbscan_fixed') or {}
        kb=r.get('kmeans_best') or {}
        rows.append({'policy':r['policy'], 'age_group':r['age_group'], 'n':r['n'], 'hdbscan_n_clusters':hb.get('n_clusters'), 'hdbscan_noise_fraction':None if hb.get('noise_fraction') is None else round(hb.get('noise_fraction'),4), 'kmeans_best_k':kb.get('k'), 'kmeans_silhouette':None if kb.get('silhouette') is None else round(kb.get('silhouette'),4)})
    return pd.DataFrame(rows)


def residual_table(obj):
    rows=[]
    for p in obj['policies']:
        h=p['hdbscan_fixed']; k=p['kmeans_best']
        rows.append({'policy':p['policy'], 'residual_matrix':p['residualized_matrix'], 'pca_var_sum':round(p['pca_explained_variance_sum'],4), 'hdbscan_n_clusters':h['n_clusters'], 'hdbscan_noise_fraction':round(h['noise_fraction'],4), 'kmeans_best_k':k['k'], 'kmeans_silhouette':round(k['silhouette_sample'],4), 'kmeans_db':round(k['davies_bouldin'],4)})
    return pd.DataFrame(rows)


def tinnitus_table(obj):
    rows=[]
    for r in obj['by_cycle']:
        rows.append({'cycle':r['cycle'], 'n':r['n'], 'nonmissing':r['tinnitus_nonmissing_n'], 'yes_n':r['tinnitus_yes_n'], 'no_n':r['tinnitus_no_n'], 'yes_rate':None if r['tinnitus_yes_rate_among_nonmissing'] is None else round(r['tinnitus_yes_rate_among_nonmissing'],4)})
    return pd.DataFrame(rows)


def main():
    raw=load(RAW_HDBSCAN_JSON); pca=load(PCA_HDBSCAN_JSON); km=load(KMEANS_JSON); art=load(ARTIFACT_JSON); resid=load(RESID_JSON); tin=load(TINNITUS_JSON)
    lines=[]
    lines.append('# RESULTADOS_CLUSTERING_V2\n\n')
    lines.append('**Sessão:** 3 — HDBSCAN no espaço PCA + validação de artefatos  \n')
    lines.append('**Data:** 2026-05-25  \n')
    lines.append('**Regra:** nenhum rótulo clínico. Geometria, artefatos e resíduos apenas.\n\n')

    lines.append('## 1. HDBSCAN em features brutas vs PCA15\n\n')
    lines.append('Sessão 2 em 95 features brutas teve ~90% ruído. Sessão 3 reexecutou HDBSCAN em 15 componentes PCA.\n\n')
    lines.append('### Melhor HDBSCAN — espaço bruto 95 features\n\n'); lines.append(md_table(best_hdb(raw)))
    lines.append('\n### Grid HDBSCAN — espaço PCA15\n\n'); lines.append(md_table(hdb_table(pca)))
    lines.append('\n### Melhor HDBSCAN — espaço PCA15\n\n'); lines.append(md_table(best_hdb(pca)))
    lines.append('\n**Observação metodológica:** se nenhum run PCA15 atingiu `noise_fraction < 0.40`, o script registrou fallback explícito. Isso impede declarar que PCA resolveu o ruído se a métrica não mostrar.\n\n')

    lines.append('## 2. KMeans baseline no PCA15\n\n')
    lines.append('KMeans força estrutura e não tem ruído; usado como baseline geométrico, não como verdade.\n\n')
    lines.append(md_table(kmeans_table(km)))
    lines.append('\n### Melhor KMeans\n\n'); lines.append(md_table(kmeans_best(km)))

    lines.append('\n## 3. Teste de artefatos: idade/ciclo/sexo\n\n')
    lines.append('Fonte usada: melhor HDBSCAN se `noise_fraction < 0.40`; caso contrário, melhor KMeans baseline.\n\n')
    lines.append(md_table(artifact_summary(art)))
    lines.append('\n### Clustering estratificado por faixa etária\n\n')
    lines.append(md_table(age_strat_table(art)))

    lines.append('\n## 4. Residualização por idade e sexo\n\n')
    lines.append('Cada feature shape foi residualizada por `feature ~ idade + idade² + sexo`. Clustering foi refeito nos resíduos.\n\n')
    lines.append(md_table(residual_table(resid)))
    # Top age/sex R2 features
    for policy, rows in resid['residualization_feature_stats'].items():
        df=pd.DataFrame(rows).sort_values('r2_age_sex_model', ascending=False).head(10)
        lines.append(f'\n### Top features explicadas por idade/sexo — {policy}\n\n')
        lines.append(md_table(df[['feature','r2_age_sex_model','coef_age','coef_age2','coef_sex']].round(6)))

    lines.append('\n## 5. Tinnitus audit\n\n')
    lines.append('Auditoria antes de usar tinnitus como descritor de cluster. `AUQ191`: 1=sim, 2=não; demais/missing tratados como ausentes para `tinnitus_any`.\n\n')
    lines.append(md_table(tinnitus_table(tin)))
    lines.append('\n### Correlações brutas tinnitus × features\n\n')
    lines.append(md_table(pd.DataFrame(tin['correlations']).round(6)))

    lines.append('\n## 6. O que sobrou de estrutura após remover idade/sexo\n\n')
    lines.append('- Esta seção deve ser lida estritamente por métricas: comparar `hdbscan_noise_fraction` e `kmeans_silhouette` antes/depois da residualização.\n')
    lines.append('- Se os clusters residuais persistem com silhouette/DB razoáveis, há estrutura além de idade/sexo. Se colapsam ou viram ruído, a estrutura original era majoritariamente demográfica/protocolo.\n')
    lines.append('- Nenhuma etiologia clínica é inferida aqui.\n\n')

    lines.append('## 7. Bloqueios para sessão 4\n\n')
    lines.append('- Não rotular clusters sem validação pós-hoc.\n')
    lines.append('- Investigar efeitos de ciclo/protocolo se `cycle_cramers_v` for alto ou p extremamente baixo.\n')
    lines.append('- Decidir se a análise principal deve usar HDBSCAN, KMeans, ou abordagem hierárquica/mixture, dado o ruído.\n')
    lines.append('- Validar clusters residuais por bootstrap e holdout por ciclo.\n')
    lines.append('- Só depois incorporar sintomas/tinnitus como descritores interpretativos.\n\n')

    lines.append('## 8. Para outros LLMs expandirem\n\n')
    lines.append('1. Calcular NMI/ARI entre KMeans e HDBSCAN PCA quando HDBSCAN tiver ruído aceitável.\n')
    lines.append('2. Testar Gaussian Mixture/UMAP+HDBSCAN como alternativa, registrando risco de distorção.\n')
    lines.append('3. Rodar validação por ciclo: treinar PCA/scaler em ciclos adultos e aplicar em holdout.\n')
    lines.append('4. Fazer model card separado para residualização, com coeficientes e R² por feature.\n')
    lines.append('5. Formalizar critério de “estrutura útil” antes de qualquer discussão clínica.\n')

    OUTPUT_MD.write_text(''.join(lines), encoding='utf-8')
    output={'script':'17_generate_results_v2_md.py','random_state':RANDOM_STATE,'output_md':str(OUTPUT_MD),'status':'ok'}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding='utf-8')
    log.info(f'Concluído. Output: {OUTPUT_MD} e {OUTPUT_PATH}')

if __name__ == '__main__':
    main()
