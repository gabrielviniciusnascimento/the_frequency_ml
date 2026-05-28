#!/usr/bin/env python3
"""
Nome: 14b_artifact_per_cluster.py
Tarefa: Complementar teste de artefato com chi-quadrado por cluster para distribuição uniforme de ciclo e sexo.
Input: outputs/json/artifact_test_v1.json; assignments usados em cada política.
Output: outputs/json/artifact_test_per_cluster_v1.json.
Dependências: 14_artifact_test.py.
"""
import logging, json
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
N_JOBS=max(mp.cpu_count()-1,1)
RANDOM_STATE=42
INPUT=Path('outputs/json/artifact_test_v1.json')
OUTPUT_PATH=Path('outputs/json/artifact_test_per_cluster_v1.json')
LOG_PATH=Path('outputs/logs/14b_artifact_per_cluster.log')
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()])
log=logging.getLogger(__name__)
if OUTPUT_PATH.exists():
    log.info(f'Output já existe em {OUTPUT_PATH}. Pulando.')
    raise SystemExit(0)

def chisq_uniform_counts(counts):
    counts=np.asarray(counts, dtype=float)
    counts=counts[counts>0]
    if counts.size < 2 or counts.sum() == 0:
        return None
    expected=np.full_like(counts, counts.mean())
    res=stats.chisquare(counts, f_exp=expected)
    return {'chi2': float(res.statistic), 'p': float(res.pvalue), 'dof': int(counts.size-1), 'max_share': float(counts.max()/counts.sum())}

def main():
    art=json.loads(INPUT.read_text(encoding='utf-8'))
    results=[]
    for p in art['policies']:
        policy=p['policy']; src=p['cluster_source_used']
        assign=pd.read_csv(src['assignments'])
        if 'cycle' not in assign.columns or 'RIAGENDR' not in assign.columns:
            emb=pd.read_csv(f'outputs/json/pca15_embeddings_{policy}.csv', usecols=['SEQN','cycle','RIAGENDR'])
            assign=assign.merge(emb, on='SEQN', how='left', validate='one_to_one')
        rows=[]
        for cid,g in assign.groupby('cluster_id'):
            if src['source'].startswith('hdbscan') and int(cid)==-1:
                continue
            cyc_counts=g['cycle'].astype(str).value_counts().sort_index()
            sex_counts=pd.to_numeric(g['RIAGENDR'], errors='coerce').dropna().value_counts().sort_index()
            cyc_test=chisq_uniform_counts(cyc_counts.to_numpy())
            sex_test=chisq_uniform_counts(sex_counts.to_numpy())
            rows.append({
                'policy': policy,
                'source': src['source'],
                'cluster_id': int(cid),
                'n': int(g.shape[0]),
                'cycle_uniform_chisq': cyc_test,
                'sex_uniform_chisq': sex_test,
                'cycle_counts': {str(k): int(v) for k,v in cyc_counts.items()},
                'sex_counts_RIAGENDR': {str(k): int(v) for k,v in sex_counts.items()},
            })
        results.append({'policy': policy, 'source': src, 'per_cluster_tests': rows})
        flagged=[r for r in rows if r['cycle_uniform_chisq'] and r['cycle_uniform_chisq']['max_share']>0.80]
        log.info(f"""
FINDING #ARTIFACT-PER-CLUSTER-{policy}
Descrição: Chi-quadrado por cluster contra distribuição uniforme de ciclos/sexo computado para detectar concentração de ciclo/sexo.
Métrica: clusters_com_max_cycle_share_gt_0p80 = {len(flagged)}
N: {assign.shape[0]}
Output salvo: {OUTPUT_PATH}
Status: PRELIMINAR — teste de artefato por cluster, sem rótulo clínico
""")
    out={'script':'14b_artifact_per_cluster.py','random_state':RANDOM_STATE,'note':'Teste por cluster contra distribuição uniforme de categorias observadas no cluster; max_share alto sinaliza concentração.', 'policies':results}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    log.info(f'Concluído. Output: {OUTPUT_PATH}')
if __name__=='__main__':
    main()
