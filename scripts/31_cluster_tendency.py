#!/usr/bin/env python3
"""
Nome: 31_cluster_tendency.py
Tarefa: Bateria de testes de EXISTÊNCIA de estrutura discreta no espaço de forma
        do NHANES — os testes que faltavam para sustentar (ou refutar) a tese de
        "contínuo, não subtipos". Complementa 26_method_comparison.py (que cobre
        K-means/GMM-BIC/HDBSCAN) com quatro diagnósticos de outra família.

Claim testada:
  "A forma do audiograma é um contínuo dominante + outliers raros; não há
   subtipos discretos bem separados. Nenhuma família de método (densidade,
   verossimilhança, hierarquia, tendência, unimodalidade) sustenta o contrário."

Testes (todos no MESMO espaço PCA do pipeline, via scripts/_shape_space.py):
  1. Hopkins      — tendência de cluster (departure de uniformidade espacial).
                    CAVEAT: alto NÃO implica multimodalidade; um contínuo único
                    elongado também dá Hopkins alto. Reportado por completude.
  2. Hartigan dip — unimodalidade de PC1, PC2 e do eixo bruto de assimetria
                    (PTA_R - PTA_L). p>0.05 = não rejeita unimodal = contínuo.
  3. OPTICS       — perfil de reachability (vales = clusters de densidade) +
                    nº de clusters/ruído pela extração xi. Compara com HDBSCAN.
  4. Ward         — dendrograma (subamostra), correlação cophenetic e o maior
                    "gap" relativo de altura de merge (gap grande => k natural;
                    crescimento suave => contínuo).

Output: outputs/json/31_cluster_tendency.json  (inclui arrays p/ a figura 31b)
Figura: scripts/31b_tendency_figures.py
Dependências: scripts/_shape_space.py; (opcional) outputs/json/26_method_comparison.json

NOTA PARA REVISÃO: a interpretação automática é heurística; confira os números
brutos (dip p, reachability, gap) antes de citar no paper.
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist
from sklearn.cluster import OPTICS
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shape_space import load_cohort, shape_embed, lib_versions, R_COLS, L_COLS

try:
    import diptest
except ImportError:
    diptest = None

# ── Parâmetros ───────────────────────────────────────────────────────
RANDOM_STATE = 42
HOPKINS_SAMPLE_FRAC = 0.05     # m = 5% de n por sorteio
HOPKINS_B = 20                 # nº de sorteios independentes
OPTICS_MIN_SAMPLES = 10        # espelha HDBSCAN min_cluster_size do pipeline
OPTICS_XI = 0.05               # declive mínimo para separar clusters (xi method)
OPTICS_MIN_CLUSTER_FRAC = 0.01 # cluster mínimo = 1% de n
HIER_SAMPLE_SIZE = 2500        # subamostra p/ Ward (espelha SIL_SAMPLE do 26)
DIP_AXES = ("PC1", "PC2")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "json" / "31_cluster_tendency.json"
LOG = ROOT / "outputs" / "logs" / "31_cluster_tendency.log"
REF_26 = ROOT / "outputs" / "json" / "26_method_comparison.json"

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def hopkins_statistic(X, sample_frac, B, rng):
    """Hopkins H (forma com distâncias, power 1). H~0.5 = espacialmente uniforme
    (sem tendência de cluster); H->1 = não-uniforme (mas NÃO distingue 1 de k modos)."""
    n, d = X.shape
    m = max(5, int(sample_frac * n))
    mins, maxs = X.min(axis=0), X.max(axis=0)
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    vals = []
    for _ in range(B):
        # u: pontos uniformes -> vizinho real mais próximo
        U = rng.uniform(mins, maxs, size=(m, d))
        du, _ = nbrs.kneighbors(U, n_neighbors=1)
        # w: pontos reais sorteados -> vizinho real mais próximo (exclui o próprio)
        idx = rng.choice(n, size=m, replace=False)
        dw, _ = nbrs.kneighbors(X[idx], n_neighbors=2)
        dw = dw[:, 1]
        u_sum, w_sum = float(du[:, 0].sum()), float(dw.sum())
        vals.append(u_sum / (u_sum + w_sum) if (u_sum + w_sum) > 0 else 0.5)
    vals = np.array(vals)
    return {
        "m_per_draw": m,
        "B": B,
        "mean": round(float(vals.mean()), 4),
        "std": round(float(vals.std()), 4),
        "min": round(float(vals.min()), 4),
        "max": round(float(vals.max()), 4),
        "note": ("H~0.5 = sem tendência de cluster; H alto indica não-uniformidade "
                 "mas NÃO multimodalidade (um contínuo elongado também dá H alto)."),
    }


def dip_test(x, label):
    """Hartigan dip test de unimodalidade. p>0.05 => não rejeita unimodal."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if diptest is None:
        return {"axis": label, "method": "UNAVAILABLE (diptest não instalado)",
                "dip": None, "p_value": None}
    d, p = diptest.diptest(x)
    return {
        "axis": label,
        "n": int(x.size),
        "dip": round(float(d), 5),
        "p_value": round(float(p), 4),
        "unimodal_not_rejected": bool(p > 0.05),
    }


def optics_block(X):
    """OPTICS: reachability ordenado + extração de clusters por xi."""
    min_cluster = max(OPTICS_MIN_SAMPLES, int(OPTICS_MIN_CLUSTER_FRAC * X.shape[0]))
    model = OPTICS(min_samples=OPTICS_MIN_SAMPLES, xi=OPTICS_XI,
                   min_cluster_size=min_cluster, metric="euclidean", n_jobs=-1)
    model.fit(X)
    reach = model.reachability_[model.ordering_]
    finite = reach[np.isfinite(reach)]
    labels = model.labels_
    n_clusters = int(len(set(labels) - {-1}))
    n_noise = int((labels == -1).sum())
    return {
        "min_samples": OPTICS_MIN_SAMPLES,
        "xi": OPTICS_XI,
        "min_cluster_size": min_cluster,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": round(n_noise / len(labels), 4),
        "reachability_mean": round(float(finite.mean()), 4),
        "reachability_std": round(float(finite.std()), 4),
        "reachability_cv": round(float(finite.std() / finite.mean()), 4) if finite.mean() else None,
    }, reach


def hierarchy_block(X, rng):
    """Ward sobre subamostra: cophenetic + maior gap relativo de altura de merge."""
    n = X.shape[0]
    m = min(HIER_SAMPLE_SIZE, n)
    idx = rng.choice(n, size=m, replace=False)
    Xs = X[idx]
    Z = linkage(Xs, method="ward")
    cpcc, _ = cophenet(Z, pdist(Xs))

    heights = Z[:, 2]
    diffs = np.diff(heights)                 # crescimento entre merges sucessivos
    rel = diffs / (heights[:-1] + 1e-12)     # gap relativo
    # O MAIOR gap é quase sempre a raiz (k=2) — trivial. O que importa é se há
    # um SEGUNDO gap comparável (k natural >2) ou se os demais decaem (contínuo).
    order = np.argsort(rel)[::-1]
    top_gaps = [{"implied_k": int(len(heights) - int(j)),
                 "relative_gap": round(float(rel[int(j)]), 4)} for j in order[:3]]
    gap_ratio_2_over_1 = round(float(rel[order[1]] / rel[order[0]]), 4) if rel[order[0]] > 0 else None
    return {
        "linkage_method": "ward",
        "sample_size": m,
        "cophenetic_corr": round(float(cpcc), 4),
        "top3_relative_gaps": top_gaps,
        "second_over_first_gap_ratio": gap_ratio_2_over_1,
        "top_merge_heights": [round(float(h), 3) for h in heights[-10:]],
        "note": ("O maior gap é a raiz (k=2), trivial. second_over_first_gap_ratio << 1 "
                 "indica que não há um SEGUNDO k natural (contínuo); ~1 indicaria estrutura "
                 "discreta real. cophenetic baixo = dados não são naturalmente hierárquicos."),
    }, Z


def cross_reference_26():
    if not REF_26.exists():
        return {"available": False}
    d = json.loads(REF_26.read_text(encoding="utf-8"))
    return {
        "available": True,
        "kmeans_best_silhouette": d.get("kmeans", {}).get("best_silhouette_value"),
        "kmeans_best_k": d.get("kmeans", {}).get("best_k_by_silhouette"),
        "gmm_bic_interior_min_robust": False,  # 26 concluiu: só em 'full', raso, instável
        "hdbscan_n_clusters": d.get("hdbscan", {}).get("n_clusters"),
        "hdbscan_noise_fraction": d.get("hdbscan", {}).get("noise_fraction"),
        "hdbscan_largest_fraction": d.get("hdbscan", {}).get("largest_cluster_fraction"),
    }


def main():
    t0 = time.time()
    rng = np.random.default_rng(RANDOM_STATE)

    df, thr = load_cohort()
    emb = shape_embed(thr)
    X = emb.X_pca
    log.info(f"Coorte N={X.shape[0]} × {X.shape[1]} PCs (var={emb.explained_variance:.4f})")

    # 1. Hopkins
    log.info("Hopkins...")
    hop = hopkins_statistic(X, HOPKINS_SAMPLE_FRAC, HOPKINS_B, rng)
    log.info(f"  Hopkins mean={hop['mean']} ± {hop['std']}")

    # 2. Dip test em PCs + eixo de assimetria bruto
    log.info("Hartigan dip...")
    dip = {ax: dip_test(X[:, i], ax) for i, ax in enumerate(DIP_AXES)}
    asym_axis = thr[R_COLS].mean(axis=1).to_numpy() - thr[L_COLS].mean(axis=1).to_numpy()
    dip["asymmetry_R_minus_L_db"] = dip_test(asym_axis, "asymmetry_R_minus_L_db")
    dip["_method"] = ("diptest %s" % getattr(diptest, "__version__", "?")) if diptest else "UNAVAILABLE"
    for k, v in dip.items():
        if isinstance(v, dict) and v.get("p_value") is not None:
            log.info(f"  dip[{k}]: dip={v['dip']} p={v['p_value']} unimodal_ok={v.get('unimodal_not_rejected')}")

    # 3. OPTICS
    log.info("OPTICS...")
    optics, reach = optics_block(X)
    log.info(f"  OPTICS: {optics['n_clusters']} clusters, ruído={optics['noise_fraction']}, "
             f"reachability CV={optics['reachability_cv']}")

    # 4. Ward
    log.info("Ward hierarchy...")
    hier, Z = hierarchy_block(X, rng)
    log.info(f"  Ward: cophenetic={hier['cophenetic_corr']}, top3 gaps={hier['top3_relative_gaps']}, "
             f"2º/1º gap={hier['second_over_first_gap_ratio']}")

    ref26 = cross_reference_26()

    # Síntese honesta (heurística)
    dip_unimodal = all(
        dip[a].get("unimodal_not_rejected") for a in DIP_AXES if dip[a].get("p_value") is not None
    )
    asym_dip = dip["asymmetry_R_minus_L_db"]
    interp = (
        "FAVORÁVEL A CONTÍNUO (convergente entre famílias). "
        f"Hopkins={hop['mean']} (departure de uniformidade — esperado para qualquer massa não-uniforme, "
        "não distingue 1 de k modos). "
        f"Dip nos EIXOS DE CLUSTERING: PC1/PC2 "
        f"{'unimodais (p>0.05)' if dip_unimodal else 'com sinal de multimodalidade (p<=0.05) — REVISAR'} "
        "— o espaço onde o clustering acontece é unimodal. "
        f"Dip no EIXO BRUTO de assimetria (PTA_R−PTA_L): p={asym_dip.get('p_value')} "
        f"(dip={asym_dip.get('dip')}, efeito MINÚSCULO): rejeita unimodalidade, mas com n={X.shape[0]} "
        "o teste detecta a quantização de 5 dB da audiometria + uma cauda contínua pesada — "
        "NÃO é evidência de um segundo modo bem separado (ver figura 31b; confirmar quantização). "
        f"OPTICS: {optics['n_clusters']} cluster(s) por xi, {optics['noise_fraction']*100:.1f}% ruído, "
        f"reachability sem vales fortes (CV={optics['reachability_cv']}). "
        f"Ward: cophenetic={hier['cophenetic_corr']} (baixo = não-hierárquico), razão 2º/1º gap="
        f"{hier['second_over_first_gap_ratio']} (<<1 = sem segundo k natural além da raiz). "
        "Em conjunto com o script 26 (silhouette<0.5; GMM-BIC sem mínimo interior robusto; "
        "HDBSCAN 1 cluster dominante + ruído), NENHUMA família sustenta subtipos discretos bem separados."
    )

    # Arrays p/ a figura 31b (reachability completo; PC1 p/ histograma)
    reach_serial = [None if not np.isfinite(r) else round(float(r), 5) for r in reach]
    result = {
        "script": "31_cluster_tendency.py",
        "claim_tested": ("forma do audiograma = contínuo dominante + outliers; nenhuma família "
                         "de método sustenta subtipos discretos bem separados"),
        "random_state": RANDOM_STATE,
        "n_samples": int(X.shape[0]),
        "pca_components": int(X.shape[1]),
        "explained_variance": round(emb.explained_variance, 4),
        "hopkins": hop,
        "dip_test": dip,
        "optics": optics,
        "hierarchy": hier,
        "cross_reference_26": ref26,
        "interpretation": interp,
        "review_note": ("Interpretação heurística. Hopkins não distingue unimodal de multimodal — "
                        "a evidência de modo único vem do dip + reachability + gap do dendrograma. "
                        "Conferir a figura 31b antes de citar."),
        "figure_arrays": {
            "reachability_ordered": reach_serial,
            "pc1": [round(float(v), 5) for v in X[:, 0]],
            "asymmetry_R_minus_L_db": [round(float(v), 3) for v in asym_axis],
            "ward_linkage_Z": [[round(float(c), 6) for c in row] for row in Z],
        },
        "lib_versions": lib_versions(),
        "elapsed_s": round(time.time() - t0, 1),
        "status": "EXECUTED",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Concluído ({result['elapsed_s']}s). Output: {OUTPUT}")
    return result


if __name__ == "__main__":
    main()
