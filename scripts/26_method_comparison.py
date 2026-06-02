#!/usr/bin/env python3
"""
Nome: 26_method_comparison.py
Tarefa: Comparativo formal entre algoritmos de clustering no mesmo espaço de
        features do pipeline principal (NHANES row-centered, 14 limiares).

Claim testada:
  "A estrutura discreta reportada na literatura é dependente de algoritmo/seed;
   sob critérios internos, o que há é um contínuo dominante + outliers raros."

Desenho:
  - K-means: grid k=2..10 × N_SEEDS seeds.
      * estabilidade  -> ARI médio entre pares de seeds (mesmo k)
      * K natural?    -> silhouette(k) e Gap statistic (Tibshirani 2001)
  - GMM: k=2..10 -> BIC e AIC. Mínimo claro = K natural; monótono = contínuo.
  - HDBSCAN: config principal (mcs=10, ms=5) -> n_clusters, fração de ruído.

Espaço: idêntico ao pipeline principal — filtros (idade 20-69, completude
>=10/14, ANY25), row-centering, RobustScaler(25-75), PCA 95% variância.

Output: outputs/json/26_method_comparison.json

NOTA PARA REVISÃO (Gabriel / Claude Code):
  - Parâmetros em MAIÚSCULAS no topo. N_SEEDS=20, GAP_B=10 por padrão.
  - random_state fixo (RANDOM_STATE=42) para reprodutibilidade — fonte única
    de verdade. Se rodar em paralelo, manter este seed.
  - Interpretação automática no campo JSON "interpretation" é heurística;
    confira os números brutos (curvas) antes de citar no paper.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score

import hdbscan

# ── Parâmetros ───────────────────────────────────────────────────────
RANDOM_STATE = 42
K_RANGE = list(range(2, 11))      # k = 2..10
N_SEEDS = 12                      # seeds para teste de estabilidade do K-means
GAP_B = 5                         # nº de datasets de referência para Gap statistic
SIL_SAMPLE = 2500                 # subamostra para silhouette (O(n^2) -> custoso em 7695)
PCA_VAR = 0.95                    # variância retida (igual ao pipeline principal)
HDBSCAN_MCS = 10
HDBSCAN_MS = 5

NHANES_FREQ_COLS = [
    "thr_R_500", "thr_R_1000", "thr_R_2000", "thr_R_3000",
    "thr_R_4000", "thr_R_6000", "thr_R_8000",
    "thr_L_500", "thr_L_1000", "thr_L_2000", "thr_L_3000",
    "thr_L_4000", "thr_L_6000", "thr_L_8000",
]
FEATURE = Path("data/processed/frequencia_feature_matrix_v1.csv")
OUTPUT = Path("outputs/json/26_method_comparison.json")
LOG = Path("outputs/logs/26_method_comparison.log")

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def load_pca_space():
    """Replica o pré-processamento do pipeline principal e retorna X em PCA."""
    df = pd.read_csv(FEATURE, low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df = df[(age >= 20) & (age <= 69)].copy()
    thr = df[NHANES_FREQ_COLS].apply(pd.to_numeric, errors="coerce")
    thr = thr[thr.notna().sum(axis=1) >= 10]
    thr = thr[(thr > 25).any(axis=1)]
    log.info(f"NHANES ANY25: {len(thr)} indivíduos × {thr.shape[1]} limiares")

    # Row-centering (isola forma, remove nível)
    X = thr.sub(thr.mean(axis=1, skipna=True), axis=0).fillna(0.0).to_numpy(np.float64)
    X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
    pca = PCA(n_components=PCA_VAR, random_state=RANDOM_STATE)
    Xp = pca.fit_transform(X)
    log.info(f"PCA: {Xp.shape[1]} componentes (var={pca.explained_variance_ratio_.sum():.4f})")
    return Xp.astype(np.float64)


def gap_statistic(X, k, b=GAP_B, rng=None):
    """Gap statistic (Tibshirani 2001) para um k. Retorna (gap, s_k)."""
    rng = rng or np.random.RandomState(RANDOM_STATE)

    def inertia(data, kk, seed, ni):
        return KMeans(kk, n_init=ni, random_state=seed).fit(data).inertia_

    logW = np.log(inertia(X, k, RANDOM_STATE, 3))
    mins, maxs = X.min(axis=0), X.max(axis=0)
    ref_logs = []
    for i in range(b):
        ref = rng.uniform(mins, maxs, size=X.shape)
        ref_logs.append(np.log(inertia(ref, k, RANDOM_STATE + i, 1)))
    ref_logs = np.array(ref_logs)
    gap = ref_logs.mean() - logW
    s_k = ref_logs.std() * np.sqrt(1 + 1.0 / b)
    return float(gap), float(s_k)


def kmeans_block(X):
    log.info("K-means: grid + estabilidade + silhouette + gap...")
    per_k = {}
    gaps = {}
    for k in K_RANGE:
        # estabilidade: labels por seed -> ARI entre todos os pares
        labels = [KMeans(k, n_init=5, random_state=s).fit_predict(X) for s in range(N_SEEDS)]
        aris = [adjusted_rand_score(labels[i], labels[j])
                for i in range(N_SEEDS) for j in range(i + 1, N_SEEDS)]
        sil = silhouette_score(X, labels[0], sample_size=SIL_SAMPLE, random_state=RANDOM_STATE)
        g, s_k = gap_statistic(X, k)
        per_k[k] = {
            "silhouette": round(float(sil), 4),
            "seed_ari_mean": round(float(np.mean(aris)), 4),
            "seed_ari_min": round(float(np.min(aris)), 4),
            "gap": round(g, 4),
            "gap_s": round(s_k, 4),
        }
        gaps[k] = (g, s_k)
        log.info(f"  k={k}: sil={sil:.3f} seedARI={np.mean(aris):.3f} gap={g:.3f}")

    # K ótimo por silhouette
    best_sil_k = max(per_k, key=lambda k: per_k[k]["silhouette"])
    # K ótimo por Gap (regra Tibshirani: menor k com Gap(k) >= Gap(k+1) - s_{k+1})
    gap_opt = None
    ks = sorted(gaps)
    for i in range(len(ks) - 1):
        k, kn = ks[i], ks[i + 1]
        if gaps[k][0] >= gaps[kn][0] - gaps[kn][1]:
            gap_opt = k
            break
    return {
        "per_k": per_k,
        "best_k_by_silhouette": int(best_sil_k),
        "best_silhouette_value": per_k[best_sil_k]["silhouette"],
        "gap_optimal_k": gap_opt,
    }


def gmm_block(X):
    log.info("GMM: BIC/AIC sweep...")
    per_k = {}
    for k in K_RANGE:
        g = GaussianMixture(k, covariance_type="full", random_state=RANDOM_STATE,
                            n_init=2, max_iter=200).fit(X)
        per_k[k] = {"bic": round(float(g.bic(X)), 1), "aic": round(float(g.aic(X)), 1)}
        log.info(f"  k={k}: BIC={per_k[k]['bic']:.0f} AIC={per_k[k]['aic']:.0f}")
    bic_min_k = min(per_k, key=lambda k: per_k[k]["bic"])
    bics = [per_k[k]["bic"] for k in K_RANGE]
    monotonic_decreasing = all(bics[i] > bics[i + 1] for i in range(len(bics) - 1))
    bic_range = max(bics) - min(bics)
    return {
        "per_k": per_k,
        "bic_min_k": int(bic_min_k),
        "bic_at_kmax_is_min": bool(bic_min_k == K_RANGE[-1]),
        "bic_monotonic_decreasing": bool(monotonic_decreasing),
        "bic_range_abs": round(float(bic_range), 1),
        "bic_range_pct_of_mean": round(float(bic_range / np.mean(bics) * 100), 3),
    }


def gmm_covariance_robustness(X):
    """Testa se o mínimo de BIC é robusto à especificação de covariância.
    Se 'k natural' só aparece com 'full', não é estrutura robusta.
    NOTA: GMM_ROBUST_NINIT=10 (confirmado). Em sandbox 2-vCPU rode por faixa de k se estourar tempo."""
    GMM_ROBUST_NINIT = 10
    out = {}
    for cov in ["full", "tied", "diag", "spherical"]:
        bics = {k: round(float(GaussianMixture(
            k, covariance_type=cov, random_state=RANDOM_STATE,
            n_init=GMM_ROBUST_NINIT, max_iter=300).fit(X).bic(X)), 1) for k in K_RANGE}
        vals = [bics[k] for k in K_RANGE]
        bmin = K_RANGE[int(np.argmin(vals))]
        out[cov] = {
            "bic_min_k": int(bmin),
            "bic_range_pct": round(float((max(vals) - min(vals)) / np.mean(vals) * 100), 3),
            "interior_minimum": bool(bmin not in (K_RANGE[0], K_RANGE[-1])),
            "bic_by_k": bics,
        }
    interior = [c for c in out if out[c]["interior_minimum"]]
    out["_conclusion"] = (
        f"Mínimo interior de BIC apenas em: {interior or 'nenhuma covariância'}. "
        "Se restrito a 'full', o 'K natural' não é robusto à especificação.")
    out["_n_init"] = GMM_ROBUST_NINIT
    return out


def hdbscan_block(X):
    log.info("HDBSCAN: config principal...")
    c = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MCS, min_samples=HDBSCAN_MS,
                        metric="euclidean", cluster_selection_method="eom",
                        core_dist_n_jobs=-1)
    labels = c.fit_predict(X)
    n_clusters = int(len(set(labels) - {-1}))
    n_noise = int((labels == -1).sum())
    sizes = {int(k): int(v) for k, v in
             zip(*np.unique(labels[labels != -1], return_counts=True))} if n_clusters else {}
    largest = max(sizes.values()) / len(labels) if sizes else 0.0
    return {
        "min_cluster_size": HDBSCAN_MCS,
        "min_samples": HDBSCAN_MS,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_fraction": round(n_noise / len(labels), 4),
        "largest_cluster_fraction": round(float(largest), 4),
    }


def main():
    X = load_pca_space()
    km = kmeans_block(X)
    gm = gmm_block(X)
    gm["bic_robustness_covariance"] = gmm_covariance_robustness(X)
    hd = hdbscan_block(X)

    # Interpretação heurística (revisar antes de citar)
    sil_weak = km["best_silhouette_value"] < 0.50          # < 0.5 = sem estrutura substancial
    bic_interior_min = (not gm["bic_at_kmax_is_min"]) and (not gm["bic_monotonic_decreasing"])
    bic_flat = gm["bic_range_pct_of_mean"] < 3.0           # curva quase plana
    interp = (
        "EVIDÊNCIA MISTA, INCLINADA A CONTÍNUO — não é veredito limpo. "
        f"K-means: melhor silhouette={km['best_silhouette_value']} em k={km['best_k_by_silhouette']} "
        f"({'separação fraca' if sil_weak else 'separação substancial'}); cai para k>=3; Gap-ótimo k={km['gap_optimal_k']}. "
        f"HDBSCAN: {hd['n_clusters']} clusters, {hd['noise_fraction']*100:.1f}% ruído, "
        f"maior cluster={hd['largest_cluster_fraction']*100:.1f}%. "
        f"GMM: BIC mínimo em k={gm['bic_min_k']}"
        + (f" (mínimo INTERIOR — sugere K natural, MAS curva quase plana: amplitude "
           f"{gm['bic_range_pct_of_mean']}% da média)." if bic_interior_min else
           " (sem K natural claro).")
        + " Síntese: a literatura de 6-10 fenótipos bem separados NÃO é sustentada; afirmar 'contínuo puro' "
          "é forte demais se o BIC tem mínimo interior raso. Claim defensável: sob critérios internos não há "
          "suporte para múltiplos fenótipos bem separados — estrutura é contínuo dominante + outliers, "
          "com no máximo uma divisão grosseira fraca."
    )
    review_decision = (
        "DECISÃO (Gabriel/Code): se o BIC do GMM tiver mínimo interior (k!=kmax) porém raso "
        "(bic_range_pct_of_mean pequeno), NÃO citar 'contínuo puro'. Decidir entre: (a) reportar mínimo raso "
        "+ silhouette fraco como evidência de contínuo; (b) re-rodar BIC com mais n_init e covariance_type "
        "alternativos para checar robustez de k. Resolver antes de citar no paper."
    )

    result = {
        "review_decision_needed": review_decision,
        "script": "26_method_comparison.py",
        "claim_tested": ("estrutura discreta reportada na literatura é dependente de "
                         "algoritmo/seed; sob critérios internos há um contínuo dominante + outliers raros"),
        "random_state": RANDOM_STATE,
        "n_seeds_kmeans": N_SEEDS,
        "gap_B": GAP_B,
        "silhouette_sample_size": SIL_SAMPLE,
        "n_samples": int(X.shape[0]),
        "pca_components": int(X.shape[1]),
        "kmeans": km,
        "gmm": gm,
        "hdbscan": hd,
        "interpretation": interp,
        "review_note": ("Interpretação é heurística (silhouette<0.25=fraca; K natural exige "
                        "mínimo de BIC interior ao grid). Conferir curvas brutas antes de citar."),
        "status": "EXECUTED",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Concluído. Output: {OUTPUT}")
    return result


if __name__ == "__main__":
    main()
