#!/usr/bin/env python3
"""
Nome: 27_binaural_pooling_ablation.py
Tarefa: Ablação controlada do pooling binaural. Roda o MESMO pipeline de
        clustering em dois braços que diferem em UMA variável — orelhas
        separadas (14D) vs. média binaural (7D) — e mede se o cluster de
        assimetria unilateral severa sobrevive ao pooling.

Claim testada:
  "A média binaural (passo comum de curadoria em órgãos pares) apaga
   fenótipos estritamente assimétricos. O cluster de assimetria unilateral
   do NHANES (N≈12-13, robusto entre ciclos) é deletado silenciosamente
   quando se faz a média das duas orelhas antes do clustering."

Desenho (mesma população N=7.695, ANY25):
  - Braço A (controle, SEM pooling): 14 limiares (7R+7L) -> row-centering ->
    RobustScaler(25-75) -> PCA 95% -> HDBSCAN(mcs=10, ms=5). É o pipeline
    principal (idêntico a scripts 18/26). Identifica o cluster pequeno
    (assimetria) e fixa os SEQN.
  - Braço B (pooling): 7 médias binaurais (R_f+L_f)/2 -> row-centering (7) ->
    RobustScaler -> PCA 95% -> HDBSCAN(mcs=10, ms=5). A dimensão R-vs-L
    é removida do espaço de features POR CONSTRUÇÃO.

Métrica: "taxa de sobrevivência da assimetria" — dos SEQN assimétricos do
Braço A, qual fração permanece num cluster pequeno e distinto no Braço B
(em vez de absorvida no cluster dominante ou virar ruído).

Output: outputs/json/27_binaural_pooling_ablation.json

NOTA: random_state fixo (42), idêntico ao 26. O resultado é quase garantido
matematicamente (pooling deleta R/L); o valor é a demonstração com N concreto
em dado populacional real, não a surpresa.
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples

import hdbscan

# ── Parâmetros (espelham scripts 18 e 26) ────────────────────────────
RANDOM_STATE = 42
AGE_MIN, AGE_MAX = 20, 69
MIN_COMPLETENESS = 10
ANY25_THRESHOLD_DB = 25.0
PCA_VAR = 0.95
HDBSCAN_MCS = 10
HDBSCAN_MS = 5

FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
R_COLS = [f"thr_R_{f}" for f in FREQS]
L_COLS = [f"thr_L_{f}" for f in FREQS]
FREQ_COLS_14 = R_COLS + L_COLS

FEATURE = Path("data/processed/frequencia_feature_matrix_v1.csv")
OUTPUT = Path("outputs/json/27_binaural_pooling_ablation.json")
LOG = Path("outputs/logs/27_binaural_pooling_ablation.log")

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def load_filtered():
    """Filtros idênticos ao pipeline principal. Retorna (df_alt, thr14)."""
    df = pd.read_csv(FEATURE, low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df = df[(age >= AGE_MIN) & (age <= AGE_MAX)].copy()

    thr = df[FREQ_COLS_14].apply(pd.to_numeric, errors="coerce")
    keep = thr.notna().sum(axis=1) >= MIN_COMPLETENESS
    df, thr = df[keep].copy(), thr[keep].copy()

    any25 = (thr > ANY25_THRESHOLD_DB).any(axis=1)
    df, thr = df[any25].copy(), thr[any25].copy()
    log.info(f"ANY25: {len(df)} indivíduos × 14 limiares")
    return df.reset_index(drop=True), thr.reset_index(drop=True)


def cluster_space(thr_df):
    """row-center -> RobustScaler -> PCA 95% -> HDBSCAN(mcs,ms).
    Retorna (labels, n_componentes, embedding_PCA)."""
    X = thr_df.sub(thr_df.mean(axis=1, skipna=True), axis=0).fillna(0.0).to_numpy(np.float64)
    X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
    pca = PCA(n_components=PCA_VAR, svd_solver="full", random_state=RANDOM_STATE)
    Xp = pca.fit_transform(X)
    c = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MCS, min_samples=HDBSCAN_MS,
                        metric="euclidean", cluster_selection_method="eom",
                        core_dist_n_jobs=-1)
    labels = c.fit_predict(Xp)
    return labels, int(Xp.shape[1]), Xp


def separability(Xp, asym_mask):
    """Separabilidade do grupo assimétrico vs resto, INDEPENDENTE do HDBSCAN.
    - mean_silhouette: silhouette médio dos pontos assimétricos sob rótulo
      binário {assimétrico, resto}. >0 = coeso e separado; ~0 ou <0 = indistinto.
    - centroid_separation_d: distância entre centróides (assim vs resto)
      padronizada pelo desvio médio do grupo 'resto' (tipo Cohen's d multivariado).
    """
    if asym_mask.sum() == 0:
        return {"mean_silhouette": None, "centroid_separation_d": None}
    binary = asym_mask.astype(int)
    sil = silhouette_samples(Xp, binary, metric="euclidean")
    mean_sil = float(np.mean(sil[asym_mask]))

    c_asym = Xp[asym_mask].mean(axis=0)
    c_rest = Xp[~asym_mask].mean(axis=0)
    rest_spread = float(np.sqrt(np.mean(np.var(Xp[~asym_mask], axis=0))))
    dist = float(np.linalg.norm(c_asym - c_rest))
    d = dist / rest_spread if rest_spread > 0 else None
    return {
        "mean_silhouette": round(mean_sil, 4),
        "centroid_separation_d": round(d, 4) if d is not None else None,
    }


def cluster_summary(labels):
    sizes = {int(k): int(v) for k, v in zip(*np.unique(labels[labels != -1], return_counts=True))}
    n_noise = int((labels == -1).sum())
    n_clusters = len(sizes)
    largest = max(sizes.values()) / len(labels) if sizes else 0.0
    return sizes, n_clusters, n_noise, round(float(largest), 4)


def ear_asymmetry(thr14, mask):
    """Média de limiares brutos por orelha no subconjunto, p/ caracterizar a assimetria."""
    sub = thr14[mask]
    pta_r = float(sub[R_COLS].mean().mean())
    pta_l = float(sub[L_COLS].mean().mean())
    return round(pta_r, 1), round(pta_l, 1), round(pta_r - pta_l, 1)


def main():
    df, thr14 = load_filtered()
    seqn = df["SEQN"].astype("int64").to_numpy()

    # ── Braço A: 14D, orelhas separadas ──────────────────────────────
    log.info("Braço A (controle, 14D sem pooling)...")
    labels_a, ncomp_a, Xp_a = cluster_space(thr14)
    sizes_a, ncl_a, noise_a, largest_a = cluster_summary(labels_a)
    log.info(f"  A: {ncl_a} clusters, ruído={noise_a}, dominante={largest_a:.4f}, tamanhos={sizes_a}")

    # Cluster pequeno = candidato a assimetria (menor cluster não-ruído).
    dominant_label_a = max(sizes_a, key=sizes_a.get)
    small_labels = sorted([l for l in sizes_a if l != dominant_label_a], key=lambda l: sizes_a[l])
    if not small_labels:
        log.warning("Braço A não produziu cluster pequeno — assimetria não isolável.")
        asym_label = None
        asym_mask = np.zeros(len(labels_a), dtype=bool)
    else:
        asym_label = small_labels[0]
        asym_mask = labels_a == asym_label
    n_asym = int(asym_mask.sum())
    asym_seqn = seqn[asym_mask].tolist()

    pta_r, pta_l, gap = ear_asymmetry(thr14, asym_mask) if n_asym else (None, None, None)
    log.info(f"  A: cluster de assimetria label={asym_label}, N={n_asym}, "
             f"PTA_R={pta_r} PTA_L={pta_l} (R-L={gap} dB), SEQN={asym_seqn}")

    # ── Braço B: 7D, média binaural ──────────────────────────────────
    log.info("Braço B (pooling, 7D média binaural)...")
    pooled = pd.DataFrame({
        f"bin_{f}": thr14[[f"thr_R_{f}", f"thr_L_{f}"]].mean(axis=1, skipna=True)
        for f in FREQS
    })
    labels_b, ncomp_b, Xp_b = cluster_space(pooled)
    sizes_b, ncl_b, noise_b, largest_b = cluster_summary(labels_b)
    dominant_label_b = max(sizes_b, key=sizes_b.get) if sizes_b else None
    log.info(f"  B: {ncl_b} clusters, ruído={noise_b}, dominante={largest_b:.4f}, tamanhos={sizes_b}")

    # ── Separabilidade independente do HDBSCAN (mesmos indivíduos) ────
    sep_a = separability(Xp_a, asym_mask)
    sep_b = separability(Xp_b, asym_mask)
    log.info(f"  Separabilidade do grupo assimétrico — A(14D): {sep_a} | B(7D pooled): {sep_b}")

    # ── Onde foram parar os SEQN assimétricos no Braço B? ────────────
    fate = {"noise": 0, "dominant_cluster": 0, "small_cluster": 0}
    per_seqn = []
    b_labels_of_asym = []
    if n_asym:
        for s in asym_seqn:
            idx = int(np.where(seqn == s)[0][0])
            lb = int(labels_b[idx])
            b_labels_of_asym.append(lb)
            if lb == -1:
                cat = "noise"
            elif lb == dominant_label_b:
                cat = "dominant_cluster"
            else:
                cat = "small_cluster"
            fate[cat] += 1
            per_seqn.append({"seqn": int(s), "braco_b_label": lb, "destino": cat})

        # "Sobrevivência": permanecem juntos num MESMO cluster pequeno e distinto?
        non_noise = [l for l in b_labels_of_asym if l != -1 and l != dominant_label_b]
        if non_noise:
            vals, counts = np.unique(non_noise, return_counts=True)
            modal_label = int(vals[np.argmax(counts)])
            n_together = int(counts.max())
        else:
            modal_label, n_together = None, 0
        survival_rate = round(n_together / n_asym, 4)
    else:
        modal_label, n_together, survival_rate = None, 0, None

    log.info(f"  Destino dos {n_asym} assimétricos no Braço B: {fate}")
    log.info(f"  Sobrevivência (maior bloco co-agrupado / N): {survival_rate}")

    verdict = (
        "POOLING APAGA A ASSIMETRIA" if (survival_rate is not None and survival_rate < 0.5)
        else "ASSIMETRIA SOBREVIVE AO POOLING" if survival_rate is not None
        else "INCONCLUSIVO (sem cluster de assimetria no braço A)"
    )

    result = {
        "script": "27_binaural_pooling_ablation.py",
        "random_state": RANDOM_STATE,
        "n_samples": int(len(df)),
        "hdbscan": {"min_cluster_size": HDBSCAN_MCS, "min_samples": HDBSCAN_MS},
        "arm_A_14d_separate_ears": {
            "pca_components": ncomp_a,
            "n_clusters": ncl_a,
            "n_noise": noise_a,
            "largest_cluster_fraction": largest_a,
            "cluster_sizes": sizes_a,
            "asymmetry_cluster": {
                "label": asym_label,
                "n": n_asym,
                "pta_right_db": pta_r,
                "pta_left_db": pta_l,
                "right_minus_left_db": gap,
                "seqn": asym_seqn,
            },
            "asymmetry_separability": sep_a,
        },
        "arm_B_7d_binaural_mean": {
            "pca_components": ncomp_b,
            "n_clusters": ncl_b,
            "n_noise": noise_b,
            "largest_cluster_fraction": largest_b,
            "cluster_sizes": sizes_b,
            "asymmetry_separability": sep_b,
        },
        "asymmetry_fate_under_pooling": {
            "n_asymmetric": n_asym,
            "fate_counts": fate,
            "largest_coclustered_block": n_together,
            "modal_label_in_B": modal_label,
            "survival_rate": survival_rate,
            "per_seqn": per_seqn,
        },
        "verdict": verdict,
        "interpretation": (
            f"Braço A (orelhas separadas) isola um cluster de assimetria unilateral "
            f"(N={n_asym}, PTA direito {pta_r} dB vs esquerdo {pta_l} dB, "
            f"diferença {gap} dB). Sob média binaural (Braço B), a dimensão R-vs-L "
            f"é removida do espaço de features; {fate.get('dominant_cluster',0)} dos {n_asym} "
            f"caem no cluster dominante e {fate.get('noise',0)} viram ruído. "
            f"Taxa de sobrevivência={survival_rate}. "
            f"Separabilidade do grupo (independente do HDBSCAN): silhouette "
            f"{sep_a.get('mean_silhouette')} (14D) -> {sep_b.get('mean_silhouette')} (7D pooled); "
            f"separação de centróides d={sep_a.get('centroid_separation_d')} -> "
            f"{sep_b.get('centroid_separation_d')}. {verdict}."
        ),
        "status": "EXECUTED",
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Concluído. Output: {OUTPUT}")
    log.info(f"VEREDITO: {verdict}")
    return result


if __name__ == "__main__":
    main()
