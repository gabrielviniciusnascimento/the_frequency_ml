#!/usr/bin/env python3
"""
Nome: 28_ipsative_check.py
Tarefa: Auditar se o Cluster 1 (assimetria unilateral, N=13) é uma anticorrelação
        inter-canal REAL ou um artefato do row-centering ipsativo (soma-zero).

Duas perguntas, lidas do disco (SEQN vêm de 27_binaural_pooling_ablation.json):

(1) CHECK IPSATIVO — para os 13 casos, os limiares BRUTOS da orelha NORMAL
    (a que parece extrema após centering) estão dentro da faixa populacional?
    Reporta, por frequência e em PTA: percentil e z-score vs a população ANY25.

(2) DECOMPOSIÇÃO SOMA/DIFERENÇA — para cada par (R_i, L_i):
        s_i = (R_i + L_i)/2   (nível binaural; = espaço de pooling do braço 27B)
        d_i = R_i - L_i       (contraste interaural)
    Roda HDBSCAN(mcs=10, ms=5) em sum-7D, diff-7D e full-14D, com e sem
    row-centering, e mede em quais espaços os 13 SEQN são recuperados como
    cluster. Se a discreteza vive no diff e some no sum -> "nível contínuo,
    contraste discreto" (claim real). Se nada recupera sem centering -> o
    cluster depende do centering ipsativo.

Output: outputs/json/28_ipsative_check.json
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import hdbscan

RANDOM_STATE = 42
AGE_MIN, AGE_MAX = 20, 69
MIN_COMPLETENESS = 10
ANY25 = 25.0
PCA_VAR = 0.95
MCS, MS = 10, 5
FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
R_COLS = [f"thr_R_{f}" for f in FREQS]
L_COLS = [f"thr_L_{f}" for f in FREQS]
COLS14 = R_COLS + L_COLS

FEATURE = Path("data/processed/frequencia_feature_matrix_v1.csv")
ABLATION = Path("outputs/json/27_binaural_pooling_ablation.json")
OUTPUT = Path("outputs/json/28_ipsative_check.json")
LOG = Path("outputs/logs/28_ipsative_check.log")

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def load_filtered():
    df = pd.read_csv(FEATURE, low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df = df[(age >= AGE_MIN) & (age <= AGE_MAX)].copy()
    thr = df[COLS14].apply(pd.to_numeric, errors="coerce")
    keep = thr.notna().sum(axis=1) >= MIN_COMPLETENESS
    df, thr = df[keep].copy(), thr[keep].copy()
    m = (thr > ANY25).any(axis=1)
    return df[m].reset_index(drop=True), thr[m].reset_index(drop=True)


def hdbscan_labels(M, row_center):
    """M: (n, p) raw matrix. Aplica (opcional row-center) -> RobustScaler -> PCA95 -> HDBSCAN."""
    X = M.astype(np.float64)
    if row_center:
        X = X - np.nanmean(X, axis=1, keepdims=True)
    X = np.where(np.isnan(X), 0.0, X)
    X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
    pca = PCA(n_components=PCA_VAR, svd_solver="full", random_state=RANDOM_STATE)
    Xp = pca.fit_transform(X)
    c = hdbscan.HDBSCAN(min_cluster_size=MCS, min_samples=MS, metric="euclidean",
                        cluster_selection_method="eom", core_dist_n_jobs=-1)
    return c.fit_predict(Xp), int(Xp.shape[1])


def recovery(labels, asym_mask):
    """Os 13 assimétricos formam um cluster pequeno distinto? Retorna métricas."""
    dom = None
    sizes = {int(k): int(v) for k, v in zip(*np.unique(labels[labels != -1], return_counts=True))}
    if sizes:
        dom = max(sizes, key=sizes.get)
    lab = labels[asym_mask]
    n = int(asym_mask.sum())
    n_noise = int((lab == -1).sum())
    n_dom = int((lab == dom).sum()) if dom is not None else 0
    # maior bloco co-agrupado num cluster pequeno (não dominante, não ruído)
    small = [l for l in lab if l != -1 and l != dom]
    if small:
        vals, counts = np.unique(small, return_counts=True)
        n_together = int(counts.max())
        # pureza: desse cluster pequeno, quantos % são dos 13
        modal = int(vals[np.argmax(counts)])
        purity = round(n_together / int((labels == modal).sum()), 3)
    else:
        n_together, modal, purity = 0, None, None
    return {
        "n_clusters_total": len(sizes),
        "largest_fraction": round(max(sizes.values()) / len(labels), 4) if sizes else 0.0,
        "asym_in_noise": n_noise,
        "asym_in_dominant": n_dom,
        "asym_largest_coclustered_block": n_together,
        "asym_recovery_rate": round(n_together / n, 4),
        "recovered_cluster_purity": purity,
    }


def main():
    df, thr14 = load_filtered()
    seqn = df["SEQN"].astype("int64").to_numpy()
    abl = json.loads(ABLATION.read_text(encoding="utf-8"))
    asym_seqn = abl["arm_A_14d_separate_ears"]["asymmetry_cluster"]["seqn"]
    asym_mask = np.isin(seqn, asym_seqn)
    log.info(f"População ANY25: N={len(df)}; cluster assimetria: N={asym_mask.sum()} ({asym_seqn})")

    R = thr14[R_COLS].to_numpy(np.float64)
    L = thr14[L_COLS].to_numpy(np.float64)

    # ── (1) CHECK IPSATIVO ───────────────────────────────────────────
    def per_freq_stats(M, name):
        pop_mean = np.nanmean(M, axis=0)
        pop_std = np.nanstd(M, axis=0)
        sub = M[asym_mask]
        out = {}
        for j, f in enumerate(FREQS):
            col = M[:, j]
            col = col[~np.isnan(col)]
            sub_vals = sub[:, j]
            sub_mean = float(np.nanmean(sub_vals))
            pct = float((col < sub_mean).mean() * 100)
            z = float((sub_mean - pop_mean[j]) / pop_std[j]) if pop_std[j] > 0 else None
            out[str(f)] = {
                "pop_mean": round(float(pop_mean[j]), 1),
                "pop_std": round(float(pop_std[j]), 1),
                "asym_mean": round(sub_mean, 1),
                "asym_mean_percentile_in_pop": round(pct, 1),
                "asym_mean_z": round(z, 2) if z is not None else None,
            }
        # PTA agregado
        pta_pop = np.nanmean(M, axis=1)
        pta_sub = np.nanmean(sub, axis=1)
        pta_sub_mean = float(np.nanmean(pta_sub))
        pta_pct = float((pta_pop < pta_sub_mean).mean() * 100)
        pta_z = float((pta_sub_mean - np.nanmean(pta_pop)) / np.nanstd(pta_pop))
        log.info(f"  {name}: PTA assim={pta_sub_mean:.1f} dB (pop {np.nanmean(pta_pop):.1f}±"
                 f"{np.nanstd(pta_pop):.1f}); percentil={pta_pct:.1f}; z={pta_z:.2f}")
        out["PTA"] = {
            "pop_mean": round(float(np.nanmean(pta_pop)), 1),
            "pop_std": round(float(np.nanstd(pta_pop)), 1),
            "asym_mean": round(pta_sub_mean, 1),
            "asym_mean_percentile_in_pop": round(pta_pct, 1),
            "asym_mean_z": round(pta_z, 2),
        }
        return out

    log.info("(1) Check ipsativo — orelha esquerda (L, a 'normal') e direita (R, a impactada):")
    L_stats = per_freq_stats(L, "L (esquerda)")
    R_stats = per_freq_stats(R, "R (direita)")

    # Contraste interaural |R-L| em PTA
    diff_pta_pop = np.nanmean(R, axis=1) - np.nanmean(L, axis=1)
    diff_pta_sub = diff_pta_pop[asym_mask]
    diff_mean = float(np.mean(diff_pta_sub))
    diff_pct = float((np.abs(diff_pta_pop) < abs(diff_mean)).mean() * 100)
    log.info(f"  Contraste R-L (PTA): assim={diff_mean:.1f} dB; "
             f"|R-L| > esse valor em {100-diff_pct:.1f}% da pop; percentil={diff_pct:.1f}")

    # Row mean dos 13 (mostra que a média é puxada por R)
    rowmean_sub = float(np.nanmean(np.nanmean(thr14.to_numpy(np.float64)[asym_mask], axis=1)))

    # ── (2) DECOMPOSIÇÃO SOMA/DIFERENÇA ──────────────────────────────
    log.info("(2) Decomposição soma/diferença — HDBSCAN por subespaço:")
    S = (R + L) / 2.0          # nível binaural (= pooling)
    D = R - L                  # contraste interaural
    conditions = {}
    for space_name, M in [("sum_7d", S), ("diff_7d", D), ("full_14d", thr14.to_numpy(np.float64))]:
        for rc in [False, True]:
            labels, ncomp = hdbscan_labels(M, row_center=rc)
            rec = recovery(labels, asym_mask)
            key = f"{space_name}{'_rowcentered' if rc else '_raw'}"
            conditions[key] = {"pca_components": ncomp, **rec}
            log.info(f"  {key}: recovery={rec['asym_recovery_rate']} "
                     f"(noise={rec['asym_in_noise']}, dom={rec['asym_in_dominant']}, "
                     f"block={rec['asym_largest_coclustered_block']}), "
                     f"purity={rec['recovered_cluster_purity']}, dominante={rec['largest_fraction']}")

    # ── Veredito automatizado ────────────────────────────────────────
    L_normal = L_stats["PTA"]["asym_mean_percentile_in_pop"] < 50  # L está na metade boa
    diff_extreme = diff_pct > 95
    diff_recovers = max(conditions["diff_7d_raw"]["asym_recovery_rate"],
                        conditions["diff_7d_rowcentered"]["asym_recovery_rate"]) >= 0.5
    sum_kills = max(conditions["sum_7d_raw"]["asym_recovery_rate"],
                    conditions["sum_7d_rowcentered"]["asym_recovery_rate"]) < 0.5

    if diff_recovers and sum_kills and diff_extreme:
        verdict = ("CONTRASTE REAL: a discreteza vive no espaço-diferença (contraste "
                   "interaural) e some no espaço-soma (nível). O cluster é uma assimetria "
                   "interaural genuína (|R-L| no topo da população), não artefato de centering. "
                   "Tese sobrevive, reescopada: nível contínuo, contraste discreto.")
    elif L_normal and not diff_recovers:
        verdict = ("POSSÍVEL ARTEFATO: orelha normal é típica e o contraste não recupera "
                   "cluster sem o centering ipsativo — a discreteza pode depender do "
                   "centering + p≈N. Rebaixar para nota metodológica honesta.")
    else:
        verdict = "MISTO — ver números brutos antes de citar."
    log.info(f"VEREDITO: {verdict}")

    result = {
        "script": "28_ipsative_check.py",
        "n_population": int(len(df)),
        "asym_seqn": asym_seqn,
        "ipsative_check": {
            "left_ear_raw_vs_population": L_stats,
            "right_ear_raw_vs_population": R_stats,
            "interaural_contrast_pta": {
                "asym_mean_R_minus_L_db": round(diff_mean, 1),
                "asym_contrast_percentile_in_pop": round(diff_pct, 1),
                "pct_of_pop_with_larger_abs_contrast": round(100 - diff_pct, 1),
            },
            "asym_row_mean_db": round(rowmean_sub, 1),
        },
        "sum_difference_decomposition": conditions,
        "verdict": verdict,
        "status": "EXECUTED",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Output: {OUTPUT}")
    return result


if __name__ == "__main__":
    main()
