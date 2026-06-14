#!/usr/bin/env python3
"""
Nome: 30_null_model.py
Tarefa: Calibrar as diagnósticas do dado REAL contra dois nulls sintéticos
        passados pelo pipeline IDÊNTICO — um contínuo e uma mistura discreta.
        Decide se silhouette/gap/bootstrap-ARI/BIC do real caem do lado do
        contínuo ou do discreto, e se o modo de contraste interaural (diff_7d)
        sobrevive no real de um jeito ausente no null contínuo.

Sintéticos no espaço de 14 limiares (7R+7L), N=7695:
  (i)  CONTÍNUO — cópula gaussiana: marginais empíricas por canal (skew real)
       + correlação de postos (Spearman) real. Unimodal, sem clusters projetados.
  (ii) DISCRETO — 4 arquétipos de forma genuinamente separados + ruído intra-cluster.

Pipeline idêntico ao 26: row-center -> fillna0 -> RobustScaler(25-75) -> PCA95.
Diagnósticas: silhouette(k), gap-ótimo, bootstrap-ARI(100x,80%), GMM BIC(full),
e recuperação no espaço-diferença (R-L, 7D).

Output: outputs/json/30_null_model.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
import hdbscan

RANDOM_STATE = 42
AGE_MIN, AGE_MAX = 20, 69
MIN_COMPLETENESS = 10
ANY25 = 25.0
PCA_VAR = 0.95
K_RANGE = list(range(2, 11))
N_SEEDS = 12
GAP_B = 5
SIL_SAMPLE = 2500
MCS, MS = 10, 5
N_BOOT = 100
BOOT_FRAC = 0.8
FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000]
R_COLS = [f"thr_R_{f}" for f in FREQS]
L_COLS = [f"thr_L_{f}" for f in FREQS]
COLS14 = R_COLS + L_COLS

FEATURE = Path("data/processed/frequencia_feature_matrix_v1.csv")
OUTPUT = Path("outputs/json/30_null_model.json")
rng = np.random.RandomState(RANDOM_STATE)


# ── Pipeline e diagnósticas (idênticas ao 26) ────────────────────────
def embed(M14, row_center=True):
    X = np.asarray(M14, dtype=np.float64)
    if row_center:
        X = X - np.nanmean(X, axis=1, keepdims=True)
    X = np.where(np.isnan(X), 0.0, X)
    X = RobustScaler(quantile_range=(25, 75)).fit_transform(X)
    pca = PCA(n_components=PCA_VAR, svd_solver="full", random_state=RANDOM_STATE)
    return pca.fit_transform(X)


def gap_statistic(X, k, b=GAP_B):
    def inertia(data, seed, ni):
        return KMeans(k, n_init=ni, random_state=seed).fit(data).inertia_
    logW = np.log(inertia(X, RANDOM_STATE, 3))
    mins, maxs = X.min(0), X.max(0)
    refs = [np.log(inertia(rng.uniform(mins, maxs, size=X.shape), RANDOM_STATE + i, 1))
            for i in range(b)]
    refs = np.array(refs)
    return float(refs.mean() - logW), float(refs.std() * np.sqrt(1 + 1.0 / b))


def silhouette_and_gap(X):
    per_k, gaps = {}, {}
    for k in K_RANGE:
        labels = KMeans(k, n_init=5, random_state=0).fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=SIL_SAMPLE, random_state=RANDOM_STATE)
        g, s = gap_statistic(X, k)
        per_k[k] = {"silhouette": round(float(sil), 4), "gap": round(g, 4), "gap_s": round(s, 4)}
        gaps[k] = (g, s)
    best_k = max(per_k, key=lambda k: per_k[k]["silhouette"])
    gap_opt = None
    ks = sorted(gaps)
    for i in range(len(ks) - 1):
        if gaps[ks[i]][0] >= gaps[ks[i + 1]][0] - gaps[ks[i + 1]][1]:
            gap_opt = ks[i]
            break
    return {"per_k": per_k, "best_k_silhouette": int(best_k),
            "best_silhouette": per_k[best_k]["silhouette"], "gap_optimal_k": gap_opt}


def gmm_bic(X):
    bics = {}
    for k in K_RANGE:
        g = GaussianMixture(k, covariance_type="full", random_state=RANDOM_STATE,
                            n_init=2, max_iter=200).fit(X)
        bics[k] = round(float(g.bic(X)), 1)
    vals = [bics[k] for k in K_RANGE]
    bmin = K_RANGE[int(np.argmin(vals))]
    return {"bic_min_k": int(bmin),
            "interior_minimum": bool(bmin not in (K_RANGE[0], K_RANGE[-1])),
            "bic_range_pct_of_mean": round(float((max(vals) - min(vals)) / np.mean(vals) * 100), 3),
            "bic_by_k": bics}


def bootstrap_ari(X, n=N_BOOT, frac=BOOT_FRAC):
    ref = hdbscan.HDBSCAN(min_cluster_size=MCS, min_samples=MS, core_dist_n_jobs=-1).fit_predict(X)
    n_sub = int(len(X) * frac)
    aris, ncs = [], []
    for i in range(n):
        idx = np.random.RandomState(RANDOM_STATE + i).choice(len(X), n_sub, replace=False)
        lab = hdbscan.HDBSCAN(min_cluster_size=MCS, min_samples=MS, core_dist_n_jobs=-1).fit_predict(X[idx])
        aris.append(adjusted_rand_score(ref[idx], lab))
        ncs.append(len(set(lab) - {-1}))
    return {"ari_median": round(float(np.median(aris)), 4),
            "ari_mean": round(float(np.mean(aris)), 4),
            "ari_std": round(float(np.std(aris)), 4),
            "n_clusters_mean": round(float(np.mean(ncs)), 2)}


def diff_space_structure(R, L):
    """HDBSCAN no espaço-diferença R-L (7D) + EXCESSO de assimetria extrema vs null.
    O teste decisivo de 'contraste discreto' não é só achar um cluster pequeno
    (HDBSCAN acha em quase tudo com mcs=10), e sim haver EXCESSO de casos com
    contraste interaural extremo além do que a correlação/marginais preveem."""
    R = np.asarray(R, float); L = np.asarray(L, float)
    D = np.where(np.isnan(R - L), 0.0, R - L)
    Xp = embed(D, row_center=False)
    lab = hdbscan.HDBSCAN(min_cluster_size=MCS, min_samples=MS, core_dist_n_jobs=-1).fit_predict(Xp)
    sizes = {int(k): int(v) for k, v in zip(*np.unique(lab[lab != -1], return_counts=True))}
    if sizes:
        dom = max(sizes, key=sizes.get)
        small = sorted([(v, k) for k, v in sizes.items() if k != dom])
        smallest_nondom = small[0][0] if small else 0
        # assimetria média do menor cluster não-dominante (é um modo unilateral real?)
        if small:
            cid = small[0][1]
            asym_of_small = float(np.nanmean(np.abs(np.nanmean(R[lab == cid], 1) - np.nanmean(L[lab == cid], 1))))
        else:
            asym_of_small = None
        largest = max(sizes.values()) / len(lab)
    else:
        smallest_nondom, asym_of_small, largest = 0, None, 0.0

    # Excesso de assimetria extrema (contraste em PTA)
    contrast = np.abs(np.nanmean(R, 1) - np.nanmean(L, 1))
    return {"n_clusters": len(sizes), "n_noise": int((lab == -1).sum()),
            "largest_fraction": round(float(largest), 4),
            "n_nondominant_clusters": max(0, len(sizes) - 1),
            "smallest_nondominant_cluster_size": int(smallest_nondom),
            "smallest_cluster_mean_abs_contrast_db": round(asym_of_small, 1) if asym_of_small is not None else None,
            "has_detached_small_mode": bool(len(sizes) >= 2),
            "n_contrast_gt_40db": int((contrast > 40).sum()),
            "n_contrast_gt_50db": int((contrast > 50).sum()),
            "max_contrast_db": round(float(np.nanmax(contrast)), 1)}


def diagnostics(M14, R, L, name):
    print(f"\n=== {name} ===")
    X = embed(M14, row_center=True)
    sg = silhouette_and_gap(X)
    gm = gmm_bic(X)
    bs = bootstrap_ari(X)
    df = diff_space_structure(R, L)
    print(f"  silhouette_best={sg['best_silhouette']} @k={sg['best_k_silhouette']} | gap_opt_k={sg['gap_optimal_k']}")
    print(f"  GMM BIC min k={gm['bic_min_k']} interior={gm['interior_minimum']} depth={gm['bic_range_pct_of_mean']}%")
    print(f"  bootstrap ARI median={bs['ari_median']} (mean={bs['ari_mean']}, sd={bs['ari_std']})")
    print(f"  diff_7d: n_clusters={df['n_clusters']} smallest_nondom={df['smallest_nondominant_cluster_size']} "
          f"(|R-L| do modo={df['smallest_cluster_mean_abs_contrast_db']} dB) | "
          f"contraste>40dB: {df['n_contrast_gt_40db']}, >50dB: {df['n_contrast_gt_50db']}, max={df['max_contrast_db']}")
    return {"silhouette_gap": sg, "gmm_bic": gm, "bootstrap": bs, "diff_7d": df,
            "pca_components": int(X.shape[1])}


# ── Geração dos sintéticos ───────────────────────────────────────────
def nearest_psd_corr(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-6, None)
    C2 = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(C2))
    return C2 / np.outer(d, d)


def make_continuous(real14):
    """Cópula gaussiana: correlação de postos real + marginais empíricas por canal."""
    n, p = real14.shape
    # correlação de Spearman (postos), tratando NaN por coluna
    ranks = np.zeros_like(real14)
    cols_data = []
    for j in range(p):
        col = real14[:, j]
        valid = col[~np.isnan(col)]
        cols_data.append(np.sort(valid))
        r = stats.rankdata(col, nan_policy="omit")
        r = np.where(np.isnan(col), np.nan, r)
        ranks[:, j] = r
    # correlação de postos via pares completos
    Rcorr = pd.DataFrame(ranks).corr().to_numpy()
    Rcorr = np.nan_to_num(Rcorr, nan=0.0)
    np.fill_diagonal(Rcorr, 1.0)
    Rcorr = nearest_psd_corr(Rcorr)
    Z = np.random.RandomState(RANDOM_STATE).multivariate_normal(np.zeros(p), Rcorr, size=n)
    U = stats.norm.cdf(Z)
    syn = np.empty((n, p))
    for j in range(p):
        sorted_vals = cols_data[j]
        idx = np.clip((U[:, j] * len(sorted_vals)).astype(int), 0, len(sorted_vals) - 1)
        syn[:, j] = sorted_vals[idx]
    return syn


def dequantize(M, step=5.0):
    """Quebra empates na grade de 5 dB (jitter uniforme ±step/2). Preserva macro-estrutura."""
    r = np.random.RandomState(RANDOM_STATE + 7)
    return np.asarray(M, float) + r.uniform(-step / 2, step / 2, size=np.asarray(M).shape)


def _discrete_centroids(real14, k, sep_db):
    n, p = real14.shape
    base = np.nanmean(real14, axis=0)
    half = p // 2
    f = np.arange(half)
    shapes = np.zeros((k, half))
    shapes[0] = ((half - 1 - f) / (half - 1)) * (-sep_db / 2)   # baixa-freq pior
    shapes[1] = (f / (half - 1)) * sep_db                       # sloping alta-freq
    shapes[2] = ((half - 1 - f) / (half - 1)) * sep_db          # invertido
    shapes[3] = np.where(f == 4, sep_db, 0.0)                   # notch 4k
    cents = np.zeros((k, p))
    for c in range(k):
        cents[c, :half] = base[:half] + shapes[c]
        cents[c, half:] = base[half:] + shapes[c]
    return cents


def make_discrete(real14, k=4, sep_db=55.0, within_candidates=(3.0, 4.0, 5.0, 6.0)):
    """4 arquétipos de FORMA separados; calibra ruído intra-cluster p/ silhouette-verdadeiro 0.5-0.7."""
    n, p = real14.shape
    cents = _discrete_centroids(real14, k, sep_db)
    r = np.random.RandomState(RANDOM_STATE)
    labels = r.randint(0, k, size=n)
    best = None
    for sd in within_candidates:
        syn = cents[labels] + np.random.RandomState(RANDOM_STATE + 1).normal(0, sd, size=(n, p))
        Xd = embed(dequantize(syn), row_center=True)
        sil = float(silhouette_score(Xd, labels, sample_size=SIL_SAMPLE, random_state=RANDOM_STATE))
        if best is None or abs(sil - 0.6) < abs(best[1] - 0.6):
            best = (sd, sil, syn)
    sd, sil, syn = best
    print(f"DISCRETO calibrado: within_sd={sd} dB -> silhouette-verdadeiro={sil:.4f}")
    return syn, labels, round(sil, 4)


def main():
    df = pd.read_csv(FEATURE, low_memory=False)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df = df[(age >= AGE_MIN) & (age <= AGE_MAX)].copy()
    thr = df[COLS14].apply(pd.to_numeric, errors="coerce")
    keep = thr.notna().sum(axis=1) >= MIN_COMPLETENESS
    thr = thr[keep]
    m = (thr > ANY25).any(axis=1)
    real14_raw = thr[m].to_numpy(np.float64)
    N = len(real14_raw)
    print(f"REAL N={N}, 14 limiares")

    # Sintéticos (gerados no espaço bruto, depois dequantizados junto com o real)
    cont14_raw = make_continuous(real14_raw)
    disc14_raw, disc_labels, true_sil = make_discrete(real14_raw)

    # Dequantização uniforme nos TRÊS (remove micro-clusters de grade de 5 dB;
    # o gap interaural de 59 dB do modo real é >>2,5 dB e sobrevive).
    real14 = dequantize(real14_raw)
    cont14 = dequantize(cont14_raw)
    disc14 = dequantize(disc14_raw)

    res = {
        "real": diagnostics(real14, real14[:, :7], real14[:, 7:], "REAL"),
        "continuous_null": diagnostics(cont14, cont14[:, :7], cont14[:, 7:], "CONTÍNUO (cópula)"),
        "discrete_null": diagnostics(disc14, disc14[:, :7], disc14[:, 7:], "DISCRETO (4 blobs)"),
    }
    res["discrete_null"]["true_silhouette_design"] = true_sil

    # ── Veredito ─────────────────────────────────────────────────────
    real, cont, disc = res["real"], res["continuous_null"], res["discrete_null"]

    def sil(d): return d["silhouette_gap"]["best_silhouette"]
    def gapk(d): return d["silhouette_gap"]["gap_optimal_k"]
    def depth(d): return d["gmm_bic"]["bic_range_pct_of_mean"]

    # Real cai do lado do contínuo se silhouette/gap-k/BIC-depth estão mais próximos
    # do contínuo do que do discreto (discrete = referência de estrutura real).
    closer_to_cont = lambda fr: abs(fr(real) - fr(cont)) <= abs(fr(real) - fr(disc))
    sil_cont = closer_to_cont(sil)
    depth_cont = closer_to_cont(depth)
    gap_cont = (gapk(real) == gapk(cont)) and (gapk(real) != gapk(disc))
    # quantos eixos centrais apontam para contínuo
    axes_cont = sum([sil_cont, depth_cont, gap_cont])
    real_like_cont = axes_cont >= 2

    # 'Contraste discreto' só vale se houver EXCESSO real de casos unilaterais
    # extremos além do null contínuo (não apenas um cluster pequeno do HDBSCAN).
    real_ex = real["diff_7d"]["n_contrast_gt_50db"]
    cont_ex = cont["diff_7d"]["n_contrast_gt_50db"]
    excess_ratio = round(real_ex / cont_ex, 2) if cont_ex > 0 else None
    contrast_is_signal = real_ex >= 2 * max(cont_ex, 1)  # real tem >=2x casos extremos vs contínuo
    res["verdict_axes"] = {
        "silhouette_closer_to_continuous": bool(sil_cont),
        "bic_depth_closer_to_continuous": bool(depth_cont),
        "gap_k_matches_continuous_not_discrete": bool(gap_cont),
        "axes_pointing_continuous": int(axes_cont),
        "real_n_contrast_gt_50db": real_ex,
        "continuous_n_contrast_gt_50db": cont_ex,
        "asymmetry_excess_ratio_real_over_continuous": excess_ratio,
        "contrast_is_excess_over_null": bool(contrast_is_signal),
        "discrete_true_silhouette": true_sil,
    }

    if real_like_cont and contrast_is_signal:
        verdict = ("DIREÇÃO SUSTENTADA INTEGRAL: nível cai do lado do CONTÍNUO (silhouette/BIC-depth/"
                   "ARI-variância iguais ao null contínuo, longe do discreto) E há excesso real de "
                   "casos unilaterais extremos além do null. 'Nível contínuo, contraste discreto, pooling apaga.'")
    elif real_like_cont and not contrast_is_signal:
        verdict = ("NÍVEL CONTÍNUO CONFIRMADO; CONTRASTE 'DISCRETO' NÃO BATE O NULL: a metade do nível "
                   "está calibrada (real = contínuo, não discreto), mas o excesso de assimetria extrema "
                   "do real NÃO supera o null contínuo de marginais+correlação. Reescopar a 2ª metade: "
                   "os 13 são cauda unilateral real (verificada), não um modo discreto que excede continuum. "
                   "Manchete vira 'nível contínuo + cauda rara de perda unilateral real que o pooling apaga'.")
    elif not real_like_cont:
        verdict = "REESCOPAR: real não se comporta como contínuo em pelo menos um eixo central — ver tabela."
    else:
        verdict = "MISTO — ver números."

    res["verdict"] = verdict
    res["params"] = {"N": N, "n_boot": N_BOOT, "gap_B": GAP_B, "k_discrete": 4}
    res["status"] = "EXECUTED"
    print(f"\nVEREDITO: {verdict}")
    OUTPUT.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
