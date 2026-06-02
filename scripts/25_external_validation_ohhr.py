#!/usr/bin/env python3
"""
Nome: 25_external_validation_ohhr.py
Tarefa: Validação externa usando OHHR (Oldenburg Hearing Health Record).
        Projeta audiogramas OHHR no espaço treinado pelo NHANES.
Input: data/processed/frequencia_feature_matrix_v1.csv
       data/external/ohhr/data/*.json
Output: outputs/json/25_ohhr_validation.json
        outputs/json/25_ohhr_assignments.csv
Dependências: 18_session4_shape_unblock.py (ou execução inline anterior)
"""

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

try:
    import hdbscan
    from hdbscan.prediction import approximate_predict
    HDBSCAN_OK = True
except ImportError:
    HDBSCAN_OK = False

# ── Constantes ───────────────────────────────────────────────────────
RANDOM_STATE = 42
NHANES_FREQ_COLS = [
    "thr_R_500", "thr_R_1000", "thr_R_2000", "thr_R_3000",
    "thr_R_4000", "thr_R_6000", "thr_R_8000",
    "thr_L_500", "thr_L_1000", "thr_L_2000", "thr_L_3000",
    "thr_L_4000", "thr_L_6000", "thr_L_8000",
]
COMMON_FREQ_COLS = ["thr_500", "thr_1000", "thr_2000", "thr_4000"]
OHHR_BASE = Path("data/external/ohhr/data")
NHANES_FEATURE = Path("data/processed/frequencia_feature_matrix_v1.csv")
OUTPUT = Path("outputs/json/25_ohhr_validation.json")
ASSIGN_OUT = Path("outputs/json/25_ohhr_assignments.csv")
LOG = Path("outputs/logs/25_ohhr_validation.log")

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

if OUTPUT.exists():
    log.info(f"Output já existe: {OUTPUT}. Pulando.")
    raise SystemExit(0)


def main():
    if not HDBSCAN_OK:
        raise ImportError("hdbscan não disponível")

    log.info("=" * 60)
    log.info("VALIDAÇÃO EXTERNA — OHHR")
    log.info("=" * 60)

    # ── 1. Carregar OHHR ────────────────────────────────────────────
    # CORREÇÃO (2026-06): a versão anterior unia points.audiogramlineid a
    # audiogram.audiogramid (chaves de espaços diferentes), casando só 3.433
    # de 20.538 pontos, e não filtrava tipo/transdutor/orelha. A cadeia
    # correta é: audiogram_point -> audiogram_line (side, transducertype,
    # type) -> audiogram (clientid). Filtra HTL (limiar, não UCL) + AC
    # (condução aérea, não óssea).
    log.info("Carregando OHHR...")
    anam = pd.DataFrame(json.loads((OHHR_BASE / "anamnesis.json").read_text()))
    audiogram = pd.DataFrame(json.loads((OHHR_BASE / "audiogram.json").read_text()))
    points = pd.DataFrame(json.loads((OHHR_BASE / "audiogram_point.json").read_text()))
    line = pd.DataFrame(json.loads((OHHR_BASE / "audiogram_line.json").read_text()))
    dtt = pd.DataFrame(json.loads((OHHR_BASE / "digit_triplets_test.json").read_text()))

    ohhr_freqs = [500, 1000, 2000, 4000]
    freq_map = {500: "thr_500", 1000: "thr_1000", 2000: "thr_2000", 4000: "thr_4000"}

    pts = points.merge(
        line[["audiogramlineid", "audiogramid", "side", "transducertype", "type"]],
        on="audiogramlineid", how="left",
    )
    pts = pts.merge(audiogram[["audiogramid", "clientid"]], on="audiogramid", how="left")
    pts = pts[
        (pts["type"] == "htl")
        & (pts["transducertype"] == "ac")
        & (pts["frequency"].isin(ohhr_freqs))
    ].copy()
    log.info(f"OHHR pontos válidos (HTL+AC): {len(pts)} de {len(points)}")

    # Média binaural por cliente (coerente com o NHANES binaural-mean) -> clustering
    wide = pts.pivot_table(
        index="clientid", columns="frequency", values="level", aggfunc="mean"
    ).reset_index()
    wide.columns = ["clientid"] + [freq_map[f] for f in sorted(wide.columns[1:])]

    # PTA por orelha -> melhor orelha (menor PTA) para a correlação com SRT
    ear = pts.pivot_table(
        index=["clientid", "side"], columns="frequency", values="level", aggfunc="mean"
    )
    ear["PTA_ear"] = ear[ohhr_freqs].mean(axis=1)
    pta_best = ear.reset_index().groupby("clientid")["PTA_ear"].min().rename("PTA_best")

    df = wide.merge(anam[["clientid", "sex", "birthday_year"]], on="clientid", how="left")
    df["age"] = 2014 - df["birthday_year"]  # dados coletados ~2014
    df = df.merge(pta_best, on="clientid", how="left")
    dtt_binaural = dtt[dtt["side"] == "binaural"][["clientid", "SRT"]].drop_duplicates("clientid")
    df = df.merge(dtt_binaural, on="clientid", how="left")
    log.info(f"OHHR carregado: {len(df)} pessoas")

    # ── 2. Carregar NHANES e filtrar ────────────────────────────────
    log.info("Carregando NHANES...")
    nhanes = pd.read_csv(NHANES_FEATURE, low_memory=False)
    age = pd.to_numeric(nhanes["RIDAGEYR"], errors="coerce")
    nhanes = nhanes[(age >= 20) & (age <= 69)].copy()
    thr = nhanes[NHANES_FREQ_COLS].apply(pd.to_numeric, errors="coerce")
    valid = thr.notna().sum(axis=1)
    nhanes = nhanes[valid >= 10].copy()
    thr = thr[valid >= 10].copy()
    any25 = (thr > 25).any(axis=1)
    nhanes = nhanes[any25].copy()
    thr = thr[any25].copy()
    log.info(f"NHANES ANY25: {len(nhanes)} pessoas")

    # ── 3. Espaço comum (4 frequências, binaural média) ─────────────
    nhanes["c500"] = nhanes[["thr_R_500", "thr_L_500"]].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1, skipna=True)
    nhanes["c1000"] = nhanes[["thr_R_1000", "thr_L_1000"]].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1, skipna=True)
    nhanes["c2000"] = nhanes[["thr_R_2000", "thr_L_2000"]].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1, skipna=True)
    nhanes["c4000"] = nhanes[["thr_R_4000", "thr_L_4000"]].apply(
        pd.to_numeric, errors="coerce"
    ).mean(axis=1, skipna=True)

    cc = ["c500", "c1000", "c2000", "c4000"]
    thr_n = nhanes[cc].apply(pd.to_numeric, errors="coerce")
    thr_o = df[COMMON_FREQ_COLS].apply(pd.to_numeric, errors="coerce")

    # Row-centering
    X_n = thr_n.sub(thr_n.mean(axis=1, skipna=True), axis=0).fillna(0.0).to_numpy(np.float32)
    X_o = thr_o.sub(thr_o.mean(axis=1, skipna=True), axis=0).fillna(0.0).to_numpy(np.float32)

    # Scaling + PCA (fit NHANES, transform OHHR)
    scaler = RobustScaler(quantile_range=(25, 75), unit_variance=False)
    X_n_s = scaler.fit_transform(X_n).astype(np.float32)
    X_o_s = scaler.transform(X_o).astype(np.float32)

    pca = PCA(n_components=min(4, X_n_s.shape[1]), random_state=RANDOM_STATE)
    X_n_pca = pca.fit_transform(X_n_s).astype(np.float32)
    X_o_pca = pca.transform(X_o_s).astype(np.float32)
    log.info(f"PCA: {X_n_pca.shape[1]} componentes, variância={pca.explained_variance_ratio_.sum():.4f}")

    # ── 4. HDBSCAN no NHANES + approximate_predict no OHHR ─────────
    c = hdbscan.HDBSCAN(
        min_cluster_size=10, min_samples=5, metric="euclidean",
        cluster_selection_method="eom", prediction_data=True, core_dist_n_jobs=-1
    )
    labels_n = c.fit_predict(X_n_pca)
    nc_n = len(set(labels_n) - {-1})
    nn_n = int(sum(labels_n == -1))
    log.info(f"NHANES: {nc_n} clusters, {nn_n} ruído ({nn_n / len(labels_n):.4f})")

    labels_o, strengths_o = approximate_predict(c, X_o_pca)
    nc_o = len(set(labels_o) - {-1})
    nn_o = int(sum(labels_o == -1))
    log.info(f"OHHR projetado: {nc_o} clusters, {nn_o} ruído ({nn_o / len(labels_o):.4f})")

    # ── 5. PTA × SRT correlação ─────────────────────────────────────
    # PTA = melhor orelha, limiares BRUTOS (não row-centered). SRT vem do
    # Digit Triplets Test (fala-no-ruído / SNR, ruído fixo 65 dB).
    df["PTA"] = df["PTA_best"]
    valid_corr = df[["PTA", "SRT"]].dropna()
    pr = stats.pearsonr(valid_corr["PTA"], valid_corr["SRT"])
    sr = stats.spearmanr(valid_corr["PTA"], valid_corr["SRT"])
    log.info(f"PTA × SRT: Pearson r={pr.statistic:.4f} (p={pr.pvalue:.6f})")
    log.info(f"PTA × SRT: Spearman r={sr.statistic:.4f} (p={sr.pvalue:.6f})")

    # ── 6. Distribuição comparativa ─────────────────────────────────
    dist_n = pd.Series(labels_n).value_counts(normalize=True).sort_index()
    dist_o = pd.Series(labels_o).value_counts(normalize=True).sort_index()

    # ── 7. Salvar ───────────────────────────────────────────────────
    result = {
        "script": "25_external_validation_ohhr.py",
        "random_state": RANDOM_STATE,
        "ohhr_n": int(len(df)),
        "nhanes_n": int(len(nhanes)),
        "common_frequencies": [500, 1000, 2000, 4000],
        "pca_variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
        "nhanes_common": {
            "n_clusters": nc_n,
            "n_noise": nn_n,
            "noise_fraction": round(nn_n / len(labels_n), 4),
        },
        "ohhr_projected": {
            "n_clusters": nc_o,
            "n_noise": nn_o,
            "noise_fraction": round(nn_o / len(labels_o), 4),
        },
        "pta_srt_correlation": {
            "n": int(len(valid_corr)),
            "pearson_r": round(float(pr.statistic), 6),
            "pearson_p": float(pr.pvalue),
            "spearman_r": round(float(sr.statistic), 6),
            "spearman_p": float(sr.pvalue),
        },
        "distribution_comparison": {
            str(cid): {
                "nhanes_pct": round(float(dist_n.get(cid, 0)) * 100, 1),
                "ohhr_pct": round(float(dist_o.get(cid, 0)) * 100, 1),
            }
            for cid in sorted(set(list(dist_n.index) + list(dist_o.index)))
        },
        "status": "EXECUTED — validação externa, sem rótulo clínico",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # Salvar assignments
    df_out = df[["clientid", "age", "sex", "SRT", "PTA"]].copy()
    for col in COMMON_FREQ_COLS:
        df_out[col] = df[col]
    df_out["projected_cluster"] = labels_o
    df_out["membership_strength"] = strengths_o
    ASSIGN_OUT.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(ASSIGN_OUT, index=False)

    log.info(f"Concluído. Output: {OUTPUT}")
    log.info(f"Assignments: {ASSIGN_OUT}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
