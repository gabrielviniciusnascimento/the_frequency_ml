#!/usr/bin/env python3
"""
Nome: 24_session5_personal_projection.py
Tarefa: Projetar caso pessoal no espaço treinado da Sessão 4.
Input: outputs/json/session4_pca_scaler_params.json, outputs/json/session4_centroids.json
Output: outputs/json/24_personal_projection.json
"""

import logging, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import hdbscan

RANDOM_STATE = 42
FREQ_COLS = [
    "thr_R_500","thr_R_1000","thr_R_2000","thr_R_3000","thr_R_4000","thr_R_6000","thr_R_8000",
    "thr_L_500","thr_L_1000","thr_L_2000","thr_L_3000","thr_L_4000","thr_L_6000","thr_L_8000",
]
ASSIGN = Path("outputs/json/session4_assignments_any25.csv")
FEATURE = Path("data/processed/frequencia_feature_matrix_v1.csv")
OUTPUT = Path("outputs/json/24_personal_projection.json")
LOG = Path("outputs/logs/24_personal_projection.log")

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger(__name__)

if OUTPUT.exists():
    log.info(f"Output já existe: {OUTPUT}. Pulando.")
    raise SystemExit(0)


def main():
    log.info("=" * 60)
    log.info("SESSÃO 5 — Projeção do Caso Pessoal")
    log.info("=" * 60)

    # ── 1. Carregar dados da Sessão 4 e reconstruir espaço ───────────
    assign = pd.read_csv(ASSIGN)
    df = pd.read_csv(FEATURE, low_memory=False)
    df = df.merge(assign[["SEQN", "cluster_id"]], on="SEQN", how="inner", validate="one_to_one")

    # Filtrar ANY25 (mesma pipeline da Sessão 4)
    age = pd.to_numeric(df["RIDAGEYR"], errors="coerce")
    df = df[(age >= 20) & (age <= 69)].copy()
    thr = df[FREQ_COLS].apply(pd.to_numeric, errors="coerce")
    valid = thr.notna().sum(axis=1)
    df = df[valid >= 10].copy()
    thr = thr[valid >= 10].copy()
    any25 = (thr > 25).any(axis=1)
    df = df[any25].copy()
    thr = thr[any25].copy()

    # Row-centering
    row_means = thr.mean(axis=1, skipna=True)
    X_shape = thr.sub(row_means, axis=0).fillna(0.0)
    X = X_shape.to_numpy(dtype=np.float32)

    # Scaling + PCA
    scaler = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=False)
    X_scaled = scaler.fit_transform(X).astype(np.float32)
    pca = PCA(n_components=0.95, svd_solver="full", random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled).astype(np.float32)

    # HDBSCAN
    c = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean",
                         cluster_selection_method="eom", prediction_data=True, core_dist_n_jobs=-1)
    labels = c.fit_predict(X_pca)

    # Centroides
    centroids = {}
    for cid in sorted(set(labels) - {-1}):
        centroids[int(cid)] = X_pca[labels == cid].mean(axis=0)

    log.info(f"Espaço reconstruído: {len(df)} pontos, {X_pca.shape[1]} dims PCA")
    log.info(f"Clusters: {len(centroids)}")

    # ── 2. Definir caso pessoal ──────────────────────────────────────
    # IMPORTANTE: estes são valores HIPOTÉTICOS para demonstração.
    # O usuário deve substituir por seus audiogramas reais.
    #
    # Cenário A: caso típico de ototoxicidade por platina
    #   - bilateral, simétrico, sloping severo em altas frequências
    #   - ouvido direito e esquerdo similares
    personal_platina = {
        "label": "hipotético_platina_like",
        "description": "Cenário A: ototoxicidade bilateral simétrica, sloping severo altas frequências",
        "age": 30, "sex": 1,
        "thr_R_500": 15, "thr_R_1000": 15, "thr_R_2000": 20, "thr_R_3000": 40,
        "thr_R_4000": 60, "thr_R_6000": 75, "thr_R_8000": 80,
        "thr_L_500": 15, "thr_L_1000": 15, "thr_L_2000": 20, "thr_L_3000": 40,
        "thr_L_4000": 60, "thr_L_6000": 75, "thr_L_8000": 80,
    }

    # Cenário B: caso com progressão atípica + distorção
    personal_atipico = {
        "label": "hipotético_progressao_atipica",
        "description": "Cenário B: progressão atípica, assimetria leve, pior em altas frequências",
        "age": 30, "sex": 1,
        "thr_R_500": 10, "thr_R_1000": 10, "thr_R_2000": 15, "thr_R_3000": 35,
        "thr_R_4000": 55, "thr_R_6000": 70, "thr_R_8000": 75,
        "thr_L_500": 15, "thr_L_1000": 20, "thr_L_2000": 25, "thr_L_3000": 50,
        "thr_L_4000": 65, "thr_L_6000": 80, "thr_L_8000": 85,
    }

    # Cenário C: audiograma normal (controle)
    personal_normal = {
        "label": "hipotético_normal",
        "description": "Cenário C: audição normal (controle)",
        "age": 30, "sex": 1,
        "thr_R_500": 10, "thr_R_1000": 10, "thr_R_2000": 10, "thr_R_3000": 10,
        "thr_R_4000": 10, "thr_R_6000": 10, "thr_R_8000": 10,
        "thr_L_500": 10, "thr_L_1000": 10, "thr_L_2000": 10, "thr_L_3000": 10,
        "thr_L_4000": 10, "thr_L_6000": 10, "thr_L_8000": 10,
    }

    scenarios = [personal_platina, personal_atipico, personal_normal]
    projections = []

    for scenario in scenarios:
        log.info(f"\n--- Cenário: {scenario['label']} ---")

        # Extrair thresholds
        thr_vals = np.array([[scenario[c] for c in FREQ_COLS]], dtype=np.float32)

        # Row-centering
        row_mean = float(thr_vals.mean())
        thr_centered = thr_vals - row_mean

        # Scaling (usar o mesmo scaler da Sessão 4)
        thr_scaled = scaler.transform(thr_centered).astype(np.float32)

        # PCA
        thr_pca = pca.transform(thr_scaled).astype(np.float32)

        # Distância a cada centroide
        dists = {}
        for cid, centroid in centroids.items():
            d = float(np.sqrt(np.sum((thr_pca[0] - centroid) ** 2)))
            dists[cid] = round(d, 4)

        # Distância ao centroide mais próximo
        nearest_cluster = min(dists, key=dists.get)
        nearest_dist = dists[nearest_cluster]

        # Distância média dos pontos do cluster mais próximo
        nearest_mask = labels == nearest_cluster
        dists_within = np.sqrt(np.sum((X_pca[nearest_mask] - centroids[nearest_cluster]) ** 2, axis=1))
        mean_within = float(dists_within.mean())
        std_within = float(dists_within.std())
        percentile = float((dists_within < nearest_dist).mean() * 100)

        # Posição no espaço PCA
        pca_coords = thr_pca[0].tolist()

        result = {
            "label": scenario["label"],
            "description": scenario["description"],
            "age": scenario["age"],
            "thresholds_db": {c: scenario[c] for c in FREQ_COLS},
            "row_centering_mean": round(row_mean, 2),
            "pca_coordinates": [round(v, 4) for v in pca_coords],
            "distances_to_centroids": {str(k): v for k, v in dists.items()},
            "nearest_cluster": int(nearest_cluster),
            "nearest_distance": round(nearest_dist, 4),
            "mean_distance_within_nearest_cluster": round(mean_within, 4),
            "std_distance_within_nearest_cluster": round(std_within, 4),
            "percentile_within_nearest": round(percentile, 1),
        }
        projections.append(result)

        log.info(f"  Centroide mais próximo: cluster {nearest_cluster} (dist={nearest_dist:.4f})")
        log.info(f"  Distância média intra-cluster: {mean_within:.4f} (±{std_within:.4f})")
        log.info(f"  Percentil dentro do cluster: {percentile:.1f}%")
        if nearest_dist > mean_within + 2 * std_within:
            log.info(f"  → OUTLIER (>2σ do centroide)")
        elif nearest_dist > mean_within + std_within:
            log.info(f"  → Periferia (>1σ do centroide)")
        else:
            log.info(f"  → Dentro do corpo principal")

    # ── 3. Salvar ────────────────────────────────────────────────────
    result = {
        "script": "24_session5_personal_projection.py",
        "random_state": RANDOM_STATE,
        "n_space": len(df),
        "pca_components": int(X_pca.shape[1]),
        "n_clusters": len(centroids),
        "cluster_info": {
            str(cid): {
                "n": int((labels == cid).sum()),
                "centroid_pca": [round(float(v), 4) for v in centroids[cid]],
            } for cid in sorted(centroids.keys())
        },
        "note": "Valores de threshold são HIPOTÉTICOS. Substituir por audiogramas reais do caso pessoal.",
        "projections": projections,
        "status": "EXECUTED — sem rótulo clínico",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    log.info(f"\nConcluído. Output: {OUTPUT}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
