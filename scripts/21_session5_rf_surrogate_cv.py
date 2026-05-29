#!/usr/bin/env python3
"""
Nome: 21_session5_rf_surrogate_cv.py
Tarefa: RF surrogate com validação cruzada estratificada + PR-AUC + Leave-One-Out nos 12 positivos.
Input: outputs/json/session4_assignments_any25.csv, data/processed/frequencia_feature_matrix_v1.csv
Output: outputs/json/rf_surrogate_cv.json
"""

import logging, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

RANDOM_STATE = 42
FREQ_COLS = [
    "thr_R_500","thr_R_1000","thr_R_2000","thr_R_3000","thr_R_4000","thr_R_6000","thr_R_8000",
    "thr_L_500","thr_L_1000","thr_L_2000","thr_L_3000","thr_L_4000","thr_L_6000","thr_L_8000",
]
ASSIGN = Path("outputs/json/session4_assignments_any25.csv")
FEATURE = Path("data/processed/frequencia_feature_matrix_v1.csv")
OUTPUT = Path("outputs/json/rf_surrogate_cv.json")
LOG = Path("outputs/logs/21_rf_surrogate_cv.log")

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger(__name__)


def main():
    log.info("=" * 60)
    log.info("RF Surrogate CV — StratifiedKFold + PR-AUC + LOO nos 12 positivos")
    log.info("=" * 60)

    assign = pd.read_csv(ASSIGN)
    df = pd.read_csv(FEATURE, low_memory=False)
    df = df.merge(assign[["SEQN", "cluster_id"]], on="SEQN", how="inner", validate="one_to_one")
    df = df[df["cluster_id"] != -1].copy()
    log.info(f"Amostras não-ruído: {len(df)}")

    X_raw = df[FREQ_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = (df["cluster_id"] == 1).astype(int).to_numpy()

    scaler = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=False)
    X = scaler.fit_transform(X_raw).astype(np.float32)

    n0, n1 = int(np.sum(y == 0)), int(np.sum(y == 1))
    log.info(f"Classes: 0={n0}, 1={n1}")

    # ── 1. StratifiedKFold k=5 ──────────────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    auc_scores, ap_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        rf_f = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                       n_jobs=-1, random_state=RANDOM_STATE, max_features="sqrt")
        rf_f.fit(X[train_idx], y[train_idx])
        y_proba = rf_f.predict_proba(X[test_idx])[:, 1]
        y_test = y[test_idx]
        try:
            auc = float(roc_auc_score(y_test, y_proba))
            ap  = float(average_precision_score(y_test, y_proba))
        except ValueError:
            auc, ap = None, None
        auc_scores.append(auc)
        ap_scores.append(ap)
        fold_results.append({"fold": fold, "auc_roc": auc, "pr_auc": ap,
                              "n_test": int(len(y_test)), "n_pos_test": int(y_test.sum())})
        log.info(f"  Fold {fold}: AUC-ROC={auc:.4f}, PR-AUC={ap:.4f}, n_pos={y_test.sum()}")

    mean_auc = float(np.mean([s for s in auc_scores if s is not None]))
    mean_ap  = float(np.mean([s for s in ap_scores  if s is not None]))
    log.info(f"Média k=5: AUC-ROC={mean_auc:.4f}, PR-AUC={mean_ap:.4f}")

    # ── 2. Leave-One-Out nos 12 positivos ───────────────────────────
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    loo_results = []

    log.info(f"\nLOO nos {len(pos_idx)} positivos do Cluster 1:")
    for left_out in pos_idx:
        # treinar em todos menos este positivo (mantém todos negativos)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[left_out] = False
        rf_loo = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                         n_jobs=-1, random_state=RANDOM_STATE, max_features="sqrt")
        rf_loo.fit(X[train_mask], y[train_mask])
        prob_left = float(rf_loo.predict_proba(X[[left_out]])[:, 1][0])
        pred_left = int(rf_loo.predict(X[[left_out]])[0])
        seqn_val = int(df.iloc[left_out]["SEQN"]) if "SEQN" in df.columns else None
        loo_results.append({"seqn": seqn_val, "prob_class1": round(prob_left, 4),
                             "predicted_class": pred_left, "correctly_identified": pred_left == 1})
        log.info(f"  SEQN={seqn_val}: prob={prob_left:.4f}, pred={pred_left}, correto={pred_left==1}")

    n_correct = sum(r["correctly_identified"] for r in loo_results)
    loo_recall = n_correct / len(pos_idx)
    log.info(f"LOO recall: {n_correct}/{len(pos_idx)} = {loo_recall:.4f}")

    # ── Salvar ───────────────────────────────────────────────────────
    result = {
        "script": "21_session5_rf_surrogate_cv.py",
        "random_state": RANDOM_STATE,
        "n_samples": int(len(df)),
        "n_class0": n0,
        "n_class1": n1,
        "class_weight": "balanced",
        "n_estimators": 500,
        "note": (
            "AUC-ROC=1.0 e PR-AUC=1.0 são esperados — rótulos derivam das mesmas features (PCA→HDBSCAN). "
            "Serve apenas para ranquear features dominantes, não como evidência de validade do cluster. "
            "LOO confirma quais positivos são reconhecíveis fora do treino."
        ),
        "stratified_kfold_k5": {
            "mean_auc_roc": round(mean_auc, 4),
            "mean_pr_auc": round(mean_ap, 4),
            "std_auc_roc": round(float(np.std([s for s in auc_scores if s is not None])), 4),
            "std_pr_auc":  round(float(np.std([s for s in ap_scores  if s is not None])), 4),
            "per_fold": fold_results,
        },
        "leave_one_out_positives": {
            "n_positives": int(len(pos_idx)),
            "n_correctly_identified": int(n_correct),
            "loo_recall": round(loo_recall, 4),
            "per_sample": loo_results,
        },
        "interpretation": (
            f"CV estratificada k=5: AUC-ROC={mean_auc:.4f}, PR-AUC={mean_ap:.4f} (circular — ver nota). "
            f"LOO: {n_correct}/{len(pos_idx)} positivos reconhecidos quando excluídos do treino."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log.info(f"\nOutput salvo: {OUTPUT}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
