#!/usr/bin/env python3
"""
Nome: 21_session5_rf_surrogate_v2.py
Tarefa: RF surrogate para explicar o que separa cluster 0 de cluster 1.
Input: outputs/json/session4_assignments_any25.csv, data/processed/frequencia_feature_matrix_v1.csv
Output: outputs/json/21_rf_surrogate_v2.json
"""

import logging, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
FREQ_COLS = [
    "thr_R_500","thr_R_1000","thr_R_2000","thr_R_3000","thr_R_4000","thr_R_6000","thr_R_8000",
    "thr_L_500","thr_L_1000","thr_L_2000","thr_L_3000","thr_L_4000","thr_L_6000","thr_L_8000",
]
ASSIGN = Path("outputs/json/session4_assignments_any25.csv")
FEATURE = Path("data/processed/frequencia_feature_matrix_v1.csv")
OUTPUT = Path("outputs/json/21_rf_surrogate_v2.json")
LOG = Path("outputs/logs/21_rf_surrogate_v2.log")

LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(LOG), logging.StreamHandler()])
log = logging.getLogger(__name__)

if OUTPUT.exists():
    log.info(f"Output já existe: {OUTPUT}. Pulando.")
    raise SystemExit(0)


def top_split_features(rf, features):
    counts = Counter()
    thresholds = defaultdict(list)
    for est in rf.estimators_:
        tree = est.tree_
        for idx, thr in zip(tree.feature[tree.feature >= 0], tree.threshold[tree.feature >= 0]):
            fname = features[int(idx)]
            counts[fname] += 1
            thresholds[fname].append(float(thr))
    rows = []
    for fname, count in counts.most_common(14):
        th = np.array(thresholds[fname])
        rows.append({"feature": fname, "split_count": int(count),
                      "threshold_median_scaled": round(float(np.median(th)), 4),
                      "threshold_p25_scaled": round(float(np.quantile(th, 0.25)), 4),
                      "threshold_p75_scaled": round(float(np.quantile(th, 0.75)), 4)})
    return rows


def main():
    log.info("=" * 60)
    log.info("SESSÃO 5 — TAREFA 3: RF Surrogate v2")
    log.info("=" * 60)

    assign = pd.read_csv(ASSIGN)
    df = pd.read_csv(FEATURE, low_memory=False)
    df = df.merge(assign[["SEQN", "cluster_id"]], on="SEQN", how="inner", validate="one_to_one")

    # Excluir outliers
    df = df[df["cluster_id"] != -1].copy()
    log.info(f"Amostras não-ruído: {len(df)}")
    log.info(f"Distribuição: {dict(df['cluster_id'].value_counts().sort_index())}")

    # Preparar features
    X_raw = df[FREQ_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    y = (df["cluster_id"] == 1).astype(int).to_numpy()  # 1 = cluster 1 (assimetria), 0 = cluster 0

    scaler = RobustScaler(quantile_range=(25.0, 75.0), unit_variance=False)
    X = scaler.fit_transform(X_raw).astype(np.float32)

    log.info(f"Classes: 0={int(np.sum(y==0))}, 1={int(np.sum(y==1))}")

    # ── Stratified K-Fold com AUC-ROC ───────────────────────────────
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                 n_jobs=-1, random_state=RANDOM_STATE, max_features="sqrt")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    auc_scores = []
    ap_scores = []
    reports = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf_fold = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                          n_jobs=-1, random_state=RANDOM_STATE, max_features="sqrt")
        rf_fold.fit(X_train, y_train)

        if hasattr(rf_fold, "predict_proba"):
            y_proba = rf_fold.predict_proba(X_test)[:, 1]
            try:
                auc = float(roc_auc_score(y_test, y_proba))
                ap = float(average_precision_score(y_test, y_proba))
            except ValueError:
                auc = None
                ap = None
        else:
            auc, ap = None, None

        y_pred = rf_fold.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        auc_scores.append(auc)
        ap_scores.append(ap)
        reports.append({"fold": fold, "auc": auc, "avg_precision": ap,
                        "n_test": int(len(y_test)), "n_class1_test": int(np.sum(y_test == 1))})
        log.info(f"  Fold {fold}: AUC={auc}, AP={ap}, n_test={len(y_test)}, n_1={np.sum(y_test==1)}")

    mean_auc = float(np.mean([s for s in auc_scores if s is not None])) if any(auc_scores) else None
    mean_ap = float(np.mean([s for s in ap_scores if s is not None])) if any(ap_scores) else None
    log.info(f"AUC médio: {mean_auc}, AP médio: {mean_ap}")

    # ── Treinar no dataset completo para importances ─────────────────
    rf.fit(X, y)
    train_acc = float((rf.predict(X) == y).mean())

    # Gini importance
    gini = pd.DataFrame({"feature": FREQ_COLS, "importance": rf.feature_importances_})
    gini = gini.sort_values("importance", ascending=False)
    log.info(f"\nTop 7 features (Gini):")
    for _, row in gini.head(7).iterrows():
        log.info(f"  {row['feature']}: {row['importance']:.6f}")

    # Permutation importance
    perm = permutation_importance(rf, X, y, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1, scoring="roc_auc")
    perm_df = pd.DataFrame({"feature": FREQ_COLS, "perm_mean": perm.importances_mean, "perm_std": perm.importances_std})
    perm_df = perm_df.sort_values("perm_mean", ascending=False)

    # Top splits
    splits = top_split_features(rf, FREQ_COLS)

    # ── Salvar ───────────────────────────────────────────────────────
    result = {
        "script": "21_session5_rf_surrogate_v2.py",
        "random_state": RANDOM_STATE,
        "n_samples": int(len(df)),
        "n_class0": int(np.sum(y == 0)),
        "n_class1": int(np.sum(y == 1)),
        "class_weight": "balanced",
        "n_estimators": 500,
        "stratified_kfold": {
            "n_splits": 5,
            "mean_auc_roc": round(mean_auc, 6) if mean_auc else None,
            "mean_avg_precision": round(mean_ap, 6) if mean_ap else None,
            "per_fold": reports,
        },
        "train_accuracy": round(train_acc, 6),
        "gini_importance_top14": gini.to_dict(orient="records"),
        "permutation_importance_top14": perm_df.round(6).to_dict(orient="records"),
        "top_split_features": splits,
        "note": "Classes desbalanceadas (7098 vs 12). AUC/AP são métricas principais, não acurácia.",
        "status": "EXECUTED — sem rótulo clínico",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    log.info(f"Concluído. Output: {OUTPUT}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
