#!/usr/bin/env python3
"""
Nome: 34_tinnitus_reexamination.py
Tarefa: Reexaminar, sob o escrutínio atual, a alegação tinnitus "38% vs 18%" que
        apareceu nos drafts v1-v4 e SUMIU em v5/v6 sem registro. A alegação original
        era *cluster-based* (outliers/Cluster-1 vs Cluster-0); como o cluster foi
        rebaixado para cauda de um contínuo (audit_06, METHODS_NOTE), a unidade
        defensável passou a ser o LIMIAR |z| de assimetria e o gradiente de severidade,
        não "o cluster". Este script:
          (1) reproduz o número cluster-based (HDBSCAN canônico) para citação honesta;
          (2) re-testa na unidade defensável (cauda |z| de assimetria; severidade pta_high)
              com Fisher exato + IC bootstrap + null por permutação;
          (3) emite um `verdict` que decide reinstaurar-com-ressalvas vs aposentar.

Input: data/processed/frequencia_feature_matrix_v1.csv (AUQ191 já presente — o
       frequencia_bruto.csv do script 16 não é necessário).
Output: outputs/json/34_tinnitus_reexamination.json
Run: .venv/Scripts/python.exe scripts/34_tinnitus_reexamination.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from _shape_space import load_cohort, shape_embed, R_COLS, L_COLS, lib_versions  # noqa: E402

import hdbscan  # noqa: E402

OUT = ROOT / "outputs" / "json" / "34_tinnitus_reexamination.json"
RANDOM_STATE = 42
B_BOOT = 2000
B_PERM = 2000
Z_LEVELS = [2.0, 3.0, 4.0]


def tinnitus_from_auq191(series: pd.Series) -> np.ndarray:
    """AUQ191: 1=yes, 2=no (same coding as scripts/16_tinnitus_audit.py). Else NaN."""
    auq = pd.to_numeric(series, errors="coerce")
    return np.select([auq == 1, auq == 2], [1.0, 0.0], default=np.nan).astype("float64")


def rate(mask_group: np.ndarray, t: np.ndarray) -> dict:
    """Tinnitus rate among non-missing within a boolean group mask."""
    sel = mask_group & np.isfinite(t)
    n = int(sel.sum())
    yes = int(np.nansum(t[sel]))
    return {"n_nonmissing": n, "yes": yes, "rate": (yes / n) if n else None}


def fisher_and_boot(in_tail: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> dict:
    """Fisher exact 2x2 (tail vs body) + bootstrap CI on the rate ratio (tail/body)."""
    fin = np.isfinite(t)
    tail = in_tail & fin
    body = (~in_tail) & fin
    a, b = int(np.nansum(t[tail])), int(tail.sum() - np.nansum(t[tail]))  # yes/no in tail
    c, d = int(np.nansum(t[body])), int(body.sum() - np.nansum(t[body]))  # yes/no in body
    odds, p_fisher = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    r_tail = a / (a + b) if (a + b) else np.nan
    r_body = c / (c + d) if (c + d) else np.nan
    ratio = (r_tail / r_body) if (r_body and np.isfinite(r_body) and r_body > 0) else None
    # bootstrap the ratio over individuals (non-missing only)
    idx_tail = np.where(tail)[0]
    idx_body = np.where(body)[0]
    boots = []
    if len(idx_tail) and len(idx_body):
        for _ in range(B_BOOT):
            bt = t[rng.choice(idx_tail, len(idx_tail), replace=True)].mean()
            bb = t[rng.choice(idx_body, len(idx_body), replace=True)].mean()
            if bb > 0:
                boots.append(bt / bb)
    ci = (
        [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        if boots else None
    )
    return {
        "tail": {"n_nonmissing": a + b, "yes": a, "rate": r_tail},
        "body": {"n_nonmissing": c + d, "yes": c, "rate": r_body},
        "rate_ratio_tail_over_body": ratio,
        "fisher_p_greater": float(p_fisher),
        "bootstrap_ratio_ci95": ci,
    }


def mantel_haenszel(strata: list[tuple[int, int, int, int]]) -> dict:
    """Severity-ADJUSTED association of asymmetry-tail -> tinnitus, pooled over severity
    strata. Each stratum = (a,b,c,d): a=tail&yes, b=tail&no, c=body&yes, d=body&no.
    Returns MH odds ratio, Robins-Breslow-Greenland 95% CI, and MH chi-square p.
    This is the test that separates a REAL asymmetry effect from severity confounding
    (the severity main-effect is non-novel and must not carry the verdict)."""
    R = S = 0.0
    num_p = den_pq = num_q = 0.0  # RBG variance accumulators (3 terms)
    a_sum = e_sum = v_sum = 0.0   # MH chi-square accumulators
    used = 0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n == 0 or (a + b) == 0 or (c + d) == 0:
            continue
        used += 1
        Rk = a * d / n
        Sk = b * c / n
        R += Rk
        S += Sk
        # RBG: Var(lnOR) = ΣP·R/(2R²) + Σ(P·S+Q·R)/(2RS) + ΣQ·S/(2S²)
        P = (a + d) / n
        Q = (b + c) / n
        num_p += P * Rk
        den_pq += P * Sk + Q * Rk
        num_q += Q * Sk
        # chi-square (a vs its expectation under independence within stratum)
        a_sum += a
        e_sum += (a + b) * (a + c) / n
        if n > 1:
            v_sum += (a + b) * (c + d) * (a + c) * (b + d) / (n * n * (n - 1))
    if S == 0 or R == 0 or used == 0:
        return {"mh_or": None, "ci95": None, "chi2_p": None, "strata_used": used}
    mh_or = R / S
    # full Robins-Breslow-Greenland variance of ln(OR_MH) — all three terms
    var_ln = (num_p / (2 * R * R)) + (den_pq / (2 * R * S)) + (num_q / (2 * S * S))
    ci = None
    if var_ln and var_ln > 0:
        import math
        se = math.sqrt(var_ln)
        ci = [float(np.exp(np.log(mh_or) - 1.96 * se)), float(np.exp(np.log(mh_or) + 1.96 * se))]
    chi2 = ((a_sum - e_sum) ** 2) / v_sum if v_sum > 0 else None
    p = float(stats.chi2.sf(chi2, 1)) if chi2 is not None else None
    return {"mh_or": float(mh_or), "ci95": ci, "chi2": (float(chi2) if chi2 else None),
            "chi2_p": p, "strata_used": used}


def perm_null(in_tail: np.ndarray, t: np.ndarray, rng: np.random.Generator) -> dict:
    """Permutation null for the rate DIFFERENCE (tail-body): shuffle tail membership
    among non-missing individuals; how often does the null beat the observed gap?"""
    fin = np.isfinite(t)
    tv = t[fin]
    tail_fin = in_tail[fin]
    n_tail = int(tail_fin.sum())
    if n_tail == 0 or n_tail == len(tv):
        return {"observed_diff": None, "p_emp": None}
    obs = tv[tail_fin].mean() - tv[~tail_fin].mean()
    ge = 0
    n = len(tv)
    for _ in range(B_PERM):
        perm = rng.permutation(n) < n_tail
        if (tv[perm].mean() - tv[~perm].mean()) >= obs:
            ge += 1
    return {"observed_diff": float(obs), "p_emp": (ge + 1) / (B_PERM + 1)}


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    df, thr = load_cohort()
    emb = shape_embed(thr)

    # --- tinnitus variable (from the same aligned cohort frame) ---
    if "AUQ191" not in df.columns:
        raise SystemExit("AUQ191 ausente na feature matrix — não dá para reexaminar tinnitus.")
    t = tinnitus_from_auq191(df["AUQ191"])
    n_nonmissing = int(np.isfinite(t).sum())

    # --- (1) reproduce the cluster-based number (HDBSCAN canônico, mesmo config do contract test) ---
    labels = hdbscan.HDBSCAN(
        min_cluster_size=10, min_samples=5, metric="euclidean",
        cluster_selection_method="eom", core_dist_n_jobs=-1,
    ).fit_predict(emb.X_pca)
    sizes = {int(c): int((labels == c).sum()) for c in np.unique(labels[labels != -1])}
    minority_lbl = min(sizes, key=sizes.get)
    dominant_lbl = max(sizes, key=sizes.get)
    cluster_based = {
        "hdbscan_sizes": {str(k): v for k, v in sizes.items()},
        "minority": rate(labels == minority_lbl, t),
        "dominant": rate(labels == dominant_lbl, t),
        "noise": rate(labels == -1, t),
        "note": "Original v1-v4 claim was cluster-based; the cluster is hyperparameter-"
                "fragile (audit_06), so this is reported for provenance, not as the test.",
    }

    # --- defensible unit: |z| asymmetry tail (contrast = mean(R)-mean(L)) ---
    c_contrast = thr[R_COLS].mean(axis=1).to_numpy() - thr[L_COLS].mean(axis=1).to_numpy()
    z = np.abs(c_contrast) / np.nanstd(c_contrast)
    asym = {}
    for lv in Z_LEVELS:
        in_tail = z > lv
        asym[f"|z|>{lv:g}"] = {
            "n_in_tail": int(in_tail.sum()),
            **fisher_and_boot(in_tail, t, rng),
            "permutation_null": perm_null(in_tail, t, rng),
        }

    # --- severity gradient: does tinnitus track continuous high-freq severity? ---
    sev = None
    if "pta_high_mean_binaural" in df.columns:
        pta = pd.to_numeric(df["pta_high_mean_binaural"], errors="coerce").to_numpy()
        fin = np.isfinite(t) & np.isfinite(pta)
        pb = stats.pointbiserialr(t[fin], pta[fin])
        q = pd.qcut(pta[fin], 5, labels=False, duplicates="drop")
        tv = t[fin]
        by_quintile = [
            {"quintile": int(k), "n": int((q == k).sum()),
             "rate": float(tv[q == k].mean())}
            for k in sorted(np.unique(q))
        ]
        # Disentangle asymmetry from overall severity. The severity main-effect above is
        # NON-NOVEL (tinnitus~hearing loss is textbook) and must NOT carry the verdict.
        # The question the dropped claim was actually about: does the ASYMMETRY tail carry
        # elevated tinnitus AFTER adjusting for severity? Use severity QUINTILES (more
        # strata, better residual-confound control) and pool with Mantel-Haenszel.
        strata, mh_cells = [], []
        squint = pd.qcut(pta[fin], 5, labels=False, duplicates="drop")
        z_fin = z[fin]
        for k in sorted(np.unique(squint)):
            m = squint == k
            in_tail_s = (z_fin > 2.0) & m
            in_body_s = (z_fin <= 2.0) & m
            a = int(np.nansum(tv[in_tail_s])); b = int(in_tail_s.sum() - a)
            c = int(np.nansum(tv[in_body_s])); d = int(in_body_s.sum() - c)
            mh_cells.append((a, b, c, d))
            rt = (a / (a + b)) if (a + b) else None
            rb = (c / (c + d)) if (c + d) else None
            strata.append({
                "severity_quintile": int(k),
                "pta_high_range": [float(pta[fin][m].min()), float(pta[fin][m].max())],
                "tail_n": int(a + b), "tail_rate": rt,
                "body_n": int(c + d), "body_rate": rb,
                "ratio": (float(rt / rb) if (rt is not None and rb) else None),
            })
        mh = mantel_haenszel(mh_cells)
        sev = {
            "pointbiserial_tinnitus_vs_pta_high": {"r": float(pb.statistic), "p": float(pb.pvalue),
                "note": "non-novel main effect (tinnitus~hearing-loss severity); does NOT "
                        "establish the asymmetry-specific claim."},
            "rate_by_pta_high_quintile": by_quintile,
            "asymmetry_within_severity_quintiles": strata,
            "asymmetry_severity_adjusted_mantel_haenszel": mh,
        }

    # --- verdict (two effects kept SEPARATE) ---
    # (1) severity main-effect: real but NON-NOVEL; reported, never the headline.
    sev_sig = bool(sev and sev["pointbiserial_tinnitus_vs_pta_high"]["p"] < 0.05
                   and sev["pointbiserial_tinnitus_vs_pta_high"]["r"] > 0)
    # (2) asymmetry-SPECIFIC effect = the dropped claim, tested SEVERITY-ADJUSTED (MH).
    mh = (sev or {}).get("asymmetry_severity_adjusted_mantel_haenszel", {})
    asym_adjusted_sig = bool(
        mh.get("mh_or") and mh["mh_or"] > 1.0
        and mh.get("chi2_p") is not None and mh["chi2_p"] < 0.05
        and mh.get("ci95") and mh["ci95"][0] > 1.0)
    if n_nonmissing < 200:
        verdict = "data_insufficient"
    elif asym_adjusted_sig:
        # asymmetry survives adjustment -> reinstate, but as MODEST/severity-adjusted, not the 2.48
        verdict = "asymmetry_assoc_survives_severity_adjusted"
    elif sev_sig:
        # only the non-novel severity correlation survives; the asymmetry-specific claim does not
        verdict = "only_severity_gradient_survives_asymmetry_confounded"
    else:
        verdict = "does_not_survive"

    out = {
        "script": "34_tinnitus_reexamination.py",
        "purpose": "Re-examine the v1-v4 tinnitus '38% vs 18%' claim under current scrutiny; "
                   "defensible unit = |z| asymmetry tail + continuous severity, not the cluster.",
        "random_state": RANDOM_STATE,
        "cohort_n": int(len(thr)),
        "tinnitus_nonmissing_n": n_nonmissing,
        "tinnitus_overall_rate": float(np.nanmean(t)) if n_nonmissing else None,
        "cluster_based_reproduction": cluster_based,
        "asymmetry_tail_test": asym,
        "severity_gradient_test": sev,
        "verdict": verdict,
        "lib_versions": lib_versions(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print(f"cohort N={len(thr)}  tinnitus non-missing={n_nonmissing}  "
          f"overall rate={out['tinnitus_overall_rate']}")
    print(f"cluster-based: minority {cluster_based['minority']['rate']} "
          f"vs dominant {cluster_based['dominant']['rate']} (noise {cluster_based['noise']['rate']})")
    for k, v in asym.items():
        print(f"  {k}: tail rate={v['tail']['rate']} body rate={v['body']['rate']} "
              f"ratio={v['rate_ratio_tail_over_body']} fisher_p={v['fisher_p_greater']:.4g} "
              f"perm_p={v['permutation_null']['p_emp']}")
    if sev:
        pb = sev["pointbiserial_tinnitus_vs_pta_high"]
        mh = sev["asymmetry_severity_adjusted_mantel_haenszel"]
        print(f"  severity point-biserial r={pb['r']:.3f} p={pb['p']:.4g} (non-novel main effect)")
        print(f"  asymmetry severity-ADJUSTED (Mantel-Haenszel): OR={mh['mh_or']} "
              f"CI95={mh['ci95']} chi2_p={mh['chi2_p']}")
    print(f"VERDICT: {verdict}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
