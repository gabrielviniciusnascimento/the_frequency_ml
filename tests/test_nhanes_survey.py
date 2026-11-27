#!/usr/bin/env python3
"""
Testes do encanamento compartilhado (scripts/_nhanes_survey.py) e do caminho
--unblind-harmonized do 34, só com dado sintético: nenhum teste lê o zumbido real
de 1999-2004 cruzado com audiograma.
Segue o padrão plain-assert do repositório (sem dependência obrigatória de pytest).
Run: python tests/test_nhanes_survey.py
"""
from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from _nhanes_survey import (  # noqa: E402
    build_5cycle_aux_weight, tinnitus_harmonized, tinnitus_from_auq191, weighted_rate,
)

NaN = np.nan


def _old_era_frame() -> pd.DataFrame:
    # cada linha: (AUQ190, AUQ200) -> esperado (monthly, weekly, any12m)
    rows = [
        (2, NaN, 0, 0, 0),      # nunca teve zumbido
        (1, 1, 1, 1, 1),        # quase sempre
        (1, 3, 1, 1, 1),        # semanal
        (1, 4, 1, 0, 1),        # mensal: sim na primária, não na secundária
        (1, 5, 0, 0, 1),        # menos que mensal
        (1, 7, NaN, NaN, 1),    # recusa a frequência
        (1, 9, NaN, NaN, 1),    # não sabe a frequência
        (1, NaN, NaN, NaN, 1),  # sim sem frequência
        (9, NaN, NaN, NaN, NaN),
        (NaN, NaN, NaN, NaN, NaN),
    ]
    df = pd.DataFrame(rows, columns=["AUQ190", "AUQ200", "m", "w", "a"])
    df["cycle"] = "2001-2002"
    df["AUQ191"] = NaN
    return df


def _eq(a: np.ndarray, b) -> bool:
    b = np.asarray(b, dtype=float)
    return bool(np.all((np.isnan(a) & np.isnan(b)) | (a == b)))


def test_harmonization_matches_lockbox() -> None:
    df = _old_era_frame()
    assert _eq(tinnitus_harmonized(df, "monthly"), df["m"]), "primária (mensal+) diverge do lockbox"
    assert _eq(tinnitus_harmonized(df, "weekly"), df["w"]), "secundária (semanal+) diverge do lockbox"
    assert _eq(tinnitus_harmonized(df, "any12m"), df["a"]), "terciária (any12m) diverge do lockbox"


def test_new_era_uses_auq191_in_all_definitions() -> None:
    df = pd.DataFrame({"cycle": ["2011-2012"] * 4, "AUQ191": [1, 2, 7, 9],
                       "AUQ190": NaN, "AUQ200": NaN})
    for d in ("monthly", "weekly", "any12m"):
        assert _eq(tinnitus_harmonized(df, d), [1, 0, NaN, NaN])
    assert _eq(tinnitus_from_auq191(df["AUQ191"]), [1, 0, NaN, NaN])


def test_era_guards_raise() -> None:
    bad_old = pd.DataFrame({"cycle": ["1999-2000"], "AUQ191": [1], "AUQ190": [NaN], "AUQ200": [NaN]})
    bad_new = pd.DataFrame({"cycle": ["2015-2016"], "AUQ191": [NaN], "AUQ190": [1], "AUQ200": [2]})
    for bad in (bad_old, bad_new):
        try:
            tinnitus_harmonized(bad)
        except ValueError:
            continue
        raise AssertionError("guarda de era não disparou")


def test_weights_rule_and_unknown_cycle() -> None:
    df = pd.DataFrame({
        "cycle": ["1999-2000", "2001-2002", "2003-2004", "2011-2012", "2015-2016"],
        "WTSAU4YR": [10.0, 20.0, NaN, NaN, NaN],
        "WTSAU2YR": [NaN, NaN, 30.0, NaN, NaN],
        "WTMEC2YR": [99.0, 99.0, 99.0, 40.0, 50.0],
    })
    w = build_5cycle_aux_weight(df)
    assert np.allclose(w, [4.0, 8.0, 6.0, 8.0, 10.0]), w
    df.loc[0, "cycle"] = "2005-2006"
    try:
        build_5cycle_aux_weight(df)
    except ValueError:
        pass
    else:
        raise AssertionError("ciclo fora da especificação deveria levantar erro")


def test_weighted_rate_skips_zero_weight_and_nan() -> None:
    y = np.array([1.0, 0.0, NaN, 1.0])
    w = np.array([3.0, 1.0, 5.0, 0.0])
    r = weighted_rate(y, w)
    assert r["n"] == 2 and abs(r["rate_weighted"] - 0.75) < 1e-12 and r["rate_unweighted"] == 0.5


def _synthetic_cohort(n: int = 1500, seed: int = 0):
    rng = np.random.default_rng(seed)
    cycles = rng.choice(["1999-2000", "2001-2002", "2003-2004", "2011-2012", "2015-2016"], n)
    old = np.isin(cycles, ["1999-2000", "2001-2002", "2003-2004"])
    freqs = [500, 1000, 2000, 3000, 4000, 6000, 8000]
    base = rng.normal(30, 10, (n, 1)) + np.linspace(0, 20, 7)
    thr = pd.DataFrame(
        np.hstack([base + rng.normal(0, 5, (n, 7)), base + rng.normal(0, 5, (n, 7))]),
        columns=[f"thr_R_{f}" for f in freqs] + [f"thr_L_{f}" for f in freqs])
    df = thr.copy()
    df["cycle"] = cycles
    df["pta_high_mean_binaural"] = thr.filter(regex="_(4000|6000|8000)$").mean(axis=1)
    df["AUQ191"] = np.where(old, NaN, rng.choice([1, 2], n, p=[0.2, 0.8]))
    a190 = rng.choice([1, 2], n, p=[0.25, 0.75])
    df["AUQ190"] = np.where(old, a190, NaN)
    df["AUQ200"] = np.where(old & (a190 == 1), rng.choice([1, 2, 3, 4, 5, 9], n), NaN)
    df["WTSAU4YR"] = rng.uniform(1e3, 5e4, n)
    df["WTSAU2YR"] = rng.uniform(1e3, 5e4, n)
    df["WTMEC2YR"] = rng.uniform(1e3, 5e4, n)
    return df, thr


def test_34_unblind_path_runs_on_synthetic() -> None:
    m = importlib.import_module("34_tinnitus_reexamination")
    df, thr = _synthetic_cohort()
    with tempfile.TemporaryDirectory() as tmp:
        m.load_cohort = lambda: (df, thr)
        m.OUT = Path(tmp) / "34.json"
        m.B_BOOT = m.B_PERM = 50
        for argv, crossed in ((["34"], False), (["34", "--unblind-harmonized"], True)):
            sys.argv = argv
            m.main()
            out = json.loads(m.OUT.read_text(encoding="utf-8"))
            h = out["harmonized_tinnitus_1999_2004"]
            assert h["unblinded"] is crossed and ("crossed" in h) is crossed
            if crossed:
                pooled = h["crossed"]["monthly"]["pooled_5cycles_era_x_quintile"]
                eras = {s["auq_era"] for s in pooled["severity_gradient_test"]["asymmetry_within_severity_quintiles"]}
                assert eras == {"1999-2004", "2011-2016"}, eras
                only = h["crossed"]["monthly"]["era_1999_2004_only"]
                assert only["tinnitus_nonmissing_n"] < pooled["tinnitus_nonmissing_n"]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)}/{len(tests)} passaram")
