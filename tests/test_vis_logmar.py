

#!/usr/bin/env python3
"""
Testes unitários isolados para scrub_ibm_underflow e snellen_code_to_logmar (scripts/vis_01_ingest.py).
Segue o padrão plain-assert do repositório (sem dependência obrigatória de pytest).
Run: python tests/test_vis_logmar.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vis_01_ingest import scrub_ibm_underflow, snellen_code_to_logmar  # noqa: E402


def test_ibm_underflow_scrub() -> None:
    df = pd.DataFrame({
        "SEQN": [1, 2],
        "OD_cyl": [5.397605346934028e-79, -1.25],
        "OS_sphere": [-5.397605346934028e-79, 0.25],
    })
    cleaned = scrub_ibm_underflow(df)
    assert cleaned.loc[0, "OD_cyl"] == 0.0, f"Underflow positivo sobreviveu: {cleaned.loc[0, 'OD_cyl']}"
    assert cleaned.loc[0, "OS_sphere"] == 0.0, f"Underflow negativo sobreviveu: {cleaned.loc[0, 'OS_sphere']}"
    assert cleaned.loc[1, "OD_cyl"] == -1.25
    assert cleaned.loc[1, "OS_sphere"] == 0.25
    print("OK test_ibm_underflow_scrub")


def test_presenting_va_conversion() -> None:
    s = pd.Series([20, 25, 30, 40, 50, 60, 80, 200, 666, 888, np.nan])
    logmar, f666, f888 = snellen_code_to_logmar(s, allow_20=True, code_666_logmar=None, strict=True)

    expected = [
        0.0,
        math.log10(25 / 20),
        math.log10(30 / 20),
        math.log10(40 / 20),
        math.log10(50 / 20),
        math.log10(60 / 20),
        math.log10(80 / 20),
        1.0,
    ]
    for idx, exp_val in enumerate(expected):
        assert abs(float(logmar.iloc[idx]) - exp_val) < 1e-12, f"Mismatch idx={idx}: {logmar.iloc[idx]} != {exp_val}"

    assert pd.isna(logmar.iloc[8]) and int(f666.iloc[8]) == 1 and int(f888.iloc[8]) == 0
    assert pd.isna(logmar.iloc[9]) and int(f666.iloc[9]) == 0 and int(f888.iloc[9]) == 1
    assert pd.isna(logmar.iloc[10]) and int(f666.iloc[10]) == 0 and int(f888.iloc[10]) == 0
    print("OK test_presenting_va_conversion")


def test_code_666_optional_cap_and_strict_guards() -> None:
    s = pd.Series([25, 200, 666])
    cap_val = math.log10(400 / 20)  # 20/400 (~1.30103 logMAR) p/ análise de sensibilidade de censura
    logmar, f666, _ = snellen_code_to_logmar(s, allow_20=False, code_666_logmar=cap_val, strict=True)
    assert abs(float(logmar.iloc[2]) - cap_val) < 1e-12
    assert int(f666.iloc[2]) == 1

    # Código 20 não existe em VIDROVA/VIDLOVA (só testados se acuidade apresentada >= 30)
    try:
        snellen_code_to_logmar(pd.Series([20, 25]), allow_20=False, strict=True)
        raise AssertionError("Deveria rejeitar código 20 quando allow_20=False")
    except ValueError:
        pass

    # Código inválido fora do codebook (ex.: 70 ou 100 não existem na tabela VIX 1999-2004)
    try:
        snellen_code_to_logmar(pd.Series([20, 100]), allow_20=True, strict=True)
        raise AssertionError("Deveria rejeitar denominador 100 inexistente em VIX 1999-2004")
    except ValueError:
        pass

    print("OK test_code_666_optional_cap_and_strict_guards")


if __name__ == "__main__":
    test_ibm_underflow_scrub()
    test_presenting_va_conversion()
    test_code_666_optional_cap_and_strict_guards()
    print("TODOS OS TESTES DE ACUIDADE / LOGMAR PASSARAM.")

