"""
Nome: _nhanes_survey.py
Tarefa: Encanamento compartilhado de desenho amostral e de harmonização de itens
        AUQ entre as duas eras do questionário. Sem análise, sem desfecho: só
        definições, para que 16/23/33/34 usem exatamente a mesma regra.

  (1) Peso composto NCHS dos 5 ciclos com audiometria 20-69 anos
      (antes vivia dentro do 33; movido para cá sem mudar uma linha da regra):
        1999-2000, 2001-2002 -> WTSAU4YR * 2/5   (meia-amostra AUX, peso de 4 anos)
        2003-2004            -> WTSAU2YR * 1/5   (meia-amostra AUX)
        2011-2012, 2015-2016 -> WTMEC2YR * 1/5   (MEC completo)

  (2) Zumbido harmonizado entre eras:
        2005+     AUQ191 "zumbido incômodo, >= 5 min, últimos 12 meses" (1=sim, 2=não)
        1999-2004 AUQ190 "qualquer zumbido, últimos 12 meses" (1=sim, 2=não)
                  + AUQ200 frequência (1..4 = quase sempre / diário / semanal /
                    mensal; 5 = menos que mensal; 7/9 = recusa / não sabe)
      AUQ190 sozinho (~24%) NÃO é o mesmo construto que AUQ191 (~14.7%).
      Definição PRIMÁRIA (prevalência casada, lockbox_v1:
      primary_prevalence_matched_monthly): AUQ190=1 E AUQ200 em {1,2,3,4} -> 1
      (~14.2% em 1999-2004, contra 14.7% do AUQ191 em 2011-2016).
      SECUNDÁRIA (sensibilidade): AUQ200 em {1,2,3} (semanal ou mais, ~9.4%).
      Resposta 7/9, ou AUQ190=1 sem AUQ200 válido -> NaN (nunca vira "não").
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POOLING_SPEC_5CYCLES = {
    "1999-2000": ("WTSAU4YR", 2.0 / 5.0),
    "2001-2002": ("WTSAU4YR", 2.0 / 5.0),
    "2003-2004": ("WTSAU2YR", 1.0 / 5.0),
    "2011-2012": ("WTMEC2YR", 1.0 / 5.0),
    "2015-2016": ("WTMEC2YR", 1.0 / 5.0),
}
REQUIRED_WEIGHT_COLS = ("WTSAU4YR", "WTSAU2YR", "WTMEC2YR")

ERA_AUQ190 = ("1999-2000", "2001-2002", "2003-2004")
TINNITUS_DEFINITIONS = {
    # nome -> códigos de AUQ200 que contam como "sim" na era 1999-2004
    "monthly": (1, 2, 3, 4),   # primária (prevalência casada com AUQ191)
    "weekly": (1, 2, 3),       # secundária (sensibilidade)
    "any12m": None,            # terciária (lockbox: tertiary_loose_any12m) — AUQ190 sozinho
}


def build_5cycle_aux_weight(df: pd.DataFrame) -> np.ndarray:
    """Peso composto NCHS para os 5 ciclos de audiometria 20-69 anos.

    Levanta erro se aparecer ciclo fora da especificação (não pondera em silêncio).
    Peso ausente vira 0 (a pessoa não entra na estimativa ponderada)."""
    missing = [c for c in REQUIRED_WEIGHT_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"Colunas de peso ausentes: {missing} (rode 01/02/03 atualizados)")
    cycles = df["cycle"].astype(str)
    unknown = sorted(set(cycles.unique()) - set(POOLING_SPEC_5CYCLES))
    if unknown:
        raise ValueError(f"Ciclos fora da especificação 20-69 de 5 ciclos: {unknown}")
    w = np.zeros(len(df), dtype=np.float64)
    for cyc, (col, factor) in POOLING_SPEC_5CYCLES.items():
        mask = (cycles == cyc).to_numpy()
        if mask.any():
            vals = pd.to_numeric(df.loc[mask, col], errors="coerce").fillna(0.0).to_numpy()
            w[mask] = vals * factor
    return w


def tinnitus_from_auq191(series: pd.Series) -> np.ndarray:
    """AUQ191: 1=sim, 2=não. Qualquer outro valor -> NaN."""
    auq = pd.to_numeric(series, errors="coerce")
    return np.select([auq == 1, auq == 2], [1.0, 0.0], default=np.nan).astype("float64")


def tinnitus_harmonized(df: pd.DataFrame, definition: str = "monthly") -> np.ndarray:
    """Zumbido harmonizado entre eras (ver docstring do módulo).

    A era é decidida pelo CICLO, nunca pela presença da coluna: uma pessoa de
    1999-2004 com AUQ191 preenchido (não deveria existir) ou de 2011+ com AUQ190
    preenchido seria erro de ingestão e levanta exceção."""
    if definition not in TINNITUS_DEFINITIONS:
        raise ValueError(f"definition deve ser uma de {sorted(TINNITUS_DEFINITIONS)}")
    yes_codes = TINNITUS_DEFINITIONS[definition]
    cycles = df["cycle"].astype(str).to_numpy()
    old = np.isin(cycles, ERA_AUQ190)

    a191 = pd.to_numeric(df.get("AUQ191"), errors="coerce") if "AUQ191" in df else pd.Series(np.nan, index=df.index)
    a190 = pd.to_numeric(df.get("AUQ190"), errors="coerce") if "AUQ190" in df else pd.Series(np.nan, index=df.index)
    a200 = pd.to_numeric(df.get("AUQ200"), errors="coerce") if "AUQ200" in df else pd.Series(np.nan, index=df.index)

    if (old & a191.notna().to_numpy()).any():
        raise ValueError("AUQ191 preenchido em ciclo 1999-2004: erro de ingestão")
    if ((~old) & (a190.notna() | a200.notna()).to_numpy()).any():
        raise ValueError("AUQ190/AUQ200 preenchido fora de 1999-2004: erro de ingestão")

    out = tinnitus_from_auq191(a191)
    a190v, a200v = a190.to_numpy(), a200.to_numpy()
    old_val = np.full(len(df), np.nan)
    old_val[a190v == 2] = 0.0
    any_yes = a190v == 1
    if yes_codes is None:  # any12m: qualquer episódio em 12 meses, sem olhar frequência
        old_val[any_yes] = 1.0
    else:
        old_val[any_yes & np.isin(a200v, yes_codes)] = 1.0
        old_val[any_yes & np.isin(a200v, [c for c in (1, 2, 3, 4, 5) if c not in yes_codes])] = 0.0
        # AUQ190=1 com AUQ200 em {7,9} ou ausente permanece NaN
    out[old] = old_val[old]
    return out


def weighted_rate(y: np.ndarray, w: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Proporção ponderada entre não-ausentes com peso > 0 (estimativa pontual;
    variância de desenho fica para quem precisar de IC — aqui é descritivo)."""
    sel = np.isfinite(y) & (w > 0)
    if mask is not None:
        sel &= mask
    n = int(sel.sum())
    if n == 0:
        return {"n": 0, "rate_weighted": None, "rate_unweighted": None}
    return {
        "n": n,
        "rate_weighted": float(np.sum(w[sel] * y[sel]) / np.sum(w[sel])),
        "rate_unweighted": float(np.mean(y[sel])),
    }
