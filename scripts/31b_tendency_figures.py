#!/usr/bin/env python3
"""
Nome: 31b_tendency_figures.py
Tarefa: Figura demonstrativa da bateria de tendência de cluster (script 31).
        Três painéis, lendo os arrays salvos em 31_cluster_tendency.json:
          (a) OPTICS reachability ordenado — parede lisa = sem vales de densidade.
          (b) Dendrograma Ward (truncado) — sem salto isolado além da raiz = sem k natural.
          (c) Histograma do eixo bruto de assimetria PTA_R−PTA_L com o dip p — para
              o leitor VER se a rejeição de unimodalidade é um pente de quantização
              (5 dB) + cauda contínua, e não um segundo modo bem separado.

Padrão: igual a audit_01b_envelope_figure.py (lê do JSON, matplotlib Agg).
Output: outputs/dashboards/cluster_tendency_figure.png
Dependências: outputs/json/31_cluster_tendency.json
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "json" / "31_cluster_tendency.json"
OUT = ROOT / "outputs" / "dashboards" / "cluster_tendency_figure.png"

d = json.loads(SRC.read_text(encoding="utf-8"))
arr = d["figure_arrays"]
optics, hier, dip = d["optics"], d["hierarchy"], d["dip_test"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# ── (a) OPTICS reachability ───────────────────────────────────────────
reach = np.array([np.nan if v is None else v for v in arr["reachability_ordered"]], dtype=float)
finite = reach[np.isfinite(reach)]
ax = axes[0]
ax.fill_between(np.arange(reach.size), reach, color="#9bbbd6", step="mid", linewidth=0)
ax.axhline(np.nanmean(finite), color="#c0392b", lw=1.3, ls="--",
           label=f"média {np.nanmean(finite):.2f}")
ax.set_title(f"(a) OPTICS reachability\n{optics['n_clusters']} cluster por xi, "
             f"{optics['noise_fraction']*100:.0f}% ruído — sem vales")
ax.set_xlabel("ordem OPTICS"); ax.set_ylabel("distância de reachability")
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)

# ── (b) Dendrograma Ward ──────────────────────────────────────────────
Z = np.array(arr["ward_linkage_Z"], dtype=float)
ax = axes[1]
dendrogram(Z, ax=ax, no_labels=True, truncate_mode="lastp", p=40, color_threshold=0,
           above_threshold_color="#4e79a7")
ratio = hier["second_over_first_gap_ratio"]
ax.set_title(f"(b) Dendrograma Ward (n={hier['sample_size']})\n"
             f"cophenetic={hier['cophenetic_corr']}, 2º/1º gap={ratio} — sem 2º k natural")
ax.set_xlabel("amostras (truncado)"); ax.set_ylabel("altura de merge")
ax.spines[["top", "right"]].set_visible(False)

# ── (c) Histograma do eixo bruto de assimetria R−L ────────────────────
asym = np.array(arr["asymmetry_R_minus_L_db"], dtype=float)
a_dip = dip["asymmetry_R_minus_L_db"]
ax = axes[2]
ax.hist(asym, bins=120, color="#b0b7c0", edgecolor="white", linewidth=0.2)
ax.axvline(0, color="#444", lw=1.0, ls=":")
ax.set_yscale("log")
ax.set_title(f"(c) Eixo bruto PTA_R−PTA_L\ndip={a_dip['dip']}, p={a_dip['p_value']} "
             "(efeito mínimo; cauda contínua)")
ax.set_xlabel("PTA direito − esquerdo (dB)"); ax.set_ylabel("contagem (log)")
ax.spines[["top", "right"]].set_visible(False)

pc1, pc2 = dip["PC1"], dip["PC2"]
fig.suptitle(
    "Tendência de cluster no espaço de forma (NHANES, N=%d): nenhuma família sustenta subtipos discretos.  "
    "Eixos de clustering unimodais — dip PC1 p=%s, PC2 p=%s." % (
        d["n_samples"], pc1["p_value"], pc2["p_value"]),
    fontsize=11, y=1.04)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Output: {OUT}")
