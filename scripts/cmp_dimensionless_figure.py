#!/usr/bin/env python3
"""
Nome: cmp_dimensionless_figure.py
Tarefa: Materializar a figura adimensional (sem análise nova). Lê
        outputs/json/cmp_dimensionless_asymmetry.json e plota a razão real/cópula
        contra |z| para os três sistemas. Espelha o estilo de 27b_pooling_figure.py.
Output: outputs/dashboards/dimensionless_asymmetry_figure.png
"""

import json
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("outputs/json/cmp_dimensionless_asymmetry.json")
OUT = Path("outputs/dashboards/dimensionless_asymmetry_figure.png")

STYLE = {  # paleta tab10, igual às figuras do repo
    "audicao": ("Auditory (dB)", "#d62728", "o-"),
    "grip":    ("Grip (kg)",     "#1f77b4", "s-"),
    "visao":   ("Vision (D)",    "#2ca02c", "^-"),
}


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))["systems"]
    zlev = [2, 3, 4, 5]

    fig, ax = plt.subplots(figsize=(8.2, 5.6))

    # Regiões: <1.0 acoplamento / abaixo do null ; >1.0 excesso de cauda
    ax.axhspan(0.1, 1.0, color="#bbbbbb", alpha=0.20, zorder=0)
    ax.axhline(1.0, color="black", ls="--", lw=1.2, zorder=1)
    ax.text(5.05, 0.62, "coupling /\nbelow null", fontsize=9, color="#555555",
            ha="left", va="center", style="italic")
    ax.text(5.05, 6.0, "tail\nexcess", fontsize=9, color="#555555",
            ha="left", va="center", style="italic")

    for key, (label, color, marker) in STYLE.items():
        s = data[key]
        ratios = [s["ratio"][str(z)] for z in zlev]
        ax.plot(zlev, ratios, marker, color=color, lw=2.2, ms=8,
                markeredgecolor="black", markeredgewidth=0.5, label=label, zorder=5)
        # rótulo bruto real:cópula no ponto |z|>5
        r5, c5 = s["real"]["5"], s["copula"]["5"]
        ax.annotate(f"{r5}:{c5}", (5, ratios[-1]), textcoords="offset points",
                    xytext=(8, 4), fontsize=8.5, color=color, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(zlev)
    ax.set_xlabel("|z|  =  |R − L| / SD(R − L of real data)", fontsize=11)
    ax.set_ylabel("Real / Copula ratio  (log scale)", fontsize=11)
    ax.set_title("Dimensionless interaural-asymmetry tail vs. Gaussian-copula null\n"
                 "Auditory ≫ Grip > Vision; only vision falls below the null (central coupling)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(1.8, 5.6)
    ax.set_ylim(0.25, 80)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95, title="System (native unit)")
    ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Figura salva: {OUT}")


if __name__ == "__main__":
    main()
