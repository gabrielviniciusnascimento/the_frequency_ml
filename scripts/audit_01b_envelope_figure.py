#!/usr/bin/env python3
"""
audit_01b_envelope_figure.py  (HANDOFF Task 1 demonstration figure)

Per-system histogram of the null tail-count distribution at |z|>4 (B=2000 copula
regenerations) with a vertical line at the real count. Reads the saved null arrays
from audit_01_mc_envelope.json. Visualizes that the auditory (and grip) real count
sits far outside the null, while vision's real count sits inside it.

Output: outputs/dashboards/audit_envelope_figure.png
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs/json/audit_01_mc_envelope.json"
OUT = ROOT / "outputs/dashboards/audit_envelope_figure.png"

d = json.loads(SRC.read_text(encoding="utf-8"))
sys = d["systems"]
order = [("audicao", "Auditory"), ("grip", "Grip"), ("visao", "Vision")]

fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
for ax, (key, title) in zip(axes, order):
    s = sys[key]
    null = np.array(s["null_dist_z_gt4"])
    real = s["z_cells"]["4"]["real"]
    p = s["z_cells"]["4"]["p_emp"]
    hi = max(null.max(), real)
    bins = np.arange(0, hi + 2) - 0.5
    ax.hist(null, bins=bins, color="#9bbbd6", edgecolor="white", linewidth=0.3,
            label=f"null (B={d['B']})")
    ax.axvline(real, color="#c0392b", lw=2.2, label=f"real = {real}")
    ax.set_title(f"{title}  (|z|>4)\nnull mean {null.mean():.1f}, real {real}, p={p}")
    ax.set_xlabel("tail count beyond |z|>4")
    ax.set_ylabel("null draws")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Monte-Carlo null envelope vs. real inter-side contrast tail (NHANES paired organs)",
             fontsize=12, y=1.02)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"Output: {OUT}")
