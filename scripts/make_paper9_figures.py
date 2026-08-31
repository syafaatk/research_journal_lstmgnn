# -*- coding: utf-8 -*-
"""Generate figures for paper #9 (static node attributes in GNN).

Produces (folder figures/paper9_nodeattr/):
- fig1_nodeattr_effect.png : effect of hospital-count node attribute (base/tuned)
- fig2_hospitals_per_region.png : hospital count per region (proxy for capacity)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:\Download\Jurnal"
FIGDIR = os.path.join(ROOT, "figures", "paper9_nodeattr")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

REGIONS = [
    "Banyuasin", "Empat Lawang", "Lahat", "Lubuk Linggau", "Muara Enim",
    "Musi Banyuasin", "Musi Rawas", "Ogan Ilir",
    "Ogan Komering Ilir", "Ogan Komering Ulu", "Ogan Komering Ulu Selatan",
    "Ogan Komering Ulu Timur", "Pagar Alam", "Palembang",
    "Penukal Abab Lematang Ilir", "Prabumulih",
]

# ---------------------------------------------------------------------------
# 1. Effect of hospital-count node attribute (from manuscript Table 1)
# ---------------------------------------------------------------------------
configs = ["Base\n(128/128/do0)", "Tuned\n(128/64/do0.2)"]
ref = [0.0592, 0.0508]
with_attr = [0.0609, 0.0528]
dm = [-2.861, -6.037]
p = [0.004, "< 0.0001"]

fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(configs))
width = 0.35
bars1 = ax.bar(x - width / 2, ref, width, color="#9aa7b0",
               edgecolor="black", linewidth=0.6, label="Reference (no node attribute)")
bars2 = ax.bar(x + width / 2, with_attr, width, color="#188038",
               edgecolor="black", linewidth=0.6, label="With hospital-count node attribute")
for i in range(len(configs)):
    ax.text(i - width / 2, ref[i] + 0.0005, f"{ref[i]:.4f}", ha="center", fontsize=9)
    ax.text(i + width / 2, with_attr[i] + 0.0005, f"{with_attr[i]:.4f}", ha="center", fontsize=9)
    ax.text(i, 0.054, f"DM = {dm[i]}\np = {p[i]}", ha="center", fontsize=8.5,
            bbox=dict(fc="white", ec="none", alpha=0.8))
ax.set_xticks(x)
ax.set_xticklabels(configs)
ax.set_ylabel("R-squared")
ax.set_title("Effect of Hospital-Count Node Attribute on GCN Path\n"
             "(significant improvement in both configurations)")
ax.set_ylim(0.048, 0.063)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig1_nodeattr_effect.png"))
plt.close(fig)
print("Saved fig1_nodeattr_effect.png")

# ---------------------------------------------------------------------------
# 2. Hospital count per region (proxy for healthcare capacity)
# ---------------------------------------------------------------------------
rs = pd.read_csv(os.path.join(ROOT, "data", "sirs_sumsel_geocoded_v2.csv"))
rs["region"] = rs["kabkota"].replace({
    "Kota Palembang": "Palembang", "Kota Lubuk Linggau": "Lubuk Linggau",
    "Kota Prabumulih": "Prabumulih", "Kota Pagar Alam": "Pagar Alam",
    "Musi Rawas Utara": "Musi Rawas",
})
hosp = rs.groupby("region").size().reindex(REGIONS).fillna(0).astype(int).sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#d93025" if r == "Palembang" else "#1a73e8" for r in hosp.index]
bars = ax.barh(hosp.index, hosp.values, color=colors, edgecolor="black", linewidth=0.5)
for i, v in enumerate(hosp.values):
    ax.text(v + 0.2, i, str(v), va="center", fontsize=9)
ax.set_xlabel("Number of hospitals")
ax.set_title("Hospital Count per Region (2024)\n"
             "Palembang: 33 hospitals (44% of province total); others 1-8")
ax.set_xlim(0, 38)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig2_hospitals_per_region.png"))
plt.close(fig)
print("Saved fig2_hospitals_per_region.png")
