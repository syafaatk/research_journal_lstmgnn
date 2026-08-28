# -*- coding: utf-8 -*-
"""Generate figures for paper #1 (graph construction in GNN forecasting).

Produces:
- figures/paper1_ablation_bar.png : bar chart of R2 across graph constructions
- figures/paper1_star_graph.png   : star graph visualization (Palembang hub)
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:\Download\Jurnal"
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Ablation bar chart
# ---------------------------------------------------------------------------
variants = [
    ("Identity adjacency", 0.0538),
    ("Random graph\n(mean of 5)", 0.0486),
    ("Distance graph\nk=3", 0.0436),
    ("Correlation graph\nk=3", 0.0501),
    ("Distribution network\n(star from Palembang)", 0.0703),
    ("Without\nCOVID-19", 0.0425),
]
labels = [v[0] for v in variants]
r2 = [v[1] for v in variants]
colors = ["#4C72B0"] * len(variants)
colors[4] = "#C44E52"  # highlight the star

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(labels, r2, color=colors, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, r2):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0015, f"{v:.4f}",
            ha="center", va="bottom", fontsize=9)
ax.axhline(0.0, color="black", linewidth=0.8)
ax.set_ylabel("R-squared")
ax.set_title("Ablation Results Across Graph Constructions\n"
             "(HybridTuned: LSTM 128, GCN 64, dropout 0.2, seed 42)")
ax.set_ylim(0.03, 0.08)
ax.tick_params(axis="x", labelsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "paper1_ablation_bar.png"), dpi=300)
plt.close(fig)
print("Saved paper1_ablation_bar.png")

# ---------------------------------------------------------------------------
# 2. Star graph visualization
# ---------------------------------------------------------------------------
# Load customer coordinates, compute invoice-weighted centroid per region.
cust = pd.read_csv(os.path.join(ROOT, "data", "customers_geocoded_final.csv"))
cust = cust.dropna(subset=["lat", "lon"])

# Invoice-weighted centroid per region
cust["w"] = cust["n_inv"].fillna(1)
cent = cust.groupby("region").apply(
    lambda g: pd.Series({
        "lat": np.average(g["lat"], weights=g["w"]),
        "lon": np.average(g["lon"], weights=g["w"]),
    }), include_groups=False
).reset_index()

# Hub = Palembang
hub = cent[cent["region"] == "Palembang"].iloc[0]
others = cent[cent["region"] != "Palembang"]

fig, ax = plt.subplots(figsize=(9, 8))
# Draw edges from hub to all others
for _, r in others.iterrows():
    ax.plot([hub["lon"], r["lon"]], [hub["lat"], r["lat"]],
            color="#4C72B0", linewidth=1.2, alpha=0.7, zorder=1)
# Hub node
ax.scatter(hub["lon"], hub["lat"], s=900, color="#C44E52",
           edgecolor="black", zorder=3, label="Palembang (hub)")
ax.text(hub["lon"], hub["lat"], "Palembang", ha="center", va="center",
        fontsize=9, color="white", fontweight="bold", zorder=4)
# Other nodes
ax.scatter(others["lon"], others["lat"], s=220, color="#4C72B0",
           edgecolor="black", zorder=3)
for _, r in others.iterrows():
    ax.annotate(r["region"], (r["lon"], r["lat"]),
                textcoords="offset points", xytext=(6, 6), fontsize=7)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Distribution Network (Star from Palembang)\n"
             "All delivery routes originate from the Palembang hub")
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "paper1_star_graph.png"), dpi=300)
plt.close(fig)
print("Saved paper1_star_graph.png")
