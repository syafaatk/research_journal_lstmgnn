# -*- coding: utf-8 -*-
"""Generate figures for paper #10 (intermittent demand at product level).

Produces (folder figures/paper10_intermittent/):
- fig1_purchase_frequency.png : purchase frequency distribution (long tail)
- fig2_sb_classification.png  : Syntetos-Boylan classification
- fig3_value_concentration.png: value concentration (Pareto-style)
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:\Download\Jurnal"
FIGDIR = os.path.join(ROOT, "figures", "paper10_intermittent")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_excel(os.path.join(ROOT, "data", "view_penjualan_detail data hingga oktober.xlsx"))
df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
df = df.dropna(subset=["tanggal"])
df["prod"] = df["barang_nama"].str.strip() + " | " + df["barang_spesifikasi"].str.strip()
df["ym"] = df["tanggal"].dt.to_period("M")

prod_value = df.groupby("prod")["jumlah"].sum().sort_values(ascending=False)
prod_months = df.groupby("prod")["ym"].nunique()
inv_prod = df.groupby(["d_jual_nofak", "prod"]).size().reset_index()
prod_inv = inv_prod.groupby("prod")["d_jual_nofak"].nunique()

# ---------------------------------------------------------------------------
# 1. Purchase frequency distribution (long tail)
# ---------------------------------------------------------------------------
bins = ["1", "2-5", "6-20", ">20"]
counts = [
    int((prod_inv == 1).sum()),
    int(((prod_inv >= 2) & (prod_inv <= 5)).sum()),
    int(((prod_inv >= 6) & (prod_inv <= 20)).sum()),
    int((prod_inv > 20).sum()),
]
tot = len(prod_inv)

fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(bins, counts, color="#1a73e8", edgecolor="black", linewidth=0.6)
for b, v in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, v + 15, f"{v}\n({v/tot*100:.1f}%)",
            ha="center", fontsize=9)
ax.set_xlabel("Invoices per product")
ax.set_ylabel("Number of products")
ax.set_title("Purchase Frequency Distribution (2,380 products)\n"
             "Long tail: 63.4% of products bought 5 or fewer times")
ax.set_ylim(0, max(counts) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig1_purchase_frequency.png"))
plt.close(fig)
print("Saved fig1_purchase_frequency.png")

# ---------------------------------------------------------------------------
# 2. Syntetos-Boylan classification
# ---------------------------------------------------------------------------
all_months = pd.period_range("2020-07", "2025-10", freq="M")
n_months = len(all_months)
pm = df.groupby(["prod", "ym"])["jumlah"].sum().reset_index()

rows = []
for prod, g in pm.groupby("prod"):
    n_active = len(g)
    if n_active >= 6:
        nz = g["jumlah"].values
        adi = n_months / n_active
        cv2 = (nz.std() / nz.mean()) ** 2 if nz.mean() > 0 else np.nan
        rows.append((prod, n_active, adi, cv2, nz.sum()))
rdf = pd.DataFrame(rows, columns=["prod", "n_active", "ADI", "CV2", "value"])
n_class = len(rdf)

classes = ["Smooth", "Erratic", "Intermittent", "Lumpy"]
masks = [
    (rdf["ADI"] < 1.32) & (rdf["CV2"] < 0.49),
    (rdf["ADI"] < 1.32) & (rdf["CV2"] >= 0.49),
    (rdf["ADI"] >= 1.32) & (rdf["CV2"] < 0.49),
    (rdf["ADI"] >= 1.32) & (rdf["CV2"] >= 0.49),
]
counts = [int(mask.sum()) for mask in masks]
colors = ["#188038", "#f9ab00", "#1a73e8", "#d93025"]

fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(classes, counts, color=colors, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, counts):
    ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v}\n({v/n_class*100:.1f}%)",
            ha="center", fontsize=9)
ax.set_xlabel("Syntetos-Boylan class")
ax.set_ylabel("Number of products")
ax.set_title("Syntetos-Boylan Classification (770 products with >= 6 active months)\n"
             "92.7% of products are intermittent or lumpy")
ax.set_ylim(0, max(counts) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig2_sb_classification.png"))
plt.close(fig)
print("Saved fig2_sb_classification.png")

# ---------------------------------------------------------------------------
# 3. Value concentration (Pareto-style cumulative)
# ---------------------------------------------------------------------------
cum = prod_value.cumsum() / prod_value.sum() * 100
x = np.arange(1, len(cum) + 1)

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.plot(x, cum.values, color="#d93025", linewidth=2)
ax.axhline(71.9, color="#188038", ls="--", lw=1.2)
ax.axvline(419, color="#188038", ls="--", lw=1.2)
ax.text(430, 74, "419 products (17.6%)\n= 71.9% of value", fontsize=9, color="#188038")
ax.set_xlabel("Products (sorted by value, descending)")
ax.set_ylabel("Cumulative share of value (%)")
ax.set_title("Value Concentration: Long Tail of Rarely Bought Products\n"
             "Value is concentrated in a small set of frequently bought products")
ax.set_xlim(0, len(cum))
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig3_value_concentration.png"))
plt.close(fig)
print("Saved fig3_value_concentration.png")
