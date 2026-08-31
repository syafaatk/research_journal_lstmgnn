# -*- coding: utf-8 -*-
"""Generate figures for paper #2 (aggregation destroys the signal).

Produces (folder figures/paper2_aggregation/):
- fig1_resolution_r2.png   : R2 by temporal resolution (monthly/weekly/daily)
- fig2_baselines.png       : baselines vs always-zero (ARIMA, XGBoost)
- fig3_zero_fraction.png   : zero-sales fraction per region
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:\Download\Jurnal"
FIGDIR = os.path.join(ROOT, "figures", "paper2_aggregation")
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
CITY_MAP = {
    "Palembang": "Palembang", "Lahat": "Lahat", "Baturaja": "Ogan Komering Ulu",
    "Oku (Ogan Komering Ulu)": "Ogan Komering Ulu Timur", "Musi Banyuasin": "Musi Banyuasin",
    "Muara Enim": "Muara Enim", "Tanjung Enim": "Muara Enim", "Muara Lawai": "Muara Enim",
    "Lubuklinggau": "Lubuk Linggau", "Ogan Ilir": "Ogan Ilir", "Oku Selatan": "Ogan Komering Ulu Selatan",
    "Prabumulih": "Prabumulih", "Prabumulih Timur": "Prabumulih",
    "Pali (Penukal Abab Lematang Ilir)": "Penukal Abab Lematang Ilir", "Talang Ubi": "Penukal Abab Lematang Ilir",
    "Musi Rawas": "Musi Rawas", "Oku Timur": "Ogan Komering Ulu Timur", "Martapura": "Ogan Komering Ulu Timur",
    "Empat Lawang": "Empat Lawang", "Banyuasin": "Banyuasin", "Pagaralam": "Pagar Alam",
    "Muara Rupit": "Musi Rawas", "Rupit": "Musi Rawas",
    "Oki (Ogan Komering Ilir)": "Ogan Komering Ilir",
}

# ---------------------------------------------------------------------------
# 1. Resolution R2 bar chart (from manuscript verified numbers)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
res = ["Monthly", "Weekly", "Daily\n(zero-inflated)"]
r2 = [-0.394, -0.343, 0.047]  # daily midpoint of +0.043 to +0.052
colors = ["#d93025", "#d93025", "#188038"]
bars = ax.bar(res, r2, color=colors, edgecolor="black", linewidth=0.6, width=0.55)
for b, v in zip(bars, r2):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.01 if v >= 0 else -0.01),
            f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("R-squared")
ax.set_title("R-squared by Temporal Resolution\n(monthly and weekly aggregation destroy the signal)")
ax.set_ylim(-0.45, 0.10)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig1_resolution_r2.png"))
plt.close(fig)
print("Saved fig1_resolution_r2.png")

# ---------------------------------------------------------------------------
# 2. Baselines comparison
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
base = ["Always-zero", "ARIMA", "XGBoost"]
vals = [0.0059, -0.0775, -0.0840]
colors = ["#188038", "#d93025", "#d93025"]
bars = ax.bar(base, vals, color=colors, edgecolor="black", linewidth=0.6, width=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.002 if v >= 0 else -0.002),
            f"{v:.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=10)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("R-squared")
ax.set_title("Baselines on Zero-Inflated Regional Demand\n(ARIMA and XGBoost fail to beat always-zero)")
ax.set_ylim(-0.10, 0.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig2_baselines.png"))
plt.close(fig)
print("Saved fig2_baselines.png")

# ---------------------------------------------------------------------------
# 3. Zero fraction per region (from sales data)
# ---------------------------------------------------------------------------
df = pd.read_excel(os.path.join(ROOT, "data", "view_penjualan_detail data hingga oktober.xlsx"))
df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
df = df.drop_duplicates("d_jual_nofak")
df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                        "jual_total_fak": "jual_total"})
df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
df["region"] = df["pelanggan_kota"].map(CITY_MAP)
df["period"] = df["jual_tanggal"].dt.to_period("D")

panel = df.pivot_table(index="period", columns="region", values="jual_total", aggfunc="sum")
panel = panel.reindex(columns=REGIONS).fillna(0.0)

# Test period (Jan-Oct 2025)
test = panel[panel.index.astype(str) >= "2025-01-01"]
zf = (test == 0).mean(axis=0).sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(zf.index, zf.values * 100, color="#1a73e8", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Fraction of zero-sales days in test period (%)")
ax.set_title("Zero-Sales Fraction per Region (test period, Jan-Oct 2025)")
for i, v in enumerate(zf.values):
    ax.text(v * 100 + 0.5, i, f"{v*100:.1f}%", va="center", fontsize=8)
ax.set_xlim(0, 105)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig3_zero_fraction.png"))
plt.close(fig)
print("Saved fig3_zero_fraction.png")
