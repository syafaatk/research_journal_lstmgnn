# -*- coding: utf-8 -*-
"""Generate figures for paper #6 (syringe procurement cycles).

Produces (folder figures/paper6_syringe/):
- fig1_monthly_series.png : monthly syringe quantity time series (Jul/Nov peaks)
- fig2_fiscal_cycle.png   : average quantity by calendar month (fiscal cycle)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:\Download\Jurnal"
FIGDIR = os.path.join(ROOT, "figures", "paper6_syringe")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

with open(os.path.join(ROOT, "results", "syringe_temporal.json"), encoding="utf-8") as f:
    data = json.load(f)

per_month = data["per_bulan_tahun"]
months = sorted(per_month.keys())
qty = [per_month[m]["qty"] for m in months]

# ---------------------------------------------------------------------------
# 1. Monthly syringe quantity time series
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(len(months))
colors = ["#1a73e8"] * len(months)
# Highlight July and November peaks (fiscal cycle)
for i, m in enumerate(months):
    mm = int(m.split("-")[1])
    if mm in (7, 11):
        colors[i] = "#f9ab00"
# Jul 2022 = highest peak; Nov 2023 = endemic peak
idx_jul22 = months.index("2022-07")
idx_nov23 = months.index("2023-11")
colors[idx_jul22] = "#d93025"
colors[idx_nov23] = "#9334e6"

bars = ax.bar(x, np.array(qty) / 1000, color=colors, edgecolor="black", linewidth=0.4)
ax.set_xticks(x[::3])
ax.set_xticklabels([months[i][:7] for i in range(0, len(months), 3)], rotation=45, fontsize=8)
ax.set_ylabel("Syringe quantity (thousand pcs)")
ax.set_title("Monthly Syringe Purchases (Jul 2020 - Oct 2025)\n"
             "Recurring July and November peaks follow the fiscal cycle")
ax.annotate("Jul 2022\n709,779 pcs\n(highest peak)",
            xy=(idx_jul22, qty[idx_jul22] / 1000),
            xytext=(idx_jul22 - 5, qty[idx_jul22] / 1000 + 120),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)
ax.annotate("Nov 2023\n460,191 pcs\n(endemic peak)",
            xy=(idx_nov23, qty[idx_nov23] / 1000),
            xytext=(idx_nov23 + 2, qty[idx_nov23] / 1000 + 120),
            arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig1_monthly_series.png"))
plt.close(fig)
print("Saved fig1_monthly_series.png")

# ---------------------------------------------------------------------------
# 2. Fiscal cycle pattern (average by calendar month)
# ---------------------------------------------------------------------------
month_vals = {m: [] for m in range(1, 13)}
for m, v in per_month.items():
    mm = int(m.split("-")[1])
    month_vals[mm].append(v["qty"])

avg = [np.mean(month_vals[m]) / 1000 if month_vals[m] else 0 for m in range(1, 13)]
labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

fig, ax = plt.subplots(figsize=(9, 5.5))
colors = ["#1a73e8"] * 12
colors[6] = "#f9ab00"   # July
colors[10] = "#f9ab00"  # November
bars = ax.bar(labels, avg, color=colors, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, avg):
    ax.text(b.get_x() + b.get_width() / 2, v + 5, f"{v:.0f}",
            ha="center", fontsize=8)
ax.set_ylabel("Mean syringe quantity (thousand pcs)")
ax.set_title("Fiscal Cycle Pattern: Mean Syringe Quantity by Calendar Month\n"
             "(peaks in July and November, following the budget cycle)")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig2_fiscal_cycle.png"))
plt.close(fig)
print("Saved fig2_fiscal_cycle.png")
