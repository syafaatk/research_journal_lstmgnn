# -*- coding: utf-8 -*-
"""Generate figures for paper #5 (event-driven COVID demand spikes).

Produces (folder figures/paper5_covid/):
- fig1_delta_spike_ratio.png : product spike ratios during Delta wave
- fig2_covid_share.png       : COVID product group share of total sales
- fig3_mask_negative.png     : mask quantity COVID peak vs El Nino (negative case)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:\Download\Jurnal"
FIGDIR = os.path.join(ROOT, "figures", "paper5_covid")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# 1. Delta wave product spike ratios (from manuscript verified numbers)
# ---------------------------------------------------------------------------
products = ["Handscoon", "Rapid test", "Auto-destruct\nsyringe", "Masker",
            "Tabung\nOxygen", "Breathing\nCircuit"]
ratios = [43.9, 25.4, 10.0, 33.8, 12.7, 11.3]
colors = ["#d93025", "#d93025", "#f9ab00", "#d93025", "#f9ab00", "#f9ab00"]

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(products, ratios, color=colors, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, ratios):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}x",
            ha="center", fontsize=9)
ax.set_ylabel("Ratio to monthly baseline median")
ax.set_title("Product Demand Spike Ratios During the Delta Wave\n"
             "(July-August 2021, relative to monthly baseline median)")
ax.set_ylim(0, 50)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig1_delta_spike_ratio.png"))
plt.close(fig)
print("Saved fig1_delta_spike_ratio.png")

# ---------------------------------------------------------------------------
# 2. COVID product group share (19.7% of total)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
labels = ["COVID product group\n(19.7%)", "Other products\n(80.3%)"]
sizes = [19.7, 80.3]
colors = ["#d93025", "#9aa7b0"]
wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                  autopct="%1.1f%%", startangle=90,
                                  textprops={"fontsize": 10})
for at in autotexts:
    at.set_color("white")
    at.set_fontweight("bold")
ax.set_title("COVID Product Group Share of Total Medical Device Sales\n"
             "(99.3M IDR of total)")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig2_covid_share.png"))
plt.close(fig)
print("Saved fig2_covid_share.png")

# ---------------------------------------------------------------------------
# 3. Mask quantity: COVID peak vs El Nino (negative case)
# ---------------------------------------------------------------------------
# From manuscript: mask ratio during El Nino = 0.47 (21,050 vs 44,907 pcs/month).
# COVID peak: Sep 2020 = 290,000 pcs; Sep 2021 = 362,740 pcs.
scenarios = ["COVID peak\n(Sep 2021)", "COVID peak\n(Sep 2020)", "Baseline\n(monthly median)", "El Nino\n(Sep-Oct 2023)"]
qty = [362740, 290000, 44907, 21050]
colors = ["#d93025", "#d93025", "#9aa7b0", "#188038"]

fig, ax = plt.subplots(figsize=(8, 5.5))
bars = ax.bar(scenarios, qty, color=colors, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, qty):
    ax.text(b.get_x() + b.get_width() / 2, v + 5000, f"{v:,}",
            ha="center", fontsize=9)
ax.set_ylabel("Mask quantity (pcs, unit-normalized)")
ax.set_title("Mask Demand: COVID Peaks vs El Nino Forest Fires\n"
             "(El Nino ratio = 0.47, a negative case)")
ax.set_ylim(0, 420000)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig3_mask_negative.png"))
plt.close(fig)
print("Saved fig3_mask_negative.png")
