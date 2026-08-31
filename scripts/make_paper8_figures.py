# -*- coding: utf-8 -*-
"""Generate figures for paper #8 (zero-inflated classifier collapse).

Produces (folder figures/paper8_collapse/):
- fig1_classifier_prob.png : mean classifier probability per region (collapse)
- fig2_mitigation.png      : mitigation experiment R-squared
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:\Download\Jurnal"
FIGDIR = os.path.join(ROOT, "figures", "paper8_collapse")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ---------------------------------------------------------------------------
# 1. Classifier probability per region (from manuscript Table 1)
# ---------------------------------------------------------------------------
regions = ["Palembang", "OKU Timur", "OKU", "Lahat", "All other 12 regions"]
mean_p = [0.969, 0.264, 0.245, 0.231, 0.20]
max_p = [0.991, 0.381, 0.403, 0.461, 0.32]
true_nonzero = [223, 72, 66, 57, 0]

fig, ax = plt.subplots(figsize=(9, 6))
x = np.arange(len(regions))
width = 0.35
bars1 = ax.bar(x - width / 2, mean_p, width, color="#1a73e8",
               edgecolor="black", linewidth=0.6, label="Mean classifier p")
bars2 = ax.bar(x + width / 2, max_p, width, color="#f9ab00",
               edgecolor="black", linewidth=0.6, label="Max classifier p")
ax.axhline(0.5, color="#d93025", ls="--", lw=1.5, label="p = 0.5 threshold")
for i, v in enumerate(mean_p):
    ax.text(i - width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
for i, v in enumerate(max_p):
    ax.text(i + width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(regions, fontsize=8)
ax.set_ylabel("Classifier probability")
ax.set_title("Zero-Inflated Classifier Collapse: Mean and Max Probability per Region\n"
             "(15 of 16 regions never exceed p = 0.5; only Palembang is gated to non-zero)")
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig1_classifier_prob.png"))
plt.close(fig)
print("Saved fig1_classifier_prob.png")

# ---------------------------------------------------------------------------
# 2. Mitigation experiment R-squared (from manuscript Table 2)
# ---------------------------------------------------------------------------
configs = ["A baseline\n(masked reg)", "C1 wt+fullreg\n(pw=4.1)", "C3 base+fullreg\n(pw=1.0)",
           "B1 weighted\n(pw=4.1)", "D2 full-only\n(no ZI gate)", "D masked-only\n(regression only)"]
r2 = [0.0367, 0.0494, 0.0494, -0.7055, -0.0753, -60.28]
colors = ["#9aa7b0", "#188038", "#188038", "#d93025", "#f9ab00", "#d93025"]

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(configs, r2, color=colors, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, r2):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.5 if v >= 0 else -0.5),
            f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}",
            ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("R-squared")
ax.set_title("Mitigation Experiments for the Classifier Collapse\n"
             "(full-regression improves; weighted BCE over-predicts; removing the ZI gate is catastrophic)")
ax.set_ylim(-65, 8)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig2_mitigation.png"))
plt.close(fig)
print("Saved fig2_mitigation.png")
