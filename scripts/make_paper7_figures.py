# -*- coding: utf-8 -*-
"""Generate figures for paper #7 (geocoding protocol).

Produces (folder figures/paper7_geocoding/):
- fig1_customer_map.png    : map of geocoded customer locations + centroids
- fig2_coverage.png        : geocoding coverage (118/156, 98.8% invoice value)
- fig3_coord_accuracy.png  : distance between original and geocoded coordinates
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:\Download\Jurnal"
FIGDIR = os.path.join(ROOT, "figures", "paper7_geocoding")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

geo = pd.read_csv(os.path.join(ROOT, "data", "customers_geocoded_final.csv"))
geo = geo.dropna(subset=["lat", "lon"])

# ---------------------------------------------------------------------------
# 1. Customer locations map + invoice-weighted centroids
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 8))
# Geocoded customers
geo_ok = geo[geo["status"].isin(["FOUND_GEO", "FOUND_SIRS"])]
geo_fb = geo[~geo["status"].isin(["FOUND_GEO", "FOUND_SIRS"])]
ax.scatter(geo_ok["lon"], geo_ok["lat"], s=20, c="#1a73e8", alpha=0.6,
           label=f"Geocoded customers ({len(geo_ok)})", zorder=2)
if len(geo_fb) > 0:
    ax.scatter(geo_fb["lon"], geo_fb["lat"], s=20, c="#f9ab00", alpha=0.7,
               label=f"City-center fallback ({len(geo_fb)})", zorder=2)

# Invoice-weighted centroids per region
geo["w"] = geo["n_inv"].fillna(1)
cent = geo.groupby("region").apply(
    lambda g: pd.Series({
        "lat": np.average(g["lat"], weights=g["w"]),
        "lon": np.average(g["lon"], weights=g["w"]),
    }), include_groups=False
).reset_index()
ax.scatter(cent["lon"], cent["lat"], s=90, c="#d93025", edgecolors="black",
           linewidths=0.8, zorder=3, label="Invoice-weighted centroid")
for _, r in cent.iterrows():
    ax.annotate(r["region"], (r["lon"], r["lat"]), textcoords="offset points",
                xytext=(6, 6), fontsize=7, zorder=4)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Geocoded Customer Locations and Invoice-Weighted Centroids\n"
             "(South Sumatra, 16 regions)")
ax.legend(loc="lower left", fontsize=8)
ax.set_aspect(1.0 / np.cos(np.radians(-3.5)))
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig1_customer_map.png"))
plt.close(fig)
print("Saved fig1_customer_map.png")

# ---------------------------------------------------------------------------
# 2. Coverage (geocoded vs city-center fallback)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
# By customer count
ax = axes[0]
labels = ["Geocoded\n(actual coords)", "City-center\nfallback"]
vals = [118, 38]
colors = ["#188038", "#9aa7b0"]
bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v}", ha="center", fontsize=10)
ax.set_ylabel("Number of customers")
ax.set_title("Geocoding Coverage by Customer Count\n(118 of 156 customers)")
ax.set_ylim(0, 135)
# By invoice value
ax = axes[1]
labels2 = ["Geocoded\n(98.8% of value)", "City-center\n(1.2% of value)"]
vals2 = [98.8, 1.2]
colors2 = ["#188038", "#9aa7b0"]
bars = ax.bar(labels2, vals2, color=colors2, edgecolor="black", linewidth=0.6)
for b, v in zip(bars, vals2):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
ax.set_ylabel("Share of invoice value (%)")
ax.set_title("Geocoding Coverage by Invoice Value\n(10,752 of 10,878 invoices)")
ax.set_ylim(0, 110)
fig.suptitle("Geocoding Coverage in an Address-Poor Market", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig2_coverage.png"))
plt.close(fig)
print("Saved fig2_coverage.png")

# ---------------------------------------------------------------------------
# 3. Coordinate accuracy (distance between original and geocoded)
# ---------------------------------------------------------------------------
dist = geo["dist_data_km"].dropna()
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.hist(dist, bins=30, color="#1a73e8", edgecolor="black", linewidth=0.4)
ax.axvline(dist.median(), color="#d93025", ls="--", lw=1.5,
           label=f"median = {dist.median():.1f} km")
ax.axvline(np.percentile(dist, 90), color="#f9ab00", ls="--", lw=1.5,
           label=f"p90 = {np.percentile(dist, 90):.1f} km")
ax.set_xlabel("Distance between original and geocoded coordinates (km)")
ax.set_ylabel("Number of customers")
ax.set_title("Coordinate Accuracy: Original vs Geocoded Locations\n"
             "(median 3.2 km, p90 25.6 km, max 125.7 km)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig3_coord_accuracy.png"))
plt.close(fig)
print("Saved fig3_coord_accuracy.png")
