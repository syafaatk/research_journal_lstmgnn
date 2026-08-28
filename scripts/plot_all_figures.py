# -*- coding: utf-8 -*-
"""
Plot semua figure paper dalam format JPG (dpi 300).

Figure yang dihasilkan (folder E:\\Download\\Jurnal\\figures\\):
  fig01_architecture.jpg        - diagram arsitektur LSTM-GNN zero-inflated 2 tahap
  fig02_study_area_map.jpg      - peta Sumsel: batas 16 kab/kota, centroid region,
                                  lokasi pelanggan (geocoded), edge graf k=3
  fig03_spatial_graph.jpg       - graf spasial: node 16 region (ukuran = volume faktur),
                                  edge k=3 (30 edge, koordinat geocoded)
  fig04_data_overview.jpg       - time series penjualan harian + fraksi nol per region
  fig05_actual_vs_predicted.jpg - actual vs predicted (test period): scatter pooled,
                                  time series region terpilih, total harian
  fig06_error_map.jpg           - peta error per region (R2) pada peta Sumsel
  fig07_error_vs_volume.jpg     - hubungan error (RMSE) dengan volume transaksi test
  fig08_residuals.jpg           - residual: histogram, over time, per region
  fig09_per_region_metrics.jpg  - bar chart R2 per region
  fig10_ablation_baselines.jpg  - bar chart ablation + baseline (R2)

Penggunaan: python plot_all_figures.py [path_results_json]
  Default: skrip/experiment_results_zi.json (hasil eksperimen lama).
  Setelah eksperimen geocoded selesai, jalankan ulang dengan
  skrip/experiment_results_zi_geocoded.json untuk regenerasi.
"""
import json
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import TwoSlopeNorm

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})

BASE = r"E:\Download\Jurnal"
RESULTS_DIR = os.path.join(BASE, "results")
FIGDIR = os.path.join(BASE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
GEO = os.path.join(BASE, "data", "customers_geocoded_final.csv")
GEOJSON_DIR = os.path.join(BASE, "data", "geojson")
RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RESULTS_DIR, "experiment_results_zi_geocoded.json")

REGIONS = [
    "Banyuasin", "Empat Lawang", "Lahat", "Lubuk Linggau", "Muara Enim",
    "Musi Banyuasin", "Musi Rawas", "Ogan Ilir",
    "Ogan Komering Ilir", "Ogan Komering Ulu", "Ogan Komering Ulu Selatan",
    "Ogan Komering Ulu Timur", "Pagar Alam", "Palembang",
    "Penukal Abab Lematang Ilir", "Prabumulih",
]
N = len(REGIONS)
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
FALLBACK = {}

# ---------------------------------------------------------------------------
# 1. Muat hasil eksperimen
# ---------------------------------------------------------------------------
with open(RESULTS, encoding="utf-8") as f:
    R = json.load(f)
print(f"Hasil: {RESULTS}")

# Cek kompatibilitas: hasil eksperimen harus 16 region (setelah merge MRU)
RESULTS_COMPATIBLE = len(R.get("per_region", {})) == N
if not RESULTS_COMPATIBLE:
    print(f"PERINGATAN: hasil eksperimen {len(R.get('per_region', {}))} region != "
          f"{N} region saat ini. Figure 05-10 (tergantung hasil) dilewati.")

# ---------------------------------------------------------------------------
# 2. Koordinat region dari geocoding (sama dengan experiment_zi_geocoded.py)
# ---------------------------------------------------------------------------
geo = pd.read_csv(GEO)
geo = geo[geo["status"].isin(["FOUND_GEO", "FOUND_SIRS"])].copy()

df = pd.read_excel(DATA)
df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
df = df.drop_duplicates("d_jual_nofak")
df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                        "jual_total_fak": "jual_total", "latitude": "Latitude"})
df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
df["region"] = df["pelanggan_kota"].map(CITY_MAP)
df["period"] = df["jual_tanggal"].dt.to_period("D")

inv_per_cust = df.groupby("pelanggan_nama")["d_jual_nofak"].nunique().rename("n_inv_data")
geo = geo.merge(inv_per_cust, left_on="pelanggan_nama", right_index=True, how="left")
geo["n_inv_data"] = geo["n_inv_data"].fillna(0)

w = geo.groupby("region").apply(
    lambda g: pd.Series({
        "lat": (g["lat"] * g["n_inv_data"]).sum() / g["n_inv_data"].sum(),
        "lon": (g["lon"] * g["n_inv_data"]).sum() / g["n_inv_data"].sum(),
        "n_cust": len(g), "n_inv": g["n_inv_data"].sum(),
    }), include_groups=False)
coords = w[["lat", "lon"]].reindex(REGIONS)
for r, (lat, lon) in FALLBACK.items():
    if np.isnan(coords.loc[r, "lat"]):
        coords.loc[r, "lat"], coords.loc[r, "lon"] = lat, lon

# ---------------------------------------------------------------------------
# 3. Panel harian + split (untuk yte dan time series)
# ---------------------------------------------------------------------------
panel = df.pivot_table(index="period", columns="region", values="jual_total", aggfunc="sum")
panel = panel.reindex(columns=REGIONS).fillna(0.0).sort_index()
days = panel.index.astype(str).tolist()
n_days = len(panel)
WINDOW = 30
t_all = panel.index[WINDOW:]
te_mask = np.array([m.start_time > pd.Timestamp("2024-12-31") for m in t_all])
yte = panel.iloc[WINDOW:].values[te_mask]          # (231, 16)
te_days = t_all[te_mask].astype(str).tolist()

# ---------------------------------------------------------------------------
# 4. Graf k=3 dari koordinat geocoded (threshold = max 3-NN)
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

D = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        D[i, j] = haversine(coords.iloc[i].lat, coords.iloc[i].lon,
                            coords.iloc[j].lat, coords.iloc[j].lon)
K = 3
knn_dists = np.array([np.sort(D[i])[1:K + 1] for i in range(N)])
THRESHOLD = float(knn_dists.max())
adj = np.zeros((N, N))
for i in range(N):
    order = np.argsort(D[i])
    for j in order[1:K + 1]:
        if D[i, j] <= THRESHOLD:
            adj[i, j] = 1
            adj[j, i] = 1
np.fill_diagonal(adj, 0)
edges = [(i, j) for i in range(N) for j in range(i + 1, N) if adj[i, j] == 1]
print(f"Graf: {len(edges)} edge, threshold {THRESHOLD:.2f} km")

# ---------------------------------------------------------------------------
# 5. Muat GeoJSON batas kab/kota (merge semua polygon per region)
# ---------------------------------------------------------------------------
def norm_name(s):
    n = str(s).replace(" ", "").replace(".", "").lower()
    if n.startswith("kota"):
        n = n[4:]
    return n

# Musi Rawas Utara digabung ke Musi Rawas (keputusan 19 Agu 2026)
MERGE = {"musirawasutara": "Musi Rawas"}

region_polys = {}   # region -> list of (lon_array, lat_array)
for fname in os.listdir(GEOJSON_DIR):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(GEOJSON_DIR, fname), encoding="utf-8") as f:
        gj = json.load(f)
    for feat in gj.get("features", []):
        name = feat.get("properties", {}).get("WADMKK")
        if not name:
            continue
        nn = norm_name(name)
        region = MERGE.get(nn) or next((r for r in REGIONS if norm_name(r) == nn), None)
        if region is None:
            continue
        geom = feat.get("geometry", {})
        rings = []
        if geom.get("type") == "Polygon":
            rings = [geom["coordinates"]]
        elif geom.get("type") == "MultiPolygon":
            rings = geom["coordinates"]
        for ring in rings:
            arr = np.array(ring[0])
            region_polys.setdefault(region, []).append((arr[:, 0], arr[:, 1]))

missing_polys = [r for r in REGIONS if r not in region_polys]
if missing_polys:
    print("Peringatan: region tanpa polygon:", missing_polys)

# ---------------------------------------------------------------------------
# Helper: prediksi ensemble (preds_mean) -> (231, 16)
# CATATAN (revisi 21 Agu 2026): seluruh figure hasil model memakai ensemble
# Hybrid BASE (128/128, dropout 0), konsisten dengan angka di naskah
# (Tabel 7 dan analisis residual Bagian 4.6 = per_region_hybrid/residual_hybrid).
# Sebelumnya keliru memakai HybridTuned seed 42.
# ---------------------------------------------------------------------------
pred_mean = np.array(R["models"]["Hybrid"]["preds_mean"])   # (231, 16)
pred_lstm = np.array(R["models"]["LSTM"]["preds_mean"])
pred_gnn = np.array(R["models"]["GNN"]["preds_mean"])
PER_REGION = R["per_region_hybrid"]

# ===========================================================================
# FIG 01: Arsitektur model
# ===========================================================================
def fig_architecture():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis("off")

    def box(x, y, w, h, text, fc="#e8f0fe", ec="#1a73e8", fs=9.5, bold=False, rounded=True):
        style = "round,pad=0.02,rounding_size=0.12" if rounded else "square,pad=0.02"
        b = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=1.4)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2, text=None, color="#444"):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                            lw=1.6, color=color)
        ax.add_patch(a)
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, text, ha="center", va="bottom",
                    fontsize=8.5, color=color, style="italic")

    # Input
    box(0.3, 3.4, 2.2, 1.4, "Input X\n(T=30, N=16, F=2)\nsales history + COVID flag",
        fc="#fef7e0", ec="#f9ab00")
    # LSTM
    box(3.2, 3.4, 2.0, 1.4, "LSTM\n(128 units)\ntemporal module", fc="#e8f0fe", ec="#1a73e8", bold=True)
    # GNN
    box(3.2, 0.6, 2.0, 1.4, "GCN\n(128 units)\nspatial module\nA: adjacency (16x16)", fc="#e8f0fe", ec="#1a73e8", bold=True)
    # Adjacency input
    box(0.3, 0.6, 2.2, 1.4, f"Adjacency A\n(k=3 distance graph,\n{len(edges)} edges)", fc="#fef7e0", ec="#f9ab00")
    # Fusion
    box(6.0, 3.4, 2.2, 1.4, "Fusion\nconcat [h_LSTM, h_GCN]\n-> FC", fc="#e6f4ea", ec="#188038", bold=True)
    # Stage 1
    box(9.0, 4.6, 2.6, 1.2, "Stage 1: Classification\np = P(sale > 0)\nBCE loss", fc="#fce8e6", ec="#d93025")
    # Stage 2
    box(9.0, 2.2, 2.6, 1.2, "Stage 2: Amount\nlog(1+y), masked MSE\nnon-zero days only", fc="#fce8e6", ec="#d93025")
    # Final
    box(9.0, 0.2, 2.6, 1.2, "Final prediction\ny_hat = 1[p>0.5] x amount", fc="#f3e8fd", ec="#9334e6", bold=True)

    arrow(2.5, 4.1, 3.2, 4.1)
    arrow(2.5, 1.3, 3.2, 1.3)
    arrow(5.2, 4.1, 6.0, 4.1, "h_LSTM")
    arrow(5.2, 1.3, 6.0, 3.4, "h_GCN")
    arrow(8.2, 4.1, 9.0, 5.2, "p")
    arrow(8.2, 4.1, 9.0, 2.8, "amount")
    arrow(10.3, 4.6, 10.3, 3.4, "", color="#d93025")
    arrow(10.3, 2.2, 10.3, 1.4, "", color="#d93025")
    ax.set_title("Figure 1. Zero-inflated Hybrid LSTM-GNN architecture (two-stage)")
    fig.savefig(os.path.join(FIGDIR, "fig01_architecture.jpg"))
    plt.close(fig)
    print("fig01_architecture.jpg")

# ===========================================================================
# FIG 02: Peta area studi
# ===========================================================================
def fig_study_area():
    fig, ax = plt.subplots(figsize=(9, 8))
    for region, polys in region_polys.items():
        for lon, lat in polys:
            ax.fill(lon, lat, fc="#eef2f5", ec="#9aa7b0", lw=0.6, zorder=1)
    # Edge graf
    for i, j in edges:
        ax.plot([coords.iloc[i].lon, coords.iloc[j].lon],
                [coords.iloc[i].lat, coords.iloc[j].lat],
                color="#1a73e8", lw=1.4, alpha=0.7, zorder=2)
    # Pelanggan (geocoded)
    ax.scatter(geo["lon"], geo["lat"], s=3, c="#d93025", alpha=0.45, zorder=3,
               label=f"Customer locations ({len(geo)})")
    # Centroid region
    ax.scatter(coords["lon"], coords["lat"], s=55, c="#188038", edgecolors="white",
               linewidths=0.8, zorder=4, label="Region centroid")
    # Palembang (pusat distribusi)
    pl = coords.loc["Palembang"]
    ax.scatter(pl.lon, pl.lat, s=160, marker="*", c="#f9ab00", edgecolors="black",
               linewidths=0.8, zorder=5, label="Distribution center (Palembang)")
    # Label region
    for r in REGIONS:
        ax.annotate(r, (coords.loc[r].lon, coords.loc[r].lat),
                    textcoords="offset points", xytext=(6, 6), fontsize=7.5, zorder=6)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Figure 2. Study area: South Sumatra, 16 regencies/cities,\n"
                 "customer locations and k=3 distance graph (geocoded coordinates)")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax.set_aspect(1.0 / np.cos(np.radians(-3.5)))
    fig.savefig(os.path.join(FIGDIR, "fig02_study_area_map.jpg"))
    plt.close(fig)
    print("fig02_study_area_map.jpg")

# ===========================================================================
# FIG 03: Graf spasial
# ===========================================================================
def fig_spatial_graph():
    fig, ax = plt.subplots(figsize=(9, 8))
    vol = panel.sum(axis=0)                       # total faktur per region
    sizes = 80 + 500 * (vol / vol.max())
    for i, j in edges:
        ax.plot([coords.iloc[i].lon, coords.iloc[j].lon],
                [coords.iloc[i].lat, coords.iloc[j].lat],
                color="#1a73e8", lw=1.6, alpha=0.75, zorder=1)
    sc = ax.scatter(coords["lon"], coords["lat"], s=sizes, c=vol,
                    cmap="YlOrRd", edgecolors="black", linewidths=0.7, zorder=3)
    for r in REGIONS:
        ax.annotate(r, (coords.loc[r].lon, coords.loc[r].lat),
                    textcoords="offset points", xytext=(6, 6), fontsize=7.5, zorder=4)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label("Total sales (IDR)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"Figure 3. Spatial graph: k=3 nearest neighbors, "
                 f"{len(edges)} edges, threshold {THRESHOLD:.2f} km\n"
                 f"node size = total sales volume")
    ax.set_aspect(1.0 / np.cos(np.radians(-3.5)))
    fig.savefig(os.path.join(FIGDIR, "fig03_spatial_graph.jpg"))
    plt.close(fig)
    print("fig03_spatial_graph.jpg")

# ===========================================================================
# FIG 04: Data overview
# ===========================================================================
def fig_data_overview():
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [1.6, 1]})
    # Panel A: total harian
    total = panel.sum(axis=1)
    ax = axes[0]
    ax.plot(days, total.values, lw=0.8, color="#1a73e8")
    ax.axvspan(days[0], "2022-12-31", color="#f9ab00", alpha=0.15, label="COVID-19 period")
    ax.set_ylabel("Total daily sales (IDR)")
    ax.set_title("Figure 4a. Daily total sales across 16 regions (Jul 2020 - Oct 2025)")
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    # Panel B: fraksi nol per region
    ax = axes[1]
    zf = (panel == 0).mean(axis=0).sort_values()
    ax.barh(zf.index, zf.values * 100, color="#188038")
    ax.set_xlabel("Fraction of zero-sales days (%)")
    ax.set_title("Figure 4b. Zero-sales fraction per region")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig04_data_overview.jpg"))
    plt.close(fig)
    print("fig04_data_overview.jpg")

# ===========================================================================
# FIG 05: Actual vs predicted
# ===========================================================================
def fig_actual_vs_pred():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # Panel A: scatter pooled
    ax = axes[0]
    ax.scatter(yte.flatten(), pred_mean.flatten(), s=4, alpha=0.35, color="#1a73e8")
    mx = max(yte.max(), pred_mean.max())
    ax.plot([0, mx], [0, mx], "k--", lw=1)
    ax.set_xscale("symlog"); ax.set_yscale("symlog")
    ax.set_xlabel("Actual (IDR)"); ax.set_ylabel("Predicted (IDR)")
    ss_res = np.sum((yte.flatten() - pred_mean.flatten()) ** 2)
    ss_tot = np.sum((yte.flatten() - yte.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    ax.set_title(f"Pooled (test, N={yte.size})\nR2 = {r2:.4f}")
    # Panel B: time series region terpilih
    ax = axes[1]
    sel = ["Palembang", "Lahat", "Muara Enim", "Ogan Komering Ulu"]
    for r in sel:
        idx = REGIONS.index(r)
        ax.plot(te_days, yte[:, idx], lw=0.7, alpha=0.85, label=f"{r} (actual)")
        ax.plot(te_days, pred_mean[:, idx], lw=0.7, ls="--", alpha=0.85, label=f"{r} (pred)")
    ax.set_ylabel("Sales (IDR)")
    ax.set_title("Actual vs predicted, selected regions (test)")
    ax.tick_params(axis="x", labelrotation=45, labelsize=6)
    ax.legend(fontsize=6.5, ncol=2)
    # Panel C: total harian
    ax = axes[2]
    ax.plot(te_days, yte.sum(axis=1), lw=0.9, color="#188038", label="Actual total")
    ax.plot(te_days, pred_mean.sum(axis=1), lw=0.9, ls="--", color="#d93025", label="Predicted total")
    ax.set_ylabel("Total sales (IDR)")
    ax.set_title("Daily total across regions (test)")
    ax.tick_params(axis="x", labelrotation=45, labelsize=6)
    ax.legend(fontsize=8)
    fig.suptitle("Figure 5. Actual vs predicted (Hybrid LSTM-GNN, ensemble mean)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig05_actual_vs_predicted.jpg"))
    plt.close(fig)
    print("fig05_actual_vs_predicted.jpg")

# ===========================================================================
# FIG 06: Peta error per region (R2)
# ===========================================================================
def fig_error_map():
    r2 = {r: PER_REGION[r]["R2"] for r in REGIONS}
    fig, ax = plt.subplots(figsize=(9, 8))
    vmin, vmax = -0.4, 1.0
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = plt.cm.RdYlGn
    for region, polys in region_polys.items():
        val = r2.get(region, np.nan)
        for lon, lat in polys:
            ax.fill(lon, lat, fc=cmap(norm(val)), ec="#666", lw=0.5, zorder=1)
    for r in REGIONS:
        ax.annotate(f"{r}\nR2={r2[r]:.3f}", (coords.loc[r].lon, coords.loc[r].lat),
                    ha="center", va="center", fontsize=7, zorder=3,
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.5))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=ax, shrink=0.8)
    cb.set_label("R-squared (test period)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title("Figure 6. Per-region R-squared on the test period")
    ax.set_aspect(1.0 / np.cos(np.radians(-3.5)))
    fig.savefig(os.path.join(FIGDIR, "fig06_error_map.jpg"))
    plt.close(fig)
    print("fig06_error_map.jpg")

# ===========================================================================
# FIG 07: Error vs volume
# ===========================================================================
def fig_error_volume():
    fig, ax = plt.subplots(figsize=(9, 7))
    vol_test = yte.sum(axis=0)
    rmse = np.array([PER_REGION[r]["RMSE"] for r in REGIONS])
    mae = np.array([PER_REGION[r]["MAE"] for r in REGIONS])
    ax.scatter(vol_test, rmse, s=70, c="#1a73e8", edgecolors="black", linewidths=0.6, zorder=3)
    for i, r in enumerate(REGIONS):
        ax.annotate(r, (vol_test[i], rmse[i]), textcoords="offset points",
                    xytext=(6, 6), fontsize=7.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Total test sales volume (IDR)")
    ax.set_ylabel("RMSE (IDR)")
    ax.set_title("Figure 7. Prediction error vs transaction volume (per region)")
    fig.savefig(os.path.join(FIGDIR, "fig07_error_vs_volume.jpg"))
    plt.close(fig)
    print("fig07_error_vs_volume.jpg")

# ===========================================================================
# FIG 08: Residual
# ===========================================================================
def fig_residuals():
    resid = (yte - pred_mean).flatten()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    # Histogram
    ax = axes[0]
    ax.hist(resid, bins=60, color="#1a73e8", alpha=0.8)
    ax.axvline(resid.mean(), color="#d93025", ls="--", lw=1.4,
               label=f"mean = {resid.mean():,.0f}")
    ax.set_xlabel("Residual (IDR)"); ax.set_ylabel("Frequency")
    ax.set_title("Residual histogram")
    ax.legend(fontsize=8)
    # Over time (agregat harian)
    ax = axes[1]
    resid_daily = (yte - pred_mean).sum(axis=1)
    ax.plot(te_days, resid_daily, lw=0.7, color="#188038", alpha=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Test day"); ax.set_ylabel("Daily total residual (IDR)")
    ax.set_title("Residuals over time (daily total)")
    ax.tick_params(axis="x", labelrotation=45, labelsize=6)
    # Per region (boxplot)
    ax = axes[2]
    data = [(yte[:, i] - pred_mean[:, i]) for i in range(N)]
    bp = ax.boxplot(data, tick_labels=REGIONS, showfliers=False, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#e8f0fe")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Residual (IDR)")
    ax.set_title("Residuals per region")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6.5)
    fig.suptitle("Figure 8. Residual analysis (Hybrid LSTM-GNN, ensemble mean)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig08_residuals.jpg"))
    plt.close(fig)
    print("fig08_residuals.jpg")

# ===========================================================================
# FIG 09: Per-region metrics
# ===========================================================================
def fig_per_region():
    fig, ax = plt.subplots(figsize=(9, 7))
    r2 = {r: PER_REGION[r]["R2"] for r in REGIONS}
    names = sorted(REGIONS, key=lambda r: r2[r])
    vals = [r2[r] for r in names]
    colors = ["#188038" if v >= 0 else "#d93025" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("R-squared (test period)")
    ax.set_title("Figure 9. Per-region R-squared (Hybrid LSTM-GNN)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig09_per_region_metrics.jpg"))
    plt.close(fig)
    print("fig09_per_region_metrics.jpg")

# ===========================================================================
# FIG 10: Ablation + baseline
# ===========================================================================
def fig_ablation_baselines():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    # Ablation
    ax = axes[0]
    abl = R["ablation"]
    names = list(abl.keys())
    vals = [abl[n]["R2"] for n in names]
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    ax.barh(names, vals, color="#1a73e8")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("R-squared")
    ax.set_title("Ablation study (tuned configuration, seed 42)")
    # Baseline
    ax = axes[1]
    base = R["baselines"]
    bnames = list(base.keys())
    bvals = [base[n]["R2"] for n in bnames]
    colors = ["#d93025" if v < 0 else "#188038" for v in bvals]
    ax.bar(bnames, bvals, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("R-squared")
    ax.set_title("Baselines")
    ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle("Figure 10. Ablation and baseline comparison", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig10_ablation_baselines.jpg"))
    plt.close(fig)
    print("fig10_ablation_baselines.jpg")

# ===========================================================================
# Jalankan semua
# ===========================================================================
if __name__ == "__main__":
    fig_architecture()
    fig_study_area()
    fig_spatial_graph()
    fig_data_overview()
    if RESULTS_COMPATIBLE:
        fig_actual_vs_pred()
        fig_error_map()
        fig_error_volume()
        fig_residuals()
        fig_per_region()
        fig_ablation_baselines()
    else:
        print("Figure 05-10 dilewati (hasil eksperimen belum 16 region).")
    print(f"\nSelesai. Semua figure di {FIGDIR}")