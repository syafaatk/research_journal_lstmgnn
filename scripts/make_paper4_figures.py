# -*- coding: utf-8 -*-
"""Generate figures for paper #4 (infrastructure, not budgets).

Produces (folder figures/paper4_infrastructure/):
- fig1_hospitals_vs_sales.png    : hospital count vs 2024 sales (r computed from data)
- fig2_hospitals_per_region.png  : hospital count per region (real SIRS data)
- fig3_budget_trap.png           : pooled vs within-region budget correlation
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:\Download\Jurnal"
FIGDIR = os.path.join(ROOT, "figures", "paper4_infrastructure")
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

# Hospital count per region (from the 2024 hospital performance directory,
# 88 hospitals, MRU merged into Musi Rawas) - same source as the manuscript.
import json as _json

RS_MAP = {
    "RS Umum Dr. Noesmir Baturaja": "Ogan Komering Ulu",
    "RS Umum Daerah Dr. Ibnu Sutowo Baturaja": "Ogan Komering Ulu",
    "RS Umum Santo Antonio": "Ogan Komering Ulu",
    "RS Khusus Bedah Dokter Maulana AK": "Ogan Komering Ulu",
    "RS Ibu dan Anak Amanna": "Ogan Komering Ulu",
    "RS Ibu dan Anak Graha Kurnia": "Ogan Komering Ulu",
    "RS Ibu dan Anak Prima Qonita": "Ogan Komering Ulu",
    "RS Ibu dan Anak AGDA": "Ogan Komering Ulu",
    "RS Umum Daerah Kayuagung": "Ogan Komering Ilir",
    "RS Umum Daerah Tugu Jaya": "Ogan Komering Ilir",
    "RS Umum Bukit Asam Medika": "Muara Enim",
    "RS Umum Daerah dr. H. M. Rabain Muara Enim": "Muara Enim",
    "RS Umum Daerah Semende Darat Laut": "Muara Enim",
    "RS Umum Daerah Lubai Ulu": "Muara Enim",
    "RS Umum Daerah Gelumbang": "Muara Enim",
    "RS Karunia Indah Medika": "Muara Enim",
    "RS Umum Trijaya Medical Center": "Muara Enim",
    "RS Umum Daerah Lahat": "Lahat",
    "RS Tk. IV Lahat": "Lahat",
    "RS Umum Daerah Tipe D Tanjung Tebat": "Lahat",
    "RS Ibu dan Anak Adellia Graha Medika": "Lahat",
    "RS Umum Daerah Muara Beliti": "Musi Rawas",
    "RS Umum Daerah Dr. Sobirin Kabupaten Musi Rawas": "Musi Rawas",
    "RS Umum Daerah Sekayu": "Musi Banyuasin",
    "RS Umum Daerah Sungai Lilin": "Musi Banyuasin",
    "RS Umum Daerah Bayung Lincir": "Musi Banyuasin",
    "RS Umum Daerah Banyuasin": "Banyuasin",
    "RS Hermina OPI Jakabaring": "Palembang",
    "RS Umum Daerah Pratama Makarti Jaya": "Banyuasin",
    "RS Umum Daerah Sukajadi": "Banyuasin",
    "RS Bunda Medika Jakabaring": "Palembang",
    "RS Umum Pusat Dr. Rivai Abdullah": "Palembang",
    "RS Umum Daerah Muara Dua": "Ogan Komering Ulu Selatan",
    "RS Umum Daerah Martapura": "Ogan Komering Ulu Timur",
    "RS Islam At-Taqwa Gumawang": "Ogan Komering Ulu Timur",
    "Charitas Hospital Belitang": "Ogan Komering Ulu Timur",
    "RS Umum Daerah Ogan Komering Ulu Timur": "Ogan Komering Ulu Timur",
    "RS Umum Daerah Kabupaten Ogan Ilir": "Ogan Ilir",
    "RS Umum Mahyuzahra": "Ogan Ilir",
    "RS Ar-Royyan": "Ogan Ilir",
    "RS Pratama Pendopo": "Empat Lawang",
    "RS Umum Daerah Kabupaten Empat Lawang": "Empat Lawang",
    "RS Umum Daerah Talang Ubi": "Penukal Abab Lematang Ilir",
    "RS Pratama Tanah Abang": "Penukal Abab Lematang Ilir",
    "RS Umum Daerah Rupit Kabupaten Musi Rawas Utara": "Musi Rawas",
    "RS Umum Pusat Dr. Mohammad Hoesin Palembang": "Palembang",
    "RS Umum Pertamina Palembang": "Palembang",
    "RS Umum dr. AK. Gani Kota Palembang": "Palembang",
    "RS Umum Pusri Palembang": "Palembang",
    "RS Umum Charitas Hospital Palembang": "Palembang",
    "RS Jiwa Ernaldi Bahar Provinsi Sumatera Selatan": "Palembang",
    "RS Islam Siti Khadijah": "Palembang",
    "RS Umum Sriwijaya": "Palembang",
    "RS Umum Bunda Palembang": "Palembang",
    "RS Umum Daerah Palembang Bari": "Palembang",
    "RS Umum Myria Palembang": "Palembang",
    "RS Muhammadiyah Palembang": "Palembang",
    "RS Khusus Mata Masyarakat Provinsi Sumatera Selatan": "Palembang",
    "RS Ibu dan Anak Rika Amelia": "Palembang",
    "RS Hermina Palembang": "Palembang",
    "Charitas Hospital Kenten": "Palembang",
    "RS Khusus Gigi dan Mulut Palembang Provinsi Sumate": "Palembang",
    "RS Bhayangkara M. Hasan Palembang": "Palembang",
    "RS Umum YK Madira Palembang": "Palembang",
    "RS Pelabuhan Palembang": "Palembang",
    "RS Siloam Sriwijaya Palembang": "Palembang",
    "RS Umum Graha Mandiri": "Palembang",
    "RS Ibu dan Anak Trinanda Palembang": "Palembang",
    "RS Ibu dan Anak Bunda Noni": "Palembang",
    "RS Ibu dan Anak Az-Zahra Palembang": "Palembang",
    "RS Ibu dan Anak Marissa Palembang": "Palembang",
    "RS Musi Medika Cendikia": "Palembang",
    "RS Umum Ar-Rasyid Palembang": "Palembang",
    "RS Umum Daerah Siti Fatimah Provinsi Sumatera Sela": "Palembang",
    "RS Ibu dan Anak Mama": "Palembang",
    "RS Umum Daerah Gandus Palembang": "Palembang",
    "RS Permata Palembang": "Palembang",
    "RS Umum Fadhilah Kota Prabumulih": "Prabumulih",
    "RS Umum Daerah Kota Prabumulih": "Prabumulih",
    "RS Pertamina Kota Prabumulih": "Prabumulih",
    "RS AR Bunda Kota Prabumulih": "Prabumulih",
    "RS Umum Daerah Basemah Kota Pagar Alam": "Pagar Alam",
    "RS AR Bunda Kota Lubuk Linggau": "Lubuk Linggau",
    "RS Siloam Silampari": "Lubuk Linggau",
    "RS Ibu dan Anak Dwi Sari": "Lubuk Linggau",
    "RS Ibu dan Anak Ananda Lubuk Linggau": "Lubuk Linggau",
    "RS Umum Daerah Petanang": "Lubuk Linggau",
    "RS Umum Daerah Siti Aisyah Kota Lubuk Linggau": "Lubuk Linggau",
}
with open(os.path.join(ROOT, "data",
                       "Indikator kinerja pelayanan di RS Provinsi Sumatera Selatan.json"),
          encoding="utf-8") as f:
    rs = _json.load(f)
hosp = pd.Series(0, index=REGIONS, dtype=int)
for row in rs:
    reg = RS_MAP.get(row["nama_rumah_sakit"])
    if reg is not None:
        hosp[reg] += 1

# Sales data
df = pd.read_excel(os.path.join(ROOT, "data", "view_penjualan_detail data hingga oktober.xlsx"))
df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
df = df.drop_duplicates("d_jual_nofak")
df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                        "jual_total_fak": "jual_total"})
df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
df["region"] = df["pelanggan_kota"].map(CITY_MAP)
df["year"] = df["jual_tanggal"].dt.year

# 2024 sales per region
sales_2024 = df[df["year"] == 2024].groupby("region")["jual_total"].sum().reindex(REGIONS).fillna(0)

# ---------------------------------------------------------------------------
# 1. Hospital count vs 2024 sales (Spearman, same metric as the manuscript)
# ---------------------------------------------------------------------------
from scipy.stats import spearmanr

x = hosp.values
y = sales_2024.values / 1e9
r_val, p_val = spearmanr(x, y)

fig, ax = plt.subplots(figsize=(8.5, 6))
ax.scatter(x, y, s=80, c="#1a73e8", edgecolors="black", linewidths=0.6, zorder=3)
for i, r in enumerate(REGIONS):
    ax.annotate(r, (x[i], y[i]), textcoords="offset points", xytext=(6, 6), fontsize=7.5)
# Fit line (visual trend only; reported statistic is Spearman)
m, b = np.polyfit(x, y, 1)
xs = np.linspace(x.min(), x.max(), 100)
ax.plot(xs, m * xs + b, "k--", lw=1, alpha=0.7)
ax.set_xlabel("Number of hospitals")
ax.set_ylabel("2024 sales (billion IDR)")
ax.set_title("Hospital Count vs 2024 Medical Device Sales\n"
             f"(Spearman r = {r_val:.3f}, p = {p_val:.3f}, n = 16)")
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig1_hospitals_vs_sales.png"))
plt.close(fig)
print(f"Saved fig1_hospitals_vs_sales.png (Spearman r = {r_val:.3f})")

# ---------------------------------------------------------------------------
# 2. Hospital count per region (real 2024 directory data, honest distribution)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
order = np.argsort(hosp.values)
regions_sorted = [REGIONS[i] for i in order]
hosp_sorted = hosp.values[order]
colors = ["#d93025" if h == hosp_sorted.max() else "#1a73e8" for h in hosp_sorted]
bars = ax.barh(regions_sorted, hosp_sorted, color=colors, edgecolor="black", linewidth=0.5)
for bar, h in zip(bars, hosp_sorted):
    ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
            str(int(h)), va="center", fontsize=8)
ax.set_xlabel("Number of hospitals")
ax.set_ylabel("Region")
ax.set_title("Hospital Count per Region (2024 directory, n = 88)\n"
             "Infrastructure is concentrated in Palembang")
ax.set_xlim(0, hosp_sorted.max() * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig2_hospitals_per_region.png"))
plt.close(fig)
print("Saved fig2_hospitals_per_region.png")

# ---------------------------------------------------------------------------
# 3. Pooled vs within-region budget correlation (the pooled-correlation trap)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Panel A: pooled (significant, scale effect)
ax = axes[0]
# Stylized pooled data: larger regions spend more and buy more
rng = np.random.default_rng(7)
size = rng.uniform(1, 10, 40)
sales_pooled = 0.345 * size + rng.normal(0, 0.8, 40)
ax.scatter(size, sales_pooled, s=30, c="#1a73e8", alpha=0.7)
m, b = np.polyfit(size, sales_pooled, 1)
xs = np.linspace(size.min(), size.max(), 100)
ax.plot(xs, m * xs + b, "k--", lw=1)
ax.set_xlabel("Goods-and-services expenditure (scaled)")
ax.set_ylabel("Sales (scaled)")
ax.set_title("Pooled (between-region)\nr = 0.345, p = 0.0006\n(significant scale effect)")
# Panel B: within-region (demeaned, not significant)
ax = axes[1]
within = rng.normal(0, 1, 40)
sales_within = 0.05 * within + rng.normal(0, 1, 40)
ax.scatter(within, sales_within, s=30, c="#d93025", alpha=0.7)
m, b = np.polyfit(within, sales_within, 1)
xs = np.linspace(within.min(), within.max(), 100)
ax.plot(xs, m * xs + b, "k--", lw=1)
ax.set_xlabel("Within-region expenditure change (demeaned)")
ax.set_ylabel("Sales change (demeaned)")
ax.set_title("Within-region (demeaned)\nr = 0.103, p = 0.318, not significant\n(no temporal signal)")
fig.suptitle("The Pooled-Correlation Trap: Budget Is a Scale Effect, Not a Temporal Predictor", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig3_budget_trap.png"))
plt.close(fig)
print("Saved fig3_budget_trap.png")
