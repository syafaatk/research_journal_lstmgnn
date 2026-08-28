# -*- coding: utf-8 -*-
"""
Analisis koordinat untuk item A1 (daftar_tbd.md):
1. Kualitas data latitude/longitude di file
2. Koordinat per region yang dipakai eksperimen (mean lokasi pelanggan)
3. Sebaran lokasi pelanggan per region (apakah mean representatif?)
4. Simpan koordinat untuk dibandingkan dengan ibu kota resmi
"""
import json
import numpy as np
import pandas as pd

DATA = r"E:\Download\Jurnal\view_penjualan_detail data hingga oktober.xlsx"

CITY_MAP = {
    "Palembang": "Palembang", "Lahat": "Lahat", "Baturaja": "Ogan Komering Ulu",
    "Oku (Ogan Komering Ulu)": "Ogan Komering Ulu Timur", "Musi Banyuasin": "Musi Banyuasin",
    "Muara Enim": "Muara Enim", "Tanjung Enim": "Muara Enim", "Muara Lawai": "Muara Enim",
    "Lubuklinggau": "Lubuk Linggau", "Ogan Ilir": "Ogan Ilir", "Oku Selatan": "Ogan Komering Ulu Selatan",
    "Prabumulih": "Prabumulih", "Prabumulih Timur": "Prabumulih",
    "Pali (Penukal Abab Lematang Ilir)": "Penukal Abab Lematang Ilir", "Talang Ubi": "Penukal Abab Lematang Ilir",
    "Musi Rawas": "Musi Rawas", "Oku Timur": "Ogan Komering Ulu Timur", "Martapura": "Ogan Komering Ulu Timur",
    "Empat Lawang": "Empat Lawang", "Banyuasin": "Banyuasin", "Pagaralam": "Pagar Alam",
    "Muara Rupit": "Musi Rawas Utara", "Rupit": "Musi Rawas Utara",
    "Oki (Ogan Komering Ilir)": "Ogan Komering Ilir",
}
REGIONS = [
    "Banyuasin", "Empat Lawang", "Lahat", "Lubuk Linggau", "Muara Enim",
    "Musi Banyuasin", "Musi Rawas", "Musi Rawas Utara", "Ogan Ilir",
    "Ogan Komering Ilir", "Ogan Komering Ulu", "Ogan Komering Ulu Selatan",
    "Ogan Komering Ulu Timur", "Pagar Alam", "Palembang",
    "Penukal Abab Lematang Ilir", "Prabumulih",
]

df = pd.read_excel(DATA)
df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
df = df.drop_duplicates("d_jual_nofak")
df = df[df["wilayah"].isin(CITY_MAP)].copy()
df["region"] = df["wilayah"].map(CITY_MAP)

print("=== Kualitas latitude/longitude ===")
print(f"  lat null: {df['latitude'].isna().sum()} | lon null: {df['longitude'].isna().sum()}")
lat = df["latitude"].astype(float)
lon = df["longitude"].astype(float)
print(f"  lat range: {lat.min():.4f} s.d. {lat.max():.4f}")
print(f"  lon range: {lon.min():.4f} s.d. {lon.max():.4f}")
# Plausibilitas: Sumatera Selatan kira-kira lat -5.5 s.d. -1.5, lon 102.5 s.d. 106
bad = ((lat < -6) | (lat > -1) | (lon < 102) | (lon > 107)).sum()
print(f"  di luar batas wajar Sumsel (lat -6..-1, lon 102..107): {bad}")

print("\n=== Koordinat per region (mean lokasi pelanggan, dipakai eksperimen) ===")
coords = df.groupby("region").agg(
    lat=("latitude", "mean"), lon=("longitude", "mean"),
    n_cust=("pelanggan_id", "nunique"),
    n_inv=("d_jual_nofak", "count"),
)
coords = coords.reindex(REGIONS)
print(f"{'region':<28} {'n_cust':>6} {'n_inv':>6} {'lat':>9} {'lon':>9} {'lat_std':>8} {'lon_std':>8}")
for r in REGIONS:
    c = coords.loc[r]
    sub = df[df["region"] == r]
    print(f"{r:<28} {c['n_cust']:>6.0f} {c['n_inv']:>6.0f} {c['lat']:>9.4f} {c['lon']:>9.4f} "
          f"{sub['latitude'].std():>8.4f} {sub['longitude'].std():>8.4f}")

# Simpan koordinat eksperimen
out = {r: {"lat": float(coords.loc[r, "lat"]), "lon": float(coords.loc[r, "lon"])} for r in REGIONS}
with open(r"E:\Download\Jurnal\skrip\coords_experiment.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nKoordinat eksperimen disimpan ke skrip/coords_experiment.json")

# Jarak Haversine antar region (koordinat eksperimen)
def haversine(a, b):
    R = 6371.0
    la1, lo1, la2, lo2 = map(np.radians, [a["lat"], a["lon"], b["lat"], b["lon"]])
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(h))

dists = []
for i, r1 in enumerate(REGIONS):
    for r2 in REGIONS[i + 1:]:
        dists.append(haversine(out[r1], out[r2]))
dists = np.array(dists)
print(f"\nJarak antar region (koordinat eksperimen): min {dists.min():.2f}, "
      f"median {np.median(dists):.2f}, max {dists.max():.2f} km")