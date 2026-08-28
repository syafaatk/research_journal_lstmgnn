# -*- coding: utf-8 -*-
"""
Ekstrak daftar pelanggan unik + koordinat dari data.
Output: skrip/customers_coords.csv (untuk verifikasi/geocoding manual)
"""
import numpy as np
import pandas as pd

DATA = r"E:\Download\Jurnal\view_penjualan_detail data hingga oktober.xlsx"
OUT = r"E:\Download\Jurnal\skrip\customers_coords.csv"

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

df = pd.read_excel(DATA)
df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
df = df.drop_duplicates("d_jual_nofak")
df = df[df["wilayah"].isin(CITY_MAP)].copy()
df["region"] = df["wilayah"].map(CITY_MAP)

g = df.groupby("pelanggan_nama").agg(
    region=("region", "first"),
    wilayah=("wilayah", "first"),
    n_inv=("d_jual_nofak", "count"),
    lat=("latitude", "mean"),
    lon=("longitude", "mean"),
    lat_std=("latitude", "std"),
    lon_std=("longitude", "std"),
    n_coord=("latitude", "nunique"),
)
g = g.sort_values("n_inv", ascending=False)

# Flag: koordinat data yang mencurigakan (std besar atau koordinat tunggal kasar)
g["flag"] = ""
g.loc[(g["lat_std"] > 0.05) | (g["lon_std"] > 0.05), "flag"] = "STD_BESAR"
g.loc[g["n_coord"] == 1, "flag"] = g["flag"] + "|KOORD_TUNGGAL"

g.to_csv(OUT, index_label="pelanggan_nama")
print(f"Total pelanggan unik: {len(g)}")
print(f"Disimpan ke {OUT}")
print()
print("=== Top 30 pelanggan (faktur) ===")
print(g.head(30).to_string())
print()
print("=== Pelanggan dengan koordinat mencurigakan ===")
sus = g[g["flag"] != ""]
print(f"Jumlah: {len(sus)}")
print(sus[["region", "n_inv", "lat", "lon", "lat_std", "lon_std", "flag"]].to_string())