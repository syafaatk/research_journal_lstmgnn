# -*- coding: utf-8 -*-
"""
Klasifikasi tipe pelanggan dari pelanggan_nama (item A1/A6):
RS, Puskesmas, Apotek, Klinik, PT/CV (perusahaan), Toko, Laboratorium, dll.
"""
import re
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

# Klasifikasi berdasarkan pola nama
PATTERNS = [
    ("Rumah Sakit", r"\b(rs|rumah sakit|rsud|rsi|rsia|rsb|rsk|rsup|rumkit|rs\.)\b"),
    ("Puskesmas", r"\b(puskesmas|pkm|pustu)\b"),
    ("Apotek", r"\b(apotek|apo?tik)\b"),
    ("Klinik", r"\b(klinik|klini|poliklinik|poli)\b"),
    ("Laboratorium", r"\b(lab|laboratorium)\b"),
    ("Perusahaan (PT/CV)", r"\b(pt\.?|cv\.?|perseroan|perusahaan|toko|distributor|suplier)\b"),
    ("Yayasan", r"\b(yay|yayasan)\b"),
    ("Pemerintah/Dinas", r"\b(dinas|kantor|pemerintah)\b"),
    ("Lainnya", None),
]

def classify(name):
    n = str(name).lower()
    for label, pat in PATTERNS:
        if pat is None:
            continue
        if re.search(pat, n):
            return label
    return "Lainnya"

df["cust_type"] = df["pelanggan_nama"].apply(classify)

print("=== Distribusi tipe pelanggan (level faktur) ===")
vc = df["cust_type"].value_counts()
for k, v in vc.items():
    print(f"  {k:<28} {v:>6} faktur ({v/len(df):.1%})")

print("\n=== Tipe pelanggan per region (faktur) ===")
ct = pd.crosstab(df["region"], df["cust_type"])
ct = ct.reindex(REGIONS)
print(ct.to_string())

print("\n=== Pelanggan unik per tipe ===")
uc = df.groupby("cust_type")["pelanggan_id"].nunique()
for k, v in uc.items():
    print(f"  {k:<28} {v:>4} pelanggan")

print("\n=== Contoh nama pelanggan per tipe ===")
for t in ["Rumah Sakit", "Puskesmas", "Apotek", "Klinik", "Perusahaan (PT/CV)", "Lainnya"]:
    ex = df[df["cust_type"] == t]["pelanggan_nama"].drop_duplicates().head(5).tolist()
    print(f"  [{t}]")
    for e in ex:
        print(f"    - {e}")

print("\n=== Top 15 pelanggan (faktur) ===")
top = df["pelanggan_nama"].value_counts().head(15)
for k, v in top.items():
    t = classify(k)
    print(f"  {v:>5}  {t:<28} {k}")