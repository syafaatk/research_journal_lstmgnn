# -*- coding: utf-8 -*-
"""Normalisasi nama barang (#6): basic + tanda baca, + daftar typo kandidat.

Pendekatan (aman, tidak mengarang penggabungan):
  1. Basic: strip + lowercase + spasi tunggal (sudah dipakai di pipeline).
  2. Tanda baca: hapus trailing '-', '.', ',', ';', ':' dan normalisasi spasi
     sebelum '%' (mis. "alkohol 70 %" -> "alkohol 70%"). Aman.
  3. Daftar typo kandidat (fuzzy) untuk KURASI MANUAL. Tidak digabung otomatis
     karena banyak pasangan mirip adalah produk/ukuran/tipe BERBEDA.

Output: results/nama_barang_normalisasi.json
"""
import json
import os
import re
from difflib import SequenceMatcher

import pandas as pd

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(BASE, "results", "nama_barang_normalisasi.json")

df = pd.read_excel(DATA)
df["jumlah"] = pd.to_numeric(df["jumlah"], errors="coerce").fillna(0.0)

def norm_basic(x):
    return re.sub(r"\s+", " ", str(x).strip().lower())

def norm_punct(x):
    # hapus trailing tanda baca, normalisasi spasi sebelum %
    s = re.sub(r"[\s\-.,;:]+$", "", x)
    s = re.sub(r"\s*%\s*", "%", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df["nama_asli"] = df["barang_nama"].astype(str)
df["nama_basic"] = df["nama_asli"].map(norm_basic)
df["nama_norm"] = df["nama_basic"].map(norm_punct)

report = {}
report["n_nama_asli"] = int(df["nama_asli"].nunique())
report["n_nama_basic"] = int(df["nama_basic"].nunique())
report["n_nama_norm"] = int(df["nama_norm"].nunique())
report["pecah_basic"] = report["n_nama_asli"] - report["n_nama_basic"]
report["pecah_punct"] = report["n_nama_basic"] - report["n_nama_norm"]

# --- Kelompok yang digabung oleh normalisasi tanda baca ---
g = df.groupby("nama_norm")["nama_basic"].nunique()
collapsed = g[g > 1].sort_values(ascending=False)
report["n_kelompok_punct"] = int(len(collapsed))
punct_groups = []
for nn, n in collapsed.items():
    names = df[df["nama_norm"] == nn]["nama_basic"].unique().tolist()
    val = df[df["nama_norm"] == nn]["jumlah"].sum()
    punct_groups.append({
        "nama_norm": nn, "n_nama": int(n),
        "nilai_jt": round(float(val) / 1e6, 1), "nama": names})
report["kelompok_punct"] = punct_groups

# --- Daftar typo kandidat (fuzzy) untuk kurasi manual ---
# Hanya pasangan dengan rasio tinggi; TIDAK digabung otomatis.
names = df.groupby("nama_norm").agg(
    n_baris=("jumlah", "count"),
    nilai_jt=("jumlah", lambda x: round(x.sum() / 1e6, 1)),
).reset_index()
name_list = names["nama_norm"].tolist()
n = len(name_list)
pairs = []
for i in range(n):
    for j in range(i + 1, n):
        a, b = name_list[i], name_list[j]
        if len(a) < 4 or len(b) < 4:
            continue
        r = SequenceMatcher(None, a, b).ratio()
        if r >= 0.90:
            pairs.append((r, a, b))
pairs.sort(reverse=True)
report["n_pasangan_fuzzy_ge90"] = len(pairs)
report["pasangan_fuzzy"] = [
    {"rasio": round(r, 3), "a": a, "b": b} for r, a, b in pairs]

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)

print("=" * 70)
print("NORMALISASI NAMA BARANG (#6)")
print("=" * 70)
print("Nama asli: {} | basic: {} (pecah {}) | +punct: {} (pecah {})".format(
    report["n_nama_asli"], report["n_nama_basic"], report["pecah_basic"],
    report["n_nama_norm"], report["pecah_punct"]))
print("Kelompok digabung oleh tanda baca: {} kelompok".format(
    report["n_kelompok_punct"]))
print()
print("--- Kelompok tanda baca (aman, digabung) ---")
for grp in punct_groups:
    print("[{}] {} nama, {:.1f} jt".format(grp["nama_norm"], grp["n_nama"], grp["nilai_jt"]))
    for nm in grp["nama"]:
        print("    -", nm)
print()
print("--- Typo kandidat (fuzzy >= 0.90) untuk KURASI MANUAL: {} pasangan ---".format(
    report["n_pasangan_fuzzy_ge90"]))
print("(TIDAK digabung otomatis; banyak yang produk/ukuran/tipe berbeda)")
print()
print("Selesai. Disimpan ke", OUT)
