# -*- coding: utf-8 -*-
"""Analisis kassa: normalisasi merek + qty per produk (nama x satuan).

Temuan eksplorasi:
  - Satuan: Roll (326), Box (288), Pack (144). barang_satuan_id Pack = 12 & 28.
  - Merek (kategori_nama): Omega, Winner, Meddis.., OneMed, Narita Farma.
  - PENTING: Roll/Box/Pack adalah PRODUK BERBEDA, bukan isi box yang sama:
      * Roll  = "Kassa" 40x80 (gulungan besar, ~120.000/roll)
      * Box   = "Kassa Steril" 16x16 (kotak steril kecil, ~4.200/box)
      * Pack  = "Gauze Swab" 10x10cm 16Ply 10s (~18.000/pack)
    Rasio harga Roll/Box ~28x bukan indikasi isi box, tapi produk berbeda.
  - Karena produk berbeda, TIDAK ada konversi satuan yang valid. Qty dianalisis
    per produk (nama + satuan) + nilai rupiah konsisten.

Output: results/kassa_analysis.json
"""
import json
import os
import re

import numpy as np
import pandas as pd

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(BASE, "results", "kassa_analysis.json")

df = pd.read_excel(DATA)
df["jumlah"] = pd.to_numeric(df["jumlah"], errors="coerce").fillna(0.0)
df["d_jual_qty"] = pd.to_numeric(df["d_jual_qty"], errors="coerce").fillna(0.0)
df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
df["month"] = df["tanggal"].dt.to_period("M").astype(str)

k = df[df["barang_nama"].str.contains("kassa|gauze", case=False, na=False)].copy()

# --- Normalisasi merek ---
def norm_brand(x):
    if pd.isna(x):
        return "UNKNOWN"
    return re.sub(r"[.\s]+", "", str(x).strip().lower())

k["merek"] = k["kategori_nama"].map(norm_brand)

# --- Normalisasi nama produk ---
k["nama_norm"] = k["barang_nama"].str.strip().str.lower()

report = {}
report["n_baris"] = int(len(k))
report["n_faktur"] = int(k["d_jual_nofak"].nunique())
report["nilai_total_jt"] = round(float(k["jumlah"].sum()) / 1e6, 1)

# --- 1. Normalisasi merek ---
report["merek"] = {
    "n_merek_asli": int(k["kategori_nama"].nunique()),
    "n_merek_ternormalisasi": int(k["merek"].nunique()),
    "selisih_pecah": int(k["kategori_nama"].nunique() - k["merek"].nunique()),
    "merek_top": k["merek"].value_counts().head(10).to_dict(),
}

# --- 2. Qty per satuan (produk berbeda, tidak dijumlahkan) ---
sat_rows = []
for sat in ["Roll", "Box", "Pack"]:
    sub = k[k["satuan_nama"] == sat]
    sat_rows.append({
        "satuan": sat,
        "n_baris": int(len(sub)),
        "qty": float(sub["d_jual_qty"].sum()),
        "nilai_jt": round(float(sub["jumlah"].sum()) / 1e6, 1),
        "produk_utama": sub["nama_norm"].value_counts().head(3).to_dict(),
    })
report["qty_per_satuan"] = sat_rows

# --- 3. Qty per produk (nama x satuan) ---
produk = k.groupby(["nama_norm", "satuan_nama"]).agg(
    n_baris=("jumlah", "count"),
    qty=("d_jual_qty", "sum"),
    nilai_jt=("jumlah", lambda x: round(x.sum() / 1e6, 1)),
    hrg_med=("d_jual_qty", lambda x: np.nan),
).reset_index()
# harga median per qty
hp = k.groupby(["nama_norm", "satuan_nama"])["jumlah"].sum() / \
     k.groupby(["nama_norm", "satuan_nama"])["d_jual_qty"].sum()
produk["harga_per_qty"] = produk.set_index(["nama_norm", "satuan_nama"]).index.map(
    lambda ix: round(hp.get(ix, 0), 0))
produk = produk.sort_values("nilai_jt", ascending=False)
report["qty_per_produk"] = produk.to_dict(orient="records")

# --- 4. Deret bulanan nilai (konsisten) ---
monthly = k.groupby("month").agg(
    nilai_jt=("jumlah", lambda x: round(x.sum() / 1e6, 1)),
    n_baris=("jumlah", "count"),
).reset_index()
report["monthly_nilai"] = monthly.to_dict(orient="records")

report["summary"] = {
    "catatan": (
        "Roll/Box/Pack kassa adalah PRODUK BERBEDA (Kassa roll 40x80, Kassa "
        "Steril 16x16, Gauze Swab 10x10), bukan isi box yang sama. Tidak ada "
        "konversi satuan yang valid. Qty dianalisis per produk (nama x satuan) "
        "+ nilai rupiah konsisten. Merek (kategori_nama) dinormalisasi."
    ),
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)

# --- Cetak ---
print("=" * 70)
print("ANALISIS KASSA (normalisasi merek + qty per produk)")
print("=" * 70)
print(f"Baris: {report['n_baris']:,} | Faktur: {report['n_faktur']:,} | Nilai: {report['nilai_total_jt']:,.1f} jt")

print("\n[1] NORMALISASI MEREK")
mr = report["merek"]
print(f"  merek asli {mr['n_merek_asli']} -> {mr['n_merek_ternormalisasi']} (pecah {mr['selisih_pecah']})")
print("  top:", ", ".join(f"{k}({v})" for k, v in list(mr['merek_top'].items())[:6]))

print("\n[2] QTY PER SATUAN (produk berbeda)")
for r in sat_rows:
    print("  {:<5} baris={:>4,} qty={:>9,.0f} nilai={:>7,.1f} jt | produk: {}".format(
        r["satuan"], r["n_baris"], r["qty"], r["nilai_jt"],
        ", ".join(f"{k}({v})" for k, v in list(r["produk_utama"].items())[:2])))

print("\n[3] QTY PER PRODUK (nama x satuan, top 10 nilai)")
for r in produk.head(10).to_dict(orient="records"):
    print("  {:<22} {:<5} baris={:>4,} qty={:>9,.0f} nilai={:>7,.1f} jt hrg/qty={:,.0f}".format(
        r["nama_norm"], r["satuan_nama"], r["n_baris"], r["qty"], r["nilai_jt"], r["harga_per_qty"]))

print("\nSelesai. Disimpan ke", OUT)
