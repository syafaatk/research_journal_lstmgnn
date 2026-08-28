# -*- coding: utf-8 -*-
"""Analisis plester: normalisasi merek + qty per produk (nama x satuan).

Temuan eksplorasi:
  - Satuan: Box (48), Roll (15).
  - Merek (kategori_nama) kotor: OneMed vs OneMed., BSN vs .BSN. -> normalisasi.
  - Roll/Box adalah produk/format BERBEDA (Hypafix roll, Zinc Oxide box),
    bukan isi box yang sama. Tidak ada konversi satuan yang valid.
  - Zinc Oxide 7,5cm x 4,5m muncul di kedua satuan (Roll ~46.850, Box ~255.515,
    rasio ~5,5x) -> kemungkinan box berisi beberapa roll, tapi tidak konsisten
    dan tidak bisa dipastikan dari data. Qty dianalisis per produk.

Output: results/plester_analysis.json
"""
import json
import os
import re

import numpy as np
import pandas as pd

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(BASE, "results", "plester_analysis.json")

df = pd.read_excel(DATA)
df["jumlah"] = pd.to_numeric(df["jumlah"], errors="coerce").fillna(0.0)
df["d_jual_qty"] = pd.to_numeric(df["d_jual_qty"], errors="coerce").fillna(0.0)
df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
df["month"] = df["tanggal"].dt.to_period("M").astype(str)

k = df[df["barang_nama"].str.contains(
    "plester|plaster|micropore|hypafix|leucoplast|band ?aid|hansaplast",
    case=False, na=False)].copy()

# --- Normalisasi merek ---
def norm_brand(x):
    if pd.isna(x):
        return "UNKNOWN"
    return re.sub(r"[.\s]+", "", str(x).strip().lower())

k["merek"] = k["kategori_nama"].map(norm_brand)
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

# --- 2. Qty per satuan ---
sat_rows = []
for sat in ["Box", "Roll"]:
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
hp = k.groupby(["nama_norm", "satuan_nama"])["jumlah"].sum() / \
     k.groupby(["nama_norm", "satuan_nama"])["d_jual_qty"].sum()
produk = k.groupby(["nama_norm", "satuan_nama"]).agg(
    n_baris=("jumlah", "count"),
    qty=("d_jual_qty", "sum"),
    nilai_jt=("jumlah", lambda x: round(x.sum() / 1e6, 1)),
).reset_index()
produk["harga_per_qty"] = produk.set_index(["nama_norm", "satuan_nama"]).index.map(
    lambda ix: round(hp.get(ix, 0), 0))
produk = produk.sort_values("nilai_jt", ascending=False)
report["qty_per_produk"] = produk.to_dict(orient="records")

# --- 4. Deret bulanan nilai ---
monthly = k.groupby("month").agg(
    nilai_jt=("jumlah", lambda x: round(x.sum() / 1e6, 1)),
    n_baris=("jumlah", "count"),
).reset_index()
report["monthly_nilai"] = monthly.to_dict(orient="records")

report["summary"] = {
    "catatan": (
        "Roll/Box plester adalah produk/format BERBEDA (Hypafix roll, Zinc "
        "Oxide box), bukan isi box yang sama. Tidak ada konversi satuan yang "
        "valid. Qty dianalisis per produk (nama x satuan) + nilai rupiah "
        "konsisten. Merek (kategori_nama) dinormalisasi (OneMed./.BSN. dll)."
    ),
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)

# --- Cetak ---
print("=" * 70)
print("ANALISIS PLESTER (normalisasi merek + qty per produk)")
print("=" * 70)
print(f"Baris: {report['n_baris']:,} | Faktur: {report['n_faktur']:,} | Nilai: {report['nilai_total_jt']:,.1f} jt")

print("\n[1] NORMALISASI MEREK")
mr = report["merek"]
print(f"  merek asli {mr['n_merek_asli']} -> {mr['n_merek_ternormalisasi']} (pecah {mr['selisih_pecah']})")
print("  top:", ", ".join(f"{k}({v})" for k, v in list(mr['merek_top'].items())[:8]))

print("\n[2] QTY PER SATUAN (produk berbeda)")
for r in sat_rows:
    print("  {:<5} baris={:>4,} qty={:>9,.0f} nilai={:>7,.1f} jt | produk: {}".format(
        r["satuan"], r["n_baris"], r["qty"], r["nilai_jt"],
        ", ".join(f"{k}({v})" for k, v in list(r["produk_utama"].items())[:2])))

print("\n[3] QTY PER PRODUK (nama x satuan, top 10 nilai)")
for r in produk.head(10).to_dict(orient="records"):
    print("  {:<32} {:<5} baris={:>4,} qty={:>9,.0f} nilai={:>7,.1f} jt hrg/qty={:,.0f}".format(
        r["nama_norm"], r["satuan_nama"], r["n_baris"], r["qty"], r["nilai_jt"], r["harga_per_qty"]))

print("\nSelesai. Disimpan ke", OUT)
