# -*- coding: utf-8 -*-
"""Analisis handscoon: normalisasi merek + qty per satuan terpisah.

Temuan eksplorasi:
  - `kategori_nama` sebenarnya berisi MEREK (GP Care, Cosmomed, Sritrang,
    Remedi, Meddis, Gammex, dll), bukan kategori produk. Sangat kotor:
    banyak typo/casing/spasi (Meddis/Meddis./Meddis.., Shamrock/.Shamrock.,
    Remedi/Remedi./REMEDI, Maxter/maxter/.Maxter./MaXtEr).
  - `barang_satuan_id` duplikat: Pair = id 20 & 23, Pack = id 12 & 28.
  - Satuan campur: Box (1132), Pair (500), Pack (12). Qty mentah antar
    satuan TIDAK bisa dijumlahkan langsung.
  - `barang_spesifikasi` hanya berisi ukuran (S/M/L, No. 6.5-8), TIDAK
    menyebut isi per box. Isi box bervariasi per merek (Gammex steril ~50
    pcs, non-steril ~100 pcs) dan TIDAK diketahui dari data.

Keputusan (28 Agu 2026): karena isi box tidak diketahui dari data dan
bervariasi per merek, qty handscoon dianalisis PER SATUAN TERPISAH (Box,
Pair, Pack) + nilai rupiah konsisten. TIDAK memaksa konversi box->pcs yang
berisiko mengarang. Konversi hanya bisa dilakukan jika user menyediakan isi
per kemasan per merek.

Output: results/handscoon_analysis.json
"""
import json
import os
import re

import numpy as np
import pandas as pd

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(BASE, "results", "handscoon_analysis.json")

df = pd.read_excel(DATA)
df["jumlah"] = pd.to_numeric(df["jumlah"], errors="coerce").fillna(0.0)
df["d_jual_qty"] = pd.to_numeric(df["d_jual_qty"], errors="coerce").fillna(0.0)
df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
df["month"] = df["tanggal"].dt.to_period("M").astype(str)

h = df[df["barang_nama"].str.contains(
    "handscoon|sarung tangan|glove", case=False, na=False)].copy()

# --- Normalisasi merek (kategori_nama) ---
def norm_brand(s):
    if pd.isna(s):
        return "UNKNOWN"
    return re.sub(r"[.\s]+", "", str(s).strip().lower())

h["merek"] = h["kategori_nama"].map(norm_brand)

# --- Pemetaan barang_satuan_id -> satuan_nama (duplikat id) ---
sat_map = h.groupby("barang_satuan_id")["satuan_nama"].agg(
    lambda x: x.mode().iloc[0] if len(x) else "?").to_dict()
h["satuan_kanonik"] = h["barang_satuan_id"].map(sat_map)

report = {}
report["n_baris"] = int(len(h))
report["n_faktur"] = int(h["d_jual_nofak"].nunique())

# --- 1. Normalisasi merek ---
report["merek"] = {
    "n_merek_asli": int(h["kategori_nama"].nunique()),
    "n_merek_ternormalisasi": int(h["merek"].nunique()),
    "selisih_pecah": int(h["kategori_nama"].nunique() - h["merek"].nunique()),
    "merek_top": h["merek"].value_counts().head(15).to_dict(),
}

# --- 2. Qty per satuan (terpisah, tidak dijumlahkan) ---
sat_rows = []
for sat in ["Box", "Pair", "Pack"]:
    sub = h[h["satuan_nama"] == sat]
    sat_rows.append({
        "satuan": sat,
        "n_baris": int(len(sub)),
        "n_faktur": int(sub["d_jual_nofak"].nunique()),
        "total_qty": float(sub["d_jual_qty"].sum()),
        "total_nilai_jt": round(float(sub["jumlah"].sum()) / 1e6, 1),
    })
report["qty_per_satuan"] = sat_rows

# --- 3. Nilai rupiah total (konsisten antar satuan) ---
report["nilai_total_jt"] = round(float(h["jumlah"].sum()) / 1e6, 1)

# --- 4. Qty per satuan per merek (untuk konteks) ---
merek_sat = h.groupby(["merek", "satuan_nama"]).agg(
    n_baris=("jumlah", "count"),
    qty=("d_jual_qty", "sum"),
    nilai_jt=("jumlah", lambda x: round(x.sum() / 1e6, 1)),
).reset_index()
report["qty_per_merek_satuan"] = merek_sat.to_dict(orient="records")

# --- 5. Deret bulanan nilai (konsisten) ---
monthly = h.groupby("month").agg(
    nilai_jt=("jumlah", lambda x: round(x.sum() / 1e6, 1)),
    n_baris=("jumlah", "count"),
).reset_index()
report["monthly_nilai"] = monthly.to_dict(orient="records")

report["summary"] = {
    "catatan": (
        "kategori_nama berisi MEREK (bukan kategori) dan kotor (typo/casing/spasi), "
        "dinormalisasi. barang_satuan_id duplikat (Pair=20/23, Pack=12/28) dipetakan. "
        "Isi box handscoon TIDAK diketahui dari data dan bervariasi per merek, "
        "jadi qty dianalisis per satuan terpisah + nilai rupiah konsisten. "
        "Konversi box->pcs hanya jika user menyediakan isi per kemasan per merek."
    ),
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)

# --- Cetak ---
print("=" * 70)
print("ANALISIS HANDSCOON (normalisasi merek + qty per satuan)")
print("=" * 70)
print(f"Baris: {report['n_baris']:,} | Faktur: {report['n_faktur']:,} | Nilai total: {report['nilai_total_jt']:,.1f} jt")

print("\n[1] NORMALISASI MEREK (kategori_nama)")
mr = report["merek"]
print(f"  merek asli {mr['n_merek_asli']} -> ternormalisasi {mr['n_merek_ternormalisasi']} (pecah {mr['selisih_pecah']})")
print("  top merek:", ", ".join(f"{k}({v})" for k, v in list(mr['merek_top'].items())[:10]))

print("\n[2] QTY PER SATUAN (terpisah, tidak dijumlahkan)")
for r in sat_rows:
    print("  {:<5} baris={:>5,} faktur={:>4,} qty={:>10,.0f} nilai={:>8,.1f} jt".format(
        r["satuan"], r["n_baris"], r["n_faktur"], r["total_qty"], r["total_nilai_jt"]))

print("\n[3] QTY PER MEREK x SATUAN (top 15 nilai)")
ms = sorted(report["qty_per_merek_satuan"], key=lambda x: -x["nilai_jt"])[:15]
for r in ms:
    print("  {:<14} {:<5} baris={:>4,} qty={:>9,.0f} nilai={:>7,.1f} jt".format(
        r["merek"], r["satuan_nama"], r["n_baris"], r["qty"], r["nilai_jt"]))

print("\nSelesai. Disimpan ke", OUT)
