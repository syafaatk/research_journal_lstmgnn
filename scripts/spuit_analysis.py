# -*- coding: utf-8 -*-
"""Analisis spuit: normalisasi merek + konversi qty ke pcs.

Temuan eksplorasi:
  - Satuan: Pcs (3.234) dan Box (243). barang_satuan_id Pcs = 8 & 21.
  - Merek (kategori_nama): Top Point, Cosmomed, Kimsmed, Terumo, OneMed.
    Kotor (Terumo./TerumO/.Terumo.).
  - Harga per qty: Pcs median 975, Box median 89.800 -> rasio ~90-100x.
    Konsisten dengan 1 box spuit = 100 pcs (standar industri).
  - Produk khusus (Bioglue Pre-Filled Syringe, insulin, tuberculin) di-exclude
    karena isi box berbeda (pre-filled, bukan spuit kosong).

Keputusan (28 Agu 2026): konversi box->pcs dengan isi_box = 100 pcs/box
untuk spuit biasa (1cc-50cc). Produk pre-filled/insulin/tuberculin di-exclude
dari konversi qty (dianalisis terpisah).

Output: results/spuit_analysis.json
"""
import json
import os
import re

import numpy as np
import pandas as pd

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(BASE, "results", "spuit_analysis.json")

BOX_CONTENT = 100  # pcs per box spuit biasa

df = pd.read_excel(DATA)
df["jumlah"] = pd.to_numeric(df["jumlah"], errors="coerce").fillna(0.0)
df["d_jual_qty"] = pd.to_numeric(df["d_jual_qty"], errors="coerce").fillna(0.0)
df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
df["month"] = df["tanggal"].dt.to_period("M").astype(str)

s = df[df["barang_nama"].str.contains("spuit|syringe|syiringe", case=False, na=False)].copy()

# Produk khusus yang di-exclude dari konversi qty
SPECIAL = s["barang_nama"].str.contains(
    "pre-filled|bioglue|insulin|tuberculin", case=False, na=False)
s_special = s[SPECIAL].copy()
s_main = s[~SPECIAL].copy()

# --- Normalisasi merek ---
def norm_brand(x):
    if pd.isna(x):
        return "UNKNOWN"
    return re.sub(r"[.\s]+", "", str(x).strip().lower())

s_main["merek"] = s_main["kategori_nama"].map(norm_brand)

# --- Konversi qty ke pcs ---
s_main["qty_pcs"] = np.where(
    s_main["satuan_nama"] == "Box",
    s_main["d_jual_qty"] * BOX_CONTENT,
    s_main["d_jual_qty"],
)

report = {}
report["n_baris"] = int(len(s))
report["n_faktur"] = int(s["d_jual_nofak"].nunique())
report["isi_box_pcs"] = BOX_CONTENT
report["n_baris_produk_khusus"] = int(len(s_special))
report["produk_khusus"] = s_special["barang_nama"].value_counts().head(10).to_dict()

# --- 1. Normalisasi merek ---
report["merek"] = {
    "n_merek_asli": int(s_main["kategori_nama"].nunique()),
    "n_merek_ternormalisasi": int(s_main["merek"].nunique()),
    "selisih_pecah": int(s_main["kategori_nama"].nunique() - s_main["merek"].nunique()),
    "merek_top": s_main["merek"].value_counts().head(10).to_dict(),
}

# --- 2. Qty total (pcs) ---
report["qty_total_pcs"] = float(s_main["qty_pcs"].sum())
report["nilai_total_jt"] = round(float(s_main["jumlah"].sum()) / 1e6, 1)

# --- 3. Qty per satuan ---
sat_rows = []
for sat in ["Box", "Pcs"]:
    sub = s_main[s_main["satuan_nama"] == sat]
    sat_rows.append({
        "satuan": sat,
        "n_baris": int(len(sub)),
        "qty_mentah": float(sub["d_jual_qty"].sum()),
        "qty_pcs": float(sub["qty_pcs"].sum()),
        "nilai_jt": round(float(sub["jumlah"].sum()) / 1e6, 1),
    })
report["qty_per_satuan"] = sat_rows

# --- 4. Qty per merek (pcs) ---
merek = s_main.groupby("merek").agg(
    n_baris=("jumlah", "count"),
    qty_pcs=("qty_pcs", "sum"),
    nilai_jt=("jumlah", lambda x: round(x.sum() / 1e6, 1)),
).reset_index().sort_values("nilai_jt", ascending=False)
report["qty_per_merek"] = merek.to_dict(orient="records")

# --- 5. Qty per ukuran cc (pcs) ---
s_main["cc"] = s_main["barang_spesifikasi"].astype(str).str.extract(
    r"(\d+)\s*cc", flags=re.IGNORECASE)[0]
cc = s_main.groupby("cc").agg(
    n_baris=("jumlah", "count"),
    qty_pcs=("qty_pcs", "sum"),
    nilai_jt=("jumlah", lambda x: round(x.sum() / 1e6, 1)),
).reset_index().sort_values("qty_pcs", ascending=False)
report["qty_per_cc"] = cc.to_dict(orient="records")

# --- 6. Deret bulanan qty (pcs) ---
monthly = s_main.groupby("month").agg(
    qty_pcs=("qty_pcs", "sum"),
    nilai_jt=("jumlah", lambda x: round(x.sum() / 1e6, 1)),
).reset_index()
report["monthly_pcs"] = monthly.to_dict(orient="records")

report["summary"] = {
    "isi_box_pcs": BOX_CONTENT,
    "catatan": (
        "Konversi box->pcs dengan isi 100 pcs/box (dari rasio harga box/pcs ~90-100x, "
        "konsisten standar industri). Produk khusus (pre-filled/insulin/tuberculin) "
        "di-exclude dari konversi qty. Merek (kategori_nama) dinormalisasi."
    ),
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)

# --- Cetak ---
print("=" * 70)
print("ANALISIS SPUIT (normalisasi merek + konversi box->pcs)")
print("=" * 70)
print(f"Baris: {report['n_baris']:,} | Faktur: {report['n_faktur']:,} | Nilai: {report['nilai_total_jt']:,.1f} jt")
print(f"Produk khusus (exclude konversi): {report['n_baris_produk_khusus']} baris")
print(f"Qty total (pcs): {report['qty_total_pcs']:,.0f}")

print("\n[1] NORMALISASI MEREK")
mr = report["merek"]
print(f"  merek asli {mr['n_merek_asli']} -> {mr['n_merek_ternormalisasi']} (pecah {mr['selisih_pecah']})")
print("  top:", ", ".join(f"{k}({v})" for k, v in list(mr['merek_top'].items())[:8]))

print("\n[2] QTY PER SATUAN")
for r in sat_rows:
    print("  {:<5} baris={:>5,} qty_mentah={:>9,.0f} qty_pcs={:>10,.0f} nilai={:>7,.1f} jt".format(
        r["satuan"], r["n_baris"], r["qty_mentah"], r["qty_pcs"], r["nilai_jt"]))

print("\n[3] QTY PER MEREK (pcs, top 8 nilai)")
for r in merek.head(8).to_dict(orient="records"):
    print("  {:<14} baris={:>5,} qty_pcs={:>10,.0f} nilai={:>7,.1f} jt".format(
        r["merek"], r["n_baris"], r["qty_pcs"], r["nilai_jt"]))

print("\n[4] QTY PER UKURAN cc (pcs)")
for r in cc.head(10).to_dict(orient="records"):
    print("  {:<5} baris={:>5,} qty_pcs={:>10,.0f} nilai={:>7,.1f} jt".format(
        r["cc"] or "?", r["n_baris"], r["qty_pcs"], r["nilai_jt"]))

print("\nSelesai. Disimpan ke", OUT)
