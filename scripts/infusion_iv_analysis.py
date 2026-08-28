# -*- coding: utf-8 -*-
"""Analisis infusion_set & iv_catheter: normalisasi merek + qty per ukuran.

Temuan eksplorasi:
  - infusion_set: hampir semua Pcs (998) + 1 Unit (outlier 1,7jt, di-exclude).
    Harga Pcs median 6.448. Merek kotor: Meddis.., G E A, Healtcare.
  - iv_catheter: semua Pcs (835). Harga median 7.200. Merek SANGAT kotor:
    Gea./GEA/G E A/GeA (=GEA), HEALTHCARE., Meddis../Meddis.
  - Keduanya sudah dalam satuan Pcs -> TIDAK perlu konversi. Hanya normalisasi
    merek + analisis per ukuran (G/cc).

Output: results/infusion_iv_analysis.json
"""
import json
import os
import re

import numpy as np
import pandas as pd

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(BASE, "results", "infusion_iv_analysis.json")

df = pd.read_excel(DATA)
df["jumlah"] = pd.to_numeric(df["jumlah"], errors="coerce").fillna(0.0)
df["d_jual_qty"] = pd.to_numeric(df["d_jual_qty"], errors="coerce").fillna(0.0)
df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
df["month"] = df["tanggal"].dt.to_period("M").astype(str)

def norm_brand(x):
    if pd.isna(x):
        return "UNKNOWN"
    return re.sub(r"[.\s]+", "", str(x).strip().lower())

def analyze(key, label, exclude_unit=True):
    k = df[df["barang_nama"].str.contains(key, case=False, na=False)].copy()
    if exclude_unit:
        k = k[k["satuan_nama"] != "Unit"]  # exclude outlier Unit
    k["merek"] = k["kategori_nama"].map(norm_brand)
    k["nama_norm"] = k["barang_nama"].str.strip().str.lower()

    # ukuran: G (gauge) untuk catheter, cc/ml untuk infusion
    k["ukuran"] = k["barang_spesifikasi"].astype(str).str.extract(
        r"(\d+)\s*[Gg]")[0]
    k["ukuran"] = k["ukuran"].fillna(
        k["barang_spesifikasi"].astype(str).str.extract(
            r"(\d+)\s*(?:cc|ml)", flags=re.IGNORECASE)[0])

    r = {}
    r["label"] = label
    r["n_baris"] = int(len(k))
    r["n_faktur"] = int(k["d_jual_nofak"].nunique())
    r["nilai_total_jt"] = round(float(k["jumlah"].sum()) / 1e6, 1)
    r["qty_total"] = float(k["d_jual_qty"].sum())

    r["merek"] = {
        "n_merek_asli": int(k["kategori_nama"].nunique()),
        "n_merek_ternormalisasi": int(k["merek"].nunique()),
        "selisih_pecah": int(k["kategori_nama"].nunique() - k["merek"].nunique()),
        "merek_top": k["merek"].value_counts().head(10).to_dict(),
    }

    # per merek
    merek = k.groupby("merek").agg(
        n_baris=("jumlah", "count"),
        qty=("d_jual_qty", "sum"),
        nilai_jt=("jumlah", lambda x: round(x.sum() / 1e6, 1)),
    ).reset_index().sort_values("nilai_jt", ascending=False)
    r["qty_per_merek"] = merek.to_dict(orient="records")

    # per ukuran
    uk = k.groupby("ukuran").agg(
        n_baris=("jumlah", "count"),
        qty=("d_jual_qty", "sum"),
        nilai_jt=("jumlah", lambda x: round(x.sum() / 1e6, 1)),
    ).reset_index().sort_values("qty", ascending=False)
    r["qty_per_ukuran"] = uk.to_dict(orient="records")

    # bulanan
    monthly = k.groupby("month").agg(
        qty=("d_jual_qty", "sum"),
        nilai_jt=("jumlah", lambda x: round(x.sum() / 1e6, 1)),
    ).reset_index()
    r["monthly"] = monthly.to_dict(orient="records")
    return r

report = {
    "infusion_set": analyze("infusion ?set|infus set|infus-set", "INFUSION_SET"),
    "iv_catheter": analyze("iv ?catheter|intravenous catheter|abbocath|venflon|braunule", "IV_CATHETER"),
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)

# --- Cetak ---
for label, r in report.items():
    print("=" * 70)
    print(f"ANALISIS {r['label']}")
    print("=" * 70)
    print(f"Baris: {r['n_baris']:,} | Faktur: {r['n_faktur']:,} | Qty: {r['qty_total']:,.0f} | Nilai: {r['nilai_total_jt']:,.1f} jt")
    mr = r["merek"]
    print(f"Merek: {mr['n_merek_asli']} -> {mr['n_merek_ternormalisasi']} (pecah {mr['selisih_pecah']})")
    print("  top:", ", ".join(f"{k}({v})" for k, v in list(mr['merek_top'].items())[:8]))
    print("Per merek (top 5 nilai):")
    for m in r["qty_per_merek"][:5]:
        print("  {:<14} baris={:>4,} qty={:>9,.0f} nilai={:>7,.1f} jt".format(
            m["merek"], m["n_baris"], m["qty"], m["nilai_jt"]))
    print("Per ukuran (top 8 qty):")
    for u in r["qty_per_ukuran"][:8]:
        print("  {:<5} baris={:>4,} qty={:>9,.0f} nilai={:>7,.1f} jt".format(
            u["ukuran"] or "?", u["n_baris"], u["qty"], u["nilai_jt"]))
    print()

print("Selesai. Disimpan ke", OUT)
