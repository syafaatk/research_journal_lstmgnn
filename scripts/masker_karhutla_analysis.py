# -*- coding: utf-8 -*-
"""Analisis qty masker dengan normalisasi satuan (Box vs Pcs).

Tujuan: menguji klaim 'masker TIDAK melonjak' saat karhutla El Nino 2023
(puncak Agu-Sep 2023) dengan memperhitungkan bahwa masker dijual campur
satuan Box dan Pcs. Qty mentah antar satuan TIDAK bisa dijumlahkan langsung
(1 box N95 isi ~20 pcs), jadi analisis dipisah per satuan + pembanding nilai
rupiah yang konsisten.

Output: results/masker_karhutla_analysis.json
"""
import json
import os

import numpy as np
import pandas as pd

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(BASE, "results", "masker_karhutla_analysis.json")

df = pd.read_excel(DATA)
df["jumlah"] = pd.to_numeric(df["jumlah"], errors="coerce").fillna(0.0)
df["d_jual_qty"] = pd.to_numeric(df["d_jual_qty"], errors="coerce").fillna(0.0)
df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
df["month"] = df["tanggal"].dt.to_period("M").astype(str)

# Masker: nama mengandung masker/n95/kn95/duckbill/face mask
mask = df["barang_nama"].str.contains(
    "masker|n95|kn95|duckbill|face mask", case=False, na=False
)
m = df[mask].copy()

# Periode karhutla El Nino 2023 (puncak Agu-Sep 2023) dan baseline
KARHUTLA_MONTHS = ["2023-08", "2023-09"]
# Baseline: semua bulan di luar jendela karhutla 2023, dan di luar puncak COVID 2021
BASELINE_MONTHS = [x for x in m["month"].unique()
                   if not x.startswith("2023-08") and not x.startswith("2023-09")
                   and not x.startswith("2021-07") and not x.startswith("2021-08")]

out = {}

# --- 1. Qty per satuan: karhutla vs baseline ---
sat_rows = []
for sat in ["Box", "Pcs"]:
    sub = m[m["satuan_nama"] == sat]
    in_win = sub[sub["month"].isin(KARHUTLA_MONTHS)]
    out_win = sub[sub["month"].isin(BASELINE_MONTHS)]
    in_qty = in_win["d_jual_qty"].sum()
    out_qty = out_win["d_jual_qty"].sum()
    n_in = len(in_win)
    n_out = len(out_win)
    # qty per bulan rata-rata (baseline dibagi jumlah bulan baseline)
    n_base_months = len(set(BASELINE_MONTHS))
    out_qty_per_month = out_qty / n_base_months if n_base_months else 0
    ratio = in_qty / out_qty_per_month if out_qty_per_month > 0 else None
    sat_rows.append({
        "satuan": sat,
        "qty_karhutla_2bln": float(in_qty),
        "n_baris_karhutla": int(n_in),
        "qty_baseline_per_bln": float(out_qty_per_month),
        "n_baris_baseline_per_bln": round(n_out / n_base_months, 1),
        "ratio_qty_karhutla_vs_baseline": round(ratio, 2) if ratio else None,
    })
out["qty_per_satuan"] = sat_rows

# --- 2. Nilai rupiah per satuan (konsisten antar satuan) ---
val_rows = []
for sat in ["Box", "Pcs"]:
    sub = m[m["satuan_nama"] == sat]
    in_win = sub[sub["month"].isin(KARHUTLA_MONTHS)]
    out_win = sub[sub["month"].isin(BASELINE_MONTHS)]
    in_val = in_win["jumlah"].sum()
    out_val_per_month = out_win["jumlah"].sum() / len(set(BASELINE_MONTHS))
    ratio = in_val / out_val_per_month if out_val_per_month > 0 else None
    val_rows.append({
        "satuan": sat,
        "nilai_karhutla_2bln_jt": round(float(in_val) / 1e6, 1),
        "nilai_baseline_per_bln_jt": round(float(out_val_per_month) / 1e6, 1),
        "ratio_nilai_karhutla_vs_baseline": round(ratio, 2) if ratio else None,
    })
out["nilai_per_satuan"] = val_rows

# --- 3. Deret bulanan qty masker (per satuan) untuk konteks temporal ---
monthly = m.groupby(["month", "satuan_nama"]).agg(
    qty=("d_jual_qty", "sum"),
    nilai_jt=("jumlah", lambda x: round(x.sum() / 1e6, 1)),
    n_baris=("jumlah", "count"),
).reset_index()
out["monthly"] = monthly.to_dict(orient="records")

# --- 4. Ringkasan: apakah ada lonjakan qty masker saat karhutla? ---
out["summary"] = {
    "periode_karhutla": KARHUTLA_MONTHS,
    "n_bulan_baseline": len(set(BASELINE_MONTHS)),
    "catatan": (
        "Qty Box dan Pcs TIDAK bisa dijumlahkan langsung (1 box N95 isi ~20 pcs). "
        "Analisis dipisah per satuan. Nilai rupiah konsisten antar satuan."
    ),
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

# --- Cetak ringkasan ---
print("=" * 70)
print("ANALISIS QTY MASKER vs KARHUTLA EL NINO 2023 (Agu-Sep 2023)")
print("=" * 70)
print("\n[1] QTY per satuan (karhutla 2 bln vs baseline per bulan):")
for r in sat_rows:
    print("  {:<5} qty_karhutla={:>10,.0f}  baseline/bln={:>9,.0f}  ratio={}".format(
        r["satuan"], r["qty_karhutla_2bln"], r["qty_baseline_per_bln"], r["ratio_qty_karhutla_vs_baseline"]))

print("\n[2] NILAI rupiah per satuan (konsisten):")
for r in val_rows:
    print("  {:<5} nilai_karhutla={:>8,.1f} jt  baseline/bln={:>8,.1f} jt  ratio={}".format(
        r["satuan"], r["nilai_karhutla_2bln_jt"], r["nilai_baseline_per_bln_jt"], r["ratio_nilai_karhutla_vs_baseline"]))

print("\n[3] Deret bulanan qty masker (per satuan):")
print("  {:<8} {:<5} {:>10} {:>10} {:>6}".format("bulan", "satuan", "qty", "nilai_jt", "baris"))
for r in sorted(monthly.to_dict(orient="records"), key=lambda x: (x["month"], x["satuan_nama"])):
    print("  {:<8} {:<5} {:>10,.0f} {:>10,.1f} {:>6}".format(
        r["month"], r["satuan_nama"], r["qty"], r["nilai_jt"], r["n_baris"]))

print("\nSelesai. Disimpan ke", OUT)
