# -*- coding: utf-8 -*-
"""Analisis qty masker dengan normalisasi satuan (Box vs Pcs) + konversi box->pcs.

Tujuan: menguji klaim 'masker TIDAK melonjak' saat karhutla El Nino 2023
(puncak Agu-Sep 2023) dengan memperhitungkan bahwa masker dijual campur
satuan Box dan Pcs, dan bahwa isi box bervariasi per tipe.

Isi box (dari user, 28 Agu 2026):
  - Surgical Face Mask / Face Mask / Masker medis (earloop, hijab, headloop,
    tie-on) dan Duckbill  -> 50 pcs/box
  - N95 / KN95 (8210, 1860, 8511, CIG 801, Pro, Medikal) -> 10 pcs/box

Qty total masker (pcs) = qty_box * isi_box + qty_pcs.
Output: results/masker_karhutla_analysis.json
"""
import json
import os

import numpy as np
import pandas as pd

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(BASE, "results", "masker_karhutla_analysis.json")

# Isi box per tipe masker (pcs per box)
BOX_CONTENT = {
    "surgical": 50,   # surgical face mask, face mask, masker medis, duckbill
    "n95": 10,        # N95 / KN95
}


def masker_type(name):
    n = name.strip().lower()
    if "n95" in n or "kn95" in n:
        return "n95"
    return "surgical"


df = pd.read_excel(DATA)
df["jumlah"] = pd.to_numeric(df["jumlah"], errors="coerce").fillna(0.0)
df["d_jual_qty"] = pd.to_numeric(df["d_jual_qty"], errors="coerce").fillna(0.0)
df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
df["month"] = df["tanggal"].dt.to_period("M").astype(str)

mask = df["barang_nama"].str.contains(
    "masker|n95|kn95|duckbill|face mask", case=False, na=False
)
m = df[mask].copy()
m["tipe"] = m["barang_nama"].map(masker_type)
m["isi_box"] = m["tipe"].map(BOX_CONTENT)
# qty dalam pcs: Box -> qty*isi_box ; Pcs -> qty
m["qty_pcs"] = np.where(
    m["satuan_nama"] == "Box",
    m["d_jual_qty"] * m["isi_box"],
    m["d_jual_qty"],
)

# Periode karhutla El Nino 2023 (puncak Agu-Sep 2023) dan baseline
KARHUTLA_MONTHS = ["2023-08", "2023-09"]
BASELINE_MONTHS = [x for x in m["month"].unique()
                   if not x.startswith("2023-08") and not x.startswith("2023-09")
                   and not x.startswith("2021-07") and not x.startswith("2021-08")]

out = {}

# --- 1. Qty total (pcs, sudah dikonversi) : karhutla vs baseline ---
in_win = m[m["month"].isin(KARHUTLA_MONTHS)]
out_win = m[m["month"].isin(BASELINE_MONTHS)]
n_base_months = len(set(BASELINE_MONTHS))
in_pcs = in_win["qty_pcs"].sum()
out_pcs_per_month = out_win["qty_pcs"].sum() / n_base_months
out["qty_total_pcs"] = {
    "qty_pcs_karhutla_2bln": float(in_pcs),
    "qty_pcs_baseline_per_bln": float(out_pcs_per_month),
    "ratio_karhutla_vs_baseline": round(in_pcs / out_pcs_per_month, 2) if out_pcs_per_month > 0 else None,
}

# --- 2. Qty per tipe (pcs) : karhutla vs baseline ---
tipe_rows = []
for t in ["surgical", "n95"]:
    sub = m[m["tipe"] == t]
    iw = sub[sub["month"].isin(KARHUTLA_MONTHS)]
    ow = sub[sub["month"].isin(BASELINE_MONTHS)]
    ip = iw["qty_pcs"].sum()
    op = ow["qty_pcs"].sum() / n_base_months
    tipe_rows.append({
        "tipe": t,
        "isi_box": BOX_CONTENT[t],
        "qty_pcs_karhutla_2bln": float(ip),
        "qty_pcs_baseline_per_bln": float(op),
        "ratio": round(ip / op, 2) if op > 0 else None,
    })
out["qty_per_tipe_pcs"] = tipe_rows

# --- 3. Qty per satuan (Box qty mentah, Pcs) : karhutla vs baseline ---
sat_rows = []
for sat in ["Box", "Pcs"]:
    sub = m[m["satuan_nama"] == sat]
    iw = sub[sub["month"].isin(KARHUTLA_MONTHS)]
    ow = sub[sub["month"].isin(BASELINE_MONTHS)]
    ip = iw["d_jual_qty"].sum()
    op = ow["d_jual_qty"].sum() / n_base_months
    sat_rows.append({
        "satuan": sat,
        "qty_karhutla_2bln": float(ip),
        "qty_baseline_per_bln": float(op),
        "ratio": round(ip / op, 2) if op > 0 else None,
    })
out["qty_per_satuan_mentah"] = sat_rows

# --- 4. Nilai rupiah per satuan (konsisten antar satuan) ---
val_rows = []
for sat in ["Box", "Pcs"]:
    sub = m[m["satuan_nama"] == sat]
    iw = sub[sub["month"].isin(KARHUTLA_MONTHS)]
    ow = sub[sub["month"].isin(BASELINE_MONTHS)]
    iv = iw["jumlah"].sum()
    ov = ow["jumlah"].sum() / n_base_months
    val_rows.append({
        "satuan": sat,
        "nilai_karhutla_2bln_jt": round(float(iv) / 1e6, 1),
        "nilai_baseline_per_bln_jt": round(float(ov) / 1e6, 1),
        "ratio": round(iv / ov, 2) if ov > 0 else None,
    })
out["nilai_per_satuan"] = val_rows

# --- 5. Deret bulanan qty total (pcs) untuk konteks temporal ---
monthly = m.groupby("month").agg(
    qty_pcs=("qty_pcs", "sum"),
    nilai_jt=("jumlah", lambda x: round(x.sum() / 1e6, 1)),
    n_baris=("jumlah", "count"),
).reset_index()
out["monthly_pcs"] = monthly.to_dict(orient="records")

out["summary"] = {
    "periode_karhutla": KARHUTLA_MONTHS,
    "n_bulan_baseline": n_base_months,
    "isi_box": BOX_CONTENT,
    "catatan": (
        "Qty total masker dihitung dalam pcs: qty_box * isi_box + qty_pcs. "
        "Isi box: surgical/duckbill=50, N95/KN95=10 (dari user 28 Agu 2026). "
        "Nilai rupiah konsisten antar satuan."
    ),
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

# --- Cetak ringkasan ---
print("=" * 70)
print("ANALISIS QTY MASKER (dikonversi ke pcs) vs KARHUTLA EL NINO 2023")
print("=" * 70)
print("\n[1] QTY TOTAL (pcs, box dikonversi):")
r = out["qty_total_pcs"]
print("  karhutla 2 bln = {:>12,.0f} pcs | baseline/bln = {:>12,.0f} pcs | ratio = {}".format(
    r["qty_pcs_karhutla_2bln"], r["qty_pcs_baseline_per_bln"], r["ratio_karhutla_vs_baseline"]))

print("\n[2] QTY per tipe (pcs):")
for r in tipe_rows:
    print("  {:<10} isi_box={:<3} karhutla={:>10,.0f} baseline/bln={:>10,.0f} ratio={}".format(
        r["tipe"], r["isi_box"], r["qty_pcs_karhutla_2bln"], r["qty_pcs_baseline_per_bln"], r["ratio"]))

print("\n[3] QTY per satuan (mentah):")
for r in sat_rows:
    print("  {:<5} karhutla={:>10,.0f} baseline/bln={:>9,.0f} ratio={}".format(
        r["satuan"], r["qty_karhutla_2bln"], r["qty_baseline_per_bln"], r["ratio"]))

print("\n[4] NILAI rupiah per satuan:")
for r in val_rows:
    print("  {:<5} karhutla={:>8,.1f} jt baseline/bln={:>8,.1f} jt ratio={}".format(
        r["satuan"], r["nilai_karhutla_2bln_jt"], r["nilai_baseline_per_bln_jt"], r["ratio"]))

print("\n[5] Deret bulanan qty total (pcs):")
print("  {:<8} {:>12} {:>10} {:>6}".format("bulan", "qty_pcs", "nilai_jt", "baris"))
for r in sorted(monthly.to_dict(orient="records"), key=lambda x: x["month"]):
    print("  {:<8} {:>12,.0f} {:>10,.1f} {:>6}".format(r["month"], r["qty_pcs"], r["nilai_jt"], r["n_baris"]))

print("\nSelesai. Disimpan ke", OUT)
