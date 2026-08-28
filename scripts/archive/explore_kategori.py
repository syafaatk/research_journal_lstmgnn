# -*- coding: utf-8 -*-
"""Eksplorasi kategori produk untuk menilai opsi agregasi produk."""
import pandas as pd
import numpy as np

DATA = r"E:\Download\Jurnal\view_penjualan_detail data hingga oktober.xlsx"
df = pd.read_excel(DATA)

print("=== kategori_nama ===")
print(f"unique: {df['kategori_nama'].nunique()}")
vc = df["kategori_nama"].value_counts()
for k, v in vc.items():
    print(f"  {k}: {v} baris ({v/len(df):.1%})")

print()
print("=== kategori per faktur ===")
per_inv = df.groupby("d_jual_nofak")["kategori_nama"].nunique()
print(f"  median {per_inv.median():.0f}, max {per_inv.max():.0f}")

print()
print("=== qty agregat per faktur vs nilai ===")
inv = df.groupby("d_jual_nofak").agg(
    qty=("d_jual_qty", "sum"),
    nilai=("jual_total_fak", "first"),
    n_item=("d_jual_id", "count"),
    n_produk=("d_barang_brg_id", "nunique"),
)
print(f"  korelasi qty vs nilai: {inv['qty'].corr(inv['nilai']):.4f}")
print(f"  korelasi n_item vs nilai: {inv['n_item'].corr(inv['nilai']):.4f}")
print(f"  korelasi n_produk vs nilai: {inv['n_produk'].corr(inv['nilai']):.4f}")

print()
print("=== Panel harian: qty agregat per region-hari ===")
df2 = df.copy()
df2["region"] = df2["wilayah"]
df2["period"] = df2["tanggal"].dt.to_period("D")
panel_qty = df2.pivot_table(index="period", columns="region", values="d_jual_qty", aggfunc="sum")
panel_val = df2.pivot_table(index="period", columns="region", values="jual_total_fak", aggfunc="sum")
print(f"  panel qty: {panel_qty.shape}")
# korelasi panel qty vs nilai (pooled)
q = panel_qty.values.ravel(); v = panel_val.values.ravel()
print(f"  korelasi pooled qty vs nilai: {np.corrcoef(q, v)[0,1]:.4f}")
# korelasi per region
corrs = [np.corrcoef(panel_qty[c].values, panel_val[c].values)[0,1] for c in panel_qty.columns]
print(f"  korelasi per region: min {min(corrs):.3f}, median {np.median(corrs):.3f}, max {max(corrs):.3f}")

print()
print("=== Apakah qty menambah sinyal di atas nol? ===")
# region-hari dengan nilai>0 tapi qty kecil vs besar
mask = v > 0
print(f"  region-hari nilai>0: {mask.sum()}")
print(f"  qty pada region-hari nilai>0: median {np.median(q[mask]):.0f}, p90 {np.percentile(q[mask], 90):.0f}")