# -*- coding: utf-8 -*-
"""Eksplorasi d_jual_qty dan d_barang_brg_id untuk menilai kelayakan pemakaian."""
import pandas as pd
import numpy as np

DATA = r"E:\Download\Jurnal\view_penjualan_detail data hingga oktober.xlsx"
df = pd.read_excel(DATA)
print("Kolom:", list(df.columns))
print("Shape:", df.shape)
print()

# Cek keberadaan kolom
for col in ["d_jual_qty", "d_barang_brg_id", "d_barang_nama", "d_barang_brg_nama"]:
    if col in df.columns:
        print(f"=== {col} ===")
        print(f"  non-null: {df[col].notna().sum()} / {len(df)}")
        print(f"  unique: {df[col].nunique()}")
        if df[col].dtype == object:
            print(f"  contoh: {df[col].dropna().unique()[:5]}")
        else:
            print(f"  min/median/max: {df[col].min()} / {df[col].median()} / {df[col].max()}")
        print()

# Hubungan qty vs nilai
if "d_jual_qty" in df.columns and "jual_total_fak" in df.columns:
    q = df[["d_jual_qty", "jual_total_fak"]].dropna()
    print("=== Hubungan qty vs jual_total_fak ===")
    print(f"  korelasi Pearson: {q['d_jual_qty'].corr(q['jual_total_fak']):.4f}")
    print(f"  qty==0: {(q['d_jual_qty']==0).mean():.2%} | qty<=0: {(q['d_jual_qty']<=0).mean():.2%}")
    print(f"  qty>0 & nilai>0: {((q['d_jual_qty']>0)&(q['jual_total_fak']>0)).mean():.2%}")
    print(f"  qty>0 & nilai==0: {((q['d_jual_qty']>0)&(q['jual_total_fak']==0)).mean():.2%}")
    print(f"  qty==0 & nilai>0: {((q['d_jual_qty']==0)&(q['jual_total_fak']>0)).mean():.2%}")
    # harga satuan implisit
    mask = (q["d_jual_qty"] > 0) & (q["jual_total_fak"] > 0)
    unit = q.loc[mask, "jual_total_fak"] / q.loc[mask, "d_jual_qty"]
    print(f"  harga satuan implisit (qty>0,nilai>0): median {unit.median():,.0f}, "
          f"p90 {unit.quantile(0.9):,.0f}, max {unit.max():,.0f}")

# Produk unik per faktur
if "d_barang_brg_id" in df.columns:
    print()
    print("=== Produk per faktur ===")
    per_inv = df.groupby("d_jual_nofak")["d_barang_brg_id"].nunique()
    print(f"  baris per faktur: median {per_inv.median():.0f}, max {per_inv.max():.0f}")
    print(f"  faktur 1 produk: {(per_inv==1).mean():.2%}")
    # produk teratas
    top = df["d_barang_brg_id"].value_counts()
    print(f"  top-5 produk: {top.head(5).to_dict()}")
    print(f"  produk dengan 1 transaksi: {(top==1).mean():.2%} dari {len(top)} produk")