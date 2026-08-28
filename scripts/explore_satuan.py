"""Langkah 1: Eksplorasi kolom satuan_nama - distribusi, nilai unik, hubungan dengan qty."""
import pandas as pd, numpy as np, json, os
from collections import Counter

DATA = r"E:\Download\Jurnal\data\view_penjualan_detail data hingga oktober.xlsx"
OUT  = r"E:\Download\Jurnal\results\satuan_analysis.json"

df = pd.read_excel(DATA)
print(f"Baris: {len(df)}, Kolom total: {len(df.columns)}")
print(f"\nSemua kolom: {list(df.columns)}")

# Cek kolom terkait satuan
satuan_cols = [c for c in df.columns if 'satuan' in c.lower() or 'unit' in c.lower() or 'qty' in c.lower()]
print(f"\nKolom terkait satuan/qty: {satuan_cols}")

# --- Distribusi satuan_nama ---
print("\n" + "="*80)
print("DISTRIBUSI satuan_nama (semua nilai unik)")
print("="*80)
sat_counts = df['satuan_nama'].value_counts(dropna=False)
for s, cnt in sat_counts.items():
    print(f"  [{cnt:>5} baris] {s}")

print(f"\nTotal nilai unik satuan_nama: {df['satuan_nama'].nunique(dropna=True)}")

# --- Hubungan satuan_nama dengan jumlah/qty ---
print("\n" + "="*80)
print("HUBUNGAN satuan_nama dengan qty & nilai")
print("="*80)

# Pastikan numeric
df['jumlah'] = pd.to_numeric(df['jumlah'], errors='coerce')
df['d_jual_qty'] = pd.to_numeric(df['d_jual_qty'], errors='coerce')

# Hitung rata-rata jumlah per d_jual_qty (= harga satuan implisit)
df['harga_per_qty'] = df['jumlah'] / df['d_jual_qty'].replace(0, np.nan)

agg = df.groupby('satuan_nama').agg(
    n_baris=('jumlah', 'count'),
    n_faktur=('d_jual_nofak', 'nunique'),
    total_nilai=('jumlah', 'sum'),
    total_qty=('d_jual_qty', 'sum'),
    mean_nilai_per_baris=('jumlah', 'mean'),
    mean_qty_per_baris=('d_jual_qty', 'mean'),
    mean_harga_per_qty=('harga_per_qty', 'median'),
    n_produk=('barang_nama', 'nunique'),
).sort_values('total_nilai', ascending=False)

print(f"\n{'satuan_nama':>30} | {'baris':>6} | {'faktur':>6} | {'nilai(M)':>10} | {'qty':>12} | {'harga/unit':>12} | {'produk':>5}")
print("-"*110)
for idx, r in agg.iterrows():
    print(f"{str(idx):>30} | {r['n_baris']:>6,.0f} | {r['n_faktur']:>6,.0f} | {r['total_nilai']/1e6:>10,.1f} | {r['total_qty']:>12,.0f} | {r['mean_harga_per_qty']:>12,.0f} | {r['n_produk']:>5,.0f}")

# --- Top produk per satuan (untuk memahami konteks) ---
print("\n" + "="*80)
print("TOP 3 PRODUK PER SATUAN (oleh jumlah baris)")
print("="*80)
for s in sat_counts.head(10).index:
    sub = df[df['satuan_nama'] == s]
    top = sub.groupby('barang_nama')['jumlah'].sum().nlargest(3)
    print(f"\n  [{s}] (total {len(sub):,} baris, {sub['jumlah'].sum()/1e6:,.1f} M IDR)")
    for pname, val in top.items():
        print(f"    - {pname}: {val/1e6:,.1f} M")

# --- Cek apakah qty sudah dalam satuan dasar atau satuan packing ---
print("\n" + "="*80)
print("CEK: qty vs satuan (apakah 1 box = qty 1 atau qty 20?)")
print("="*80)
for s in sat_counts.head(8).index:
    sub = df[df['satuan_nama'] == s].dropna(subset=['d_jual_qty'])
    if len(sub) > 0:
        qty_vals = sub['d_jual_qty'].values
        print(f"\n  [{s}]")
        print(f"    qty: min={qty_vals.min():.0f}, median={np.median(qty_vals):.0f}, max={qty_vals.max():.0f}")
        print(f"    qty==1: {(qty_vals==1).sum()} baris ({(qty_vals==1).mean()*100:.1f}%)")
        print(f"    qty<=10: {(qty_vals<=10).sum()} baris ({(qty_vals<=10).mean()*100:.1f}%)")
        print(f"    qty>100: {(qty_vals>100).sum()} baris ({(qty_vals>100).mean()*100:.1f}%)")

# --- Simpan summary ---
summary = {
    'n_satuan_unik': int(df['satuan_nama'].nunique(dropna=True)),
    'distribusi': {str(k): int(v) for k, v in sat_counts.items()},
    'agg_per_satuan': [],
}
for idx, r in agg.iterrows():
    summary['agg_per_satuan'].append({
        'satuan': str(idx),
        'n_baris': int(r['n_baris']),
        'n_faktur': int(r['n_faktur']),
        'total_nilai_jt': round(float(r['total_nilai']/1e6), 1),
        'total_qty': float(r['total_qty']),
        'harga_per_qty_median': float(r['mean_harga_per_qty']) if pd.notna(r['mean_harga_per_qty']) else None,
        'n_produk': int(r['n_produk']),
    })

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved: {OUT}")
