"""Analisis kolom satuan_nama: distribusi, hubungan dengan qty, potensi sebagai fitur."""
import pandas as pd
import numpy as np
import json, os

DATA = r"E:\Download\Jurnal\data\view_penjualan_detail data hingga oktober.xlsx"
OUT  = r"E:\Download\Jurnal\results\satuan_analysis.json"

df = pd.read_excel(DATA)
print(f"Baris: {len(df)}, Kolom: {list(df.columns)}")

# Pastikan kolom ada
assert 'satuan_nama' in df.columns, f"Kolom satuan_nama tidak ditemukan. Kolom: {list(df.columns)}"
assert 'jumlah' in df.columns
assert 'd_jual_qty' in df.columns

# --- 1. Distribusi satuan_nama ---
satuan_counts = df['satuan_nama'].value_counts(dropna=False)
satuan_nilai  = df.groupby('satuan_nama', dropna=False).agg(
    total_nilai=('jumlah', 'sum'),
    total_qty=('d_jual_qty', 'sum'),
    n_faktur=('jual_no_fak', 'nunique'),
    n_produk=('barang_nama', 'nunique'),
    mean_harga_per_unit=('jumlah', 'mean'),
).sort_values('total_nilai', ascending=False)

print("\n=== DISTRIBUSI SATUAN_NAMA (top 20) ===")
for idx, row in satuan_nilai.head(20).iterrows():
    print(f"  {str(idx):30s} | nilai={row['total_nilai']/1e6:10,.1f} jt | qty={row['total_qty']:>14,.0f} | faktur={row['n_faktur']:>5,.0f} | produk={row['n_produk']:>4,.0f}")

# --- 2. Harga satuan implisit per satuan_nama ---
df['harga_per_unit'] = df['jumlah'] / df['d_jual_qty'].replace(0, np.nan)
harga_stats = df.groupby('satuan_nama', dropna=False)['harga_per_unit'].agg(['median', 'mean', 'min', 'max', 'count'])
harga_stats = harga_stats[harga_stats['count'] >= 5].sort_values('median', ascending=False)

print("\n=== HARGA SATUAN IMPLISIT PER SATUAN (median, top 15) ===")
for idx, row in harga_stats.head(15).iterrows():
    print(f"  {str(idx):30s} | median={row['median']:>14,.0f} | mean={row['mean']:>14,.0f} | n={row['count']:>5,.0f}")

# --- 3. Korelasi satuan_nama dengan qty per region-hari ---
# Buat pivot: satuan per region-hari
df['tanggal'] = pd.to_datetime(df['jual_tanggal'])
df['region']  = df['pelanggan_kota'].map(lambda x: str(x).strip())

# Top 10 satuan by total qty
top_satuan = satuan_nilai.head(10).index.tolist()

# Agregasi qty per satuan per region-hari
qty_by_satuan = {}
for s in top_satuan:
    mask = df['satuan_nama'] == s
    agg = df[mask].groupby(['region', 'tanggal'])['d_jual_qty'].sum().reset_index()
    qty_by_satuan[s] = agg

# Total qty per region-hari
total_qty = df.groupby(['region', 'tanggal'])['d_jual_qty'].sum().reset_index()
total_qty.columns = ['region', 'tanggal', 'total_qty']

# Korelasi antar satuan
print("\n=== KORELASI QTY ANTAR SATUAN (region-hari level) ===")
satuan_pivot = {}
for s in top_satuan:
    mask = df['satuan_nama'] == s
    agg = df[mask].groupby(['region', 'tanggal'])['d_jual_qty'].sum().reset_index()
    key = str(s)[:20]
    satuan_pivot[key] = agg.set_index(['region', 'tanggal'])['d_jual_qty']

pivot_df = pd.DataFrame(satuan_pivot)
corr = pivot_df.corr()
print(corr.round(3).to_string())

# --- 4. Variance per satuanNama ---
print("\n=== VARIASI QTY PER SATUAN (CV = std/mean per region) ===")
for s in top_satuan[:8]:
    mask = df['satuan_nama'] == s
    region_qty = df[mask].groupby('region')['d_jual_qty'].sum()
    cv = region_qty.std() / region_qty.mean() if region_qty.mean() > 0 else np.nan
    print(f"  {str(s):30s} | mean={region_qty.mean():>12,.0f} | std={region_qty.std():>12,.0f} | CV={cv:.2f}")

# --- 5. Apakah satuan_nama menambah sinyal di atas qty total? ---
# Uji: untuk setiap region-hari, qty per satuan sebagai fitur tambahan
# Target: total_qty hari berikutnya (1-step ahead, lag 1 per satuan)
print("\n=== SINYAL FITUR: qty per satuan sebagai prediktor total_qty (lag 1) ===")
for s in top_satuan[:6]:
    mask = df['satuan_nama'] == s
    agg = df[mask].groupby(['region', 'tanggal'])['d_jual_qty'].sum().reset_index()
    agg.columns = ['region', 'tanggal', f'qty_{s}']
    
    merged = total_qty.merge(agg, on=['region', 'tanggal'], how='left')
    merged[f'qty_{s}'] = merged.groupby('region')[f'qty_{s}'].shift(1)  # lag 1
    merged['total_qty_lag1'] = merged.groupby('region')['total_qty'].shift(1)
    
    valid = merged.dropna()
    
    # R2 dari lag1 total saja
    from sklearn.metrics import r2_score
    if len(valid) > 10:
        r2_base = r2_score(valid['total_qty'], valid['total_qty_lag1'])
        # R2 dari lag1 total + lag1 satuan ini
        X = valid[['total_qty_lag1', f'qty_{s}']].values
        y = valid['total_qty'].values
        # Simple linear regression
        from numpy.linalg import lstsq
        X_with_const = np.column_stack([X, np.ones(len(X))])
        coef, _, _, _ = lstsq(X_with_const, y, rcond=None)
        y_pred = X_with_const @ coef
        r2_combo = r2_score(y, y_pred)
        delta = r2_combo - r2_base
        print(f"  {str(s):30s} | R2_base={r2_base:.6f} | R2+satuan={r2_combo:.6f} | delta={delta:+.6f}")
    else:
        print(f"  {str(s):30s} | data terlalu sedikit ({len(valid)} baris)")

# --- 6. Ringkasan ---
summary = {
    'n_satuan_unique': int(df['satuan_nama'].nunique(dropna=True)),
    'n_satuan_total': int(satuan_counts.shape[0]),
    'top_10_satuan': [],
}

for s in top_satuan[:10]:
    row = satuan_nilai.loc[s]
    summary['top_10_satuan'].append({
        'satuan': str(s),
        'total_nilai_jt': float(row['total_nilai'] / 1e6),
        'total_qty': float(row['total_qty']),
        'n_faktur': int(row['n_faktur']),
        'n_produk': int(row['n_produk']),
    })

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

print(f"\nSummary saved to {OUT}")
