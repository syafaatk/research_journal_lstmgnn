"""
Eksplorasi mendalam: merek KN95, pengaruh satuan_nama terhadap qty,
dan produk lain dengan pola packaging serupa (box isi N pcs).
"""
import pandas as pd, numpy as np, json, os

DATA = r"E:\Download\Jurnal\data\view_penjualan_detail data hingga oktober.xlsx"
OUT  = r"E:\Download\Jurnal\results\satuan_deep_analysis.json"

df = pd.read_excel(DATA)
df['jumlah'] = pd.to_numeric(df['jumlah'], errors='coerce')
df['d_jual_qty'] = pd.to_numeric(df['d_jual_qty'], errors='coerce')
df['harga_per_qty'] = df['jumlah'] / df['d_jual_qty'].replace(0, np.nan)

# Filter only South Sumatra 16 regions (for context)
REGION_MAP = {
    'BANYUASIN': 'Banyuasin', 'EMPAT LAWANG': 'Empat Lawang',
    'LAHAT': 'Lahat', 'LUBUK LINGGAU': 'Lubuk Linggau',
    'MUARA ENIM': 'Muara Enim', 'MUSI BANYUASIN': 'Musi Banyuasin',
    'MUSI RAWAS': 'Musi Rawas', 'MUSI RAWAS UTARA': 'Musi Rawas',
    'OGAN ILIR': 'Ogan Ilir', 'OGAN KOMERING ILIR': 'Ogan Komering Ilir',
    'OGAN KOMERING ULU': 'Ogan Komering Ulu',
    'OGAN KOMERING ULU SELATAN': 'Ogan Komering Ulu Selatan',
    'OGAN KOMERING ULU TIMUR': 'Ogan Komering Ulu Timur',
    'PAGAR ALAM': 'Pagar Alam', 'PALEMBANG': 'Palembang',
    'PENUKAL ABAB LEMATANG ILIR': 'Penukal Abab Lematang Ilir',
    'PRABUMULIH': 'Prabumulih',
}

print("="*90)
print("BAGIAN 1: SEMUA MEREK KN95 / N95 / MASKER KESEHATAN")
print("="*90)

# Cari semua produk yang mengandung masker/KN95/N95
masker_keywords = ['KN95', 'N95', 'KN95', 'MASKER', 'MASK', 'RESPIRAT', 'NASAL CANNULA', 'FACE SHIELD']
mask = df['barang_nama'].str.upper().str.contains('|'.join(masker_keywords), na=False, case=False)
masker_products = df[mask].copy()

print(f"\nTotal baris masker/respirasi: {len(masker_products)}")
print(f"Produk unik: {masker_products['barang_nama'].nunique()}")

# Detail per produk
masker_agg = masker_products.groupby(['barang_nama', 'barang_spesifikasi', 'satuan_nama']).agg(
    n_baris=('jumlah', 'count'),
    total_nilai=('jumlah', 'sum'),
    total_qty=('d_jual_qty', 'sum'),
    harga_per_qty_median=('harga_per_qty', 'median'),
    harga_per_qty_min=('harga_per_qty', 'min'),
    harga_per_qty_max=('harga_per_qty', 'max'),
).sort_values('total_nilai', ascending=False)

print(f"\n{'Produk + Spesifikasi':<65} | {'Satuan':>6} | {'Baris':>5} | {'Nilai(M)':>9} | {'Qty':>10} | {'Hrg/unit med':>12}")
print("-"*130)
for (pname, spec, satuan), r in masker_agg.iterrows():
    label = f"{pname}"
    if pd.notna(spec) and str(spec).strip():
        label += f" [{str(spec)[:30]}]"
    label = label[:63]
    print(f"{label:<65} | {str(satuan):>6} | {r['n_baris']:>5,.0f} | {r['total_nilai']/1e6:>9,.1f} | {r['total_qty']:>10,.0f} | {r['harga_per_qty_median']:>12,.0f}")

# Fokus khusus KN95
print("\n" + "="*90)
print("BAGIAN 2: SPESIFIK KN95 (semua merek)")
print("="*90)

kn95_mask = df['barang_nama'].str.upper().str.contains('KN95|N95', na=False, case=False)
kn95 = df[kn95_mask].copy()

kn95_detail = kn95.groupby(['barang_nama', 'barang_spesifikasi', 'satuan_nama']).agg(
    n_baris=('jumlah', 'count'),
    n_faktur=('d_jual_nofak', 'nunique'),
    total_nilai=('jumlah', 'sum'),
    total_qty=('d_jual_qty', 'sum'),
    harga_median=('harga_per_qty', 'median'),
    harga_min=('harga_per_qty', 'min'),
    harga_max=('harga_per_qty', 'max'),
    mean_qty=('d_jual_qty', 'mean'),
).sort_values('total_nilai', ascending=False)

print(f"\n{'Produk':<45} | {'Spesifikasi':<30} | {'Sat':>5} | {'Baris':>5} | {'Fakt':>5} | {'Nilai(M)':>9} | {'Qty':>10} | {'Hrg/Unit':>10}")
print("-"*155)
for (pname, spec, satuan), r in kn95_detail.iterrows():
    pname_s = str(pname)[:43]
    spec_s = str(spec)[:28] if pd.notna(spec) else '-'
    print(f"{pname_s:<45} | {spec_s:<30} | {str(satuan):>5} | {r['n_baris']:>5,.0f} | {r['n_faktur']:>5,.0f} | {r['total_nilai']/1e6:>9,.1f} | {r['total_qty']:>10,.0f} | {r['harga_median']:>10,.0f}")

print(f"\nTotal KN95/N95: {len(kn95)} baris, {kn95['jumlah'].sum()/1e6:,.1f} M IDR")

print("\n" + "="*90)
print("BAGIAN 3: PRODUK 'BOX' DENGAN HARGA/UNIT TINGGI (= box isi banyak)")
print("  -> Box isi 10-50 pcs: harga per pcs rendah, tapi harga per box tinggi")
print("="*90)

# Produk dengan satuan=Box
box_products = df[df['satuan_nama'] == 'Box'].copy()

box_agg = box_products.groupby(['barang_nama', 'barang_spesifikasi']).agg(
    n_baris=('jumlah', 'count'),
    total_nilai=('jumlah', 'sum'),
    total_qty=('d_jual_qty', 'sum'),
    harga_per_box_median=('harga_per_qty', 'median'),
    harga_per_box_min=('harga_per_qty', 'min'),
    harga_per_box_max=('harga_per_qty', 'max'),
    mean_qty=('d_jual_qty', 'mean'),
).sort_values('total_nilai', ascending=False)

print(f"\nTOP 30 PRODUK 'Box' (oleh total nilai):")
print(f"{'Produk':<50} | {'Baris':>5} | {'Nilai(M)':>9} | {'Qty':>8} | {'Hrg/Box med':>12} | {'Qty med':>8}")
print("-"*115)
for (pname, spec), r in box_agg.head(30).iterrows():
    label = str(pname)[:48]
    print(f"{label:<50} | {r['n_baris']:>5,.0f} | {r['total_nilai']/1e6:>9,.1f} | {r['total_qty']:>8,.0f} | {r['harga_per_box_median']:>12,.0f} | {r['mean_qty']:>8,.1f}")

print("\n" + "="*90)
print("BAGIAN 4: PRODUK 'PACK' DENGAN POLA SERUPA")
print("="*90)

pack_products = df[df['satuan_nama'] == 'Pack'].copy()
pack_agg = pack_products.groupby(['barang_nama', 'barang_spesifikasi']).agg(
    n_baris=('jumlah', 'count'),
    total_nilai=('jumlah', 'sum'),
    total_qty=('d_jual_qty', 'sum'),
    harga_per_pack_median=('harga_per_qty', 'median'),
    mean_qty=('d_jual_qty', 'mean'),
).sort_values('total_nilai', ascending=False)

print(f"\nTOP 20 PRODUK 'Pack':")
print(f"{'Produk':<50} | {'Baris':>5} | {'Nilai(M)':>9} | {'Qty':>8} | {'Hrg/Pack med':>12} | {'Qty med':>8}")
print("-"*115)
for (pname, spec), r in pack_agg.head(20).iterrows():
    label = str(pname)[:48]
    print(f"{label:<50} | {r['n_baris']:>5,.0f} | {r['total_nilai']/1e6:>9,.1f} | {r['total_qty']:>8,.0f} | {r['harga_per_pack_median']:>12,.0f} | {r['mean_qty']:>8,.1f}")

print("\n" + "="*90)
print("BAGIAN 5: ANALISIS 'BOX ISI N' - deteksi box product dari harga satuan")
print("  Jika 1 box = 20 pcs @ ~15rb/pc -> harga box ~300rb")
print("  Jika 1 box = 50 pcs @ ~15rb/pc -> harga box ~750rb")
print("="*90)

# Untuk produk Box, hitung rasio harga/qty (proxy isi per box)
# Jika harga per box rendah -> box kecil (isi 10-20)
# Jika harga per box tinggi -> box besar atau barang mahal
box_detail = df[df['satuan_nama'] == 'Box'].copy()
box_detail['harga_per_unit'] = box_detail['jumlah'] / box_detail['d_jual_qty'].replace(0, np.nan)

# Grup berdasarkan harga per box
box_price_ranges = [
    ('< 50.000', 0, 50000),
    ('50.000-150.000', 50000, 150000),
    ('150.000-500.000', 150000, 500000),
    ('500.000-1.500.000', 500000, 1500000),
    ('> 1.500.000', 1500000, float('inf')),
]

for label, lo, hi in box_price_ranges:
    mask = (box_detail['harga_per_unit'] >= lo) & (box_detail['harga_per_unit'] < hi)
    sub = box_detail[mask]
    if len(sub) > 0:
        top = sub.groupby('barang_nama')['jumlah'].sum().nlargest(5)
        print(f"\n  Harga box {label} ({len(sub)} baris, total {sub['jumlah'].sum()/1e6:,.1f} M):")
        for pname, val in top.items():
            sub2 = sub[sub['barang_nama'] == pname]
            med_harga = sub2['harga_per_unit'].median()
            med_qty = sub2['d_jual_qty'].median()
            print(f"    {pname[:55]:55s} | hrg/box={med_harga:>10,.0f} | qty/ trans={med_qty:.0f}")
