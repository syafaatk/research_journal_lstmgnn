# -*- coding: utf-8 -*-
"""
Uji empiris: apakah fitur turunan dari sales/kategori(supplier)/barang
menambah sinyal pada tahap jumlah (amount stage)?

Struktur sama dengan test_qty_signal.py: fitur LAGGED (mean 30-hari),
incremental R2 di atas riwayat nilai 30 hari, periode test.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

DATA = r"E:\Download\Jurnal\view_penjualan_detail data hingga oktober.xlsx"

CITY_MAP = {
    "Palembang": "Palembang", "Lahat": "Lahat", "Baturaja": "Ogan Komering Ulu",
    "Oku (Ogan Komering Ulu)": "Ogan Komering Ulu Timur", "Musi Banyuasin": "Musi Banyuasin",
    "Muara Enim": "Muara Enim", "Tanjung Enim": "Muara Enim", "Muara Lawai": "Muara Enim",
    "Lubuklinggau": "Lubuk Linggau", "Ogan Ilir": "Ogan Ilir", "Oku Selatan": "Ogan Komering Ulu Selatan",
    "Prabumulih": "Prabumulih", "Prabumulih Timur": "Prabumulih",
    "Pali (Penukal Abab Lematang Ilir)": "Penukal Abab Lematang Ilir", "Talang Ubi": "Penukal Abab Lematang Ilir",
    "Musi Rawas": "Musi Rawas", "Oku Timur": "Ogan Komering Ulu Timur", "Martapura": "Ogan Komering Ulu Timur",
    "Empat Lawang": "Empat Lawang", "Banyuasin": "Banyuasin", "Pagaralam": "Pagar Alam",
    "Muara Rupit": "Musi Rawas Utara", "Rupit": "Musi Rawas Utara",
    "Oki (Ogan Komering Ilir)": "Ogan Komering Ilir",
}
REGIONS = [
    "Banyuasin", "Empat Lawang", "Lahat", "Lubuk Linggau", "Muara Enim",
    "Musi Banyuasin", "Musi Rawas", "Musi Rawas Utara", "Ogan Ilir",
    "Ogan Komering Ilir", "Ogan Komering Ulu", "Ogan Komering Ulu Selatan",
    "Ogan Komering Ulu Timur", "Pagar Alam", "Palembang",
    "Penukal Abab Lematang Ilir", "Prabumulih",
]

df = pd.read_excel(DATA)
df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])

# Cek kelengkapan kolom
print("=== Kelengkapan kolom (level baris) ===")
for c in ["sales_id", "sales_nama", "suplier_id", "kategori_nama", "barang_nama"]:
    print(f"  {c}: null {df[c].isna().sum():>6} | unique {df[c].nunique():>6}")

# Agregat level faktur (konsisten dengan panel nilai)
inv = df.groupby("d_jual_nofak").agg(
    jual_total=("jual_total_fak", "first"),
    tanggal=("tanggal", "first"),
    wilayah=("wilayah", "first"),
    n_brand=("kategori_nama", "nunique"),
    n_supplier=("suplier_id", "nunique"),
    n_sales=("sales_id", "nunique"),
)
inv = inv[inv["wilayah"].isin(CITY_MAP)].copy()
inv["region"] = inv["wilayah"].map(CITY_MAP)
inv["period"] = inv["tanggal"].dt.to_period("D")

def panel_of(col):
    return (inv.pivot_table(index="period", columns="region", values=col, aggfunc="sum")
            .reindex(columns=REGIONS).fillna(0.0).sort_index())

P_val = panel_of("jual_total")
P_brand = panel_of("n_brand")
P_supplier = panel_of("n_supplier")
P_sales = panel_of("n_sales")

W = 30
def lagged_mean(P):
    m = pd.DataFrame(P).rolling(W).mean().shift(1).values
    return m[W:]

val = P_val.values
y = np.log1p(val)
y_l = y[W:]
te_mask = P_val.index.astype(str) > "2024-12-31"
m_hist = (val[W:] > 0) & te_mask[W:, None]

feats = {
    "n_brand_lag": lagged_mean(P_brand),
    "n_supplier_lag": lagged_mean(P_supplier),
    "n_sales_lag": lagged_mean(P_sales),
}
hist_val = lagged_mean(P_val)

print("\n=== Uji sinyal fitur LAGGED (hari non-zero, periode test) ===")
print(f"hari non-zero test: {m_hist.sum()} dari {te_mask.sum()}")
for name, f in feats.items():
    r = np.corrcoef(f[m_hist], y_l[m_hist])[0, 1]
    print(f"  corr({name}, log1p(nilai)) = {r:.4f}")

X_f = StandardScaler().fit_transform(np.stack([f[m_hist] for f in feats.values()], axis=1))
r2_f = LinearRegression().fit(X_f, y_l[m_hist]).score(X_f, y_l[m_hist])
print(f"  R2 [brand, supplier, sales] lagged saja: {r2_f:.4f}")

X_h = StandardScaler().fit_transform(hist_val[m_hist].reshape(-1, 1))
r2_h = LinearRegression().fit(X_h, y_l[m_hist]).score(X_h, y_l[m_hist])
X_hf = np.hstack([X_h, X_f])
r2_hf = LinearRegression().fit(X_hf, y_l[m_hist]).score(X_hf, y_l[m_hist])
print(f"  R2 riwayat nilai 30-hari: {r2_h:.4f}")
print(f"  R2 + fitur brand/supplier/sales: {r2_hf:.4f}  (incremental {r2_hf - r2_h:+.4f})")

# Robustness: semua hari non-zero
print("\n=== Robustness: semua hari non-zero ===")
m_all = val[W:] > 0
r2_f_all = LinearRegression().fit(
    StandardScaler().fit_transform(np.stack([f[m_all] for f in feats.values()], axis=1)),
    y_l[m_all]).score(
    StandardScaler().fit_transform(np.stack([f[m_all] for f in feats.values()], axis=1)),
    y_l[m_all])
X_h_all = StandardScaler().fit_transform(hist_val[m_all].reshape(-1, 1))
r2_h_all = LinearRegression().fit(X_h_all, y_l[m_all]).score(X_h_all, y_l[m_all])
X_hf_all = np.hstack([X_h_all, StandardScaler().fit_transform(np.stack([f[m_all] for f in feats.values()], axis=1))])
r2_hf_all = LinearRegression().fit(X_hf_all, y_l[m_all]).score(X_hf_all, y_l[m_all])
print(f"  R2 fitur lagged saja: {r2_f_all:.4f}")
print(f"  R2 riwayat nilai: {r2_h_all:.4f}")
print(f"  R2 + fitur: {r2_hf_all:.4f}  (incremental {r2_hf_all - r2_h_all:+.4f})")

# Statistik deskriptif tambahan untuk paper
print("\n=== Statistik deskriptif (level faktur terpeta) ===")
print(f"  brand per faktur: median {inv['n_brand'].median():.0f}, max {inv['n_brand'].max():.0f}")
print(f"  supplier per faktur: median {inv['n_supplier'].median():.0f}, max {inv['n_supplier'].max():.0f}")
print(f"  sales per faktur: median {inv['n_sales'].median():.0f}, max {inv['n_sales'].max():.0f}")
print(f"  sales unik: {df['sales_id'].nunique()} | supplier unik: {df['suplier_id'].nunique()}")