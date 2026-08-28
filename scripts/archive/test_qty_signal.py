# -*- coding: utf-8 -*-
"""
Uji empiris: apakah fitur qty/produk menambah sinyal di atas riwayat penjualan?

Struktur model: tahap jumlah memprediksi log1p(nilai) pada hari non-zero
menggunakan 30 hari riwayat. Uji ini meniru itu dengan regresi linier:
  1) R2 log1p(nilai) ~ fitur qty saja
  2) R2 log1p(nilai) ~ mean 30-hari riwayat (proxy sinyal model)
  3) R2 incremental saat fitur qty ditambahkan ke riwayat
Jika incremental R2 ~ 0, fitur qty tidak akan membantu model.
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

# Agregat level faktur (sebelum dedup, agar qty total per faktur benar)
inv = df.groupby("d_jual_nofak").agg(
    qty=("d_jual_qty", "sum"),
    n_items=("d_jual_id", "count"),
    n_prod=("d_barang_brg_id", "nunique"),
    tanggal=("tanggal", "first"),
    wilayah=("wilayah", "first"),
    jual_total=("jual_total_fak", "first"),
)
inv = inv[inv["wilayah"].isin(CITY_MAP)].copy()
inv["region"] = inv["wilayah"].map(CITY_MAP)
inv["period"] = inv["tanggal"].dt.to_period("D")

def panel_of(col):
    return (inv.pivot_table(index="period", columns="region", values=col, aggfunc="sum")
            .reindex(columns=REGIONS).fillna(0.0).sort_index())

P_val = panel_of("jual_total")
P_qty = panel_of("qty")
P_items = panel_of("n_items")
P_prod = panel_of("n_prod")

# Periode test sama dengan eksperimen: > 2024-12-31
te_mask = P_val.index.astype(str) > "2024-12-31"
val = P_val.values
qty = P_qty.values
items = P_items.values
prod = P_prod.values

# --- Statistik deskriptif untuk paper ---
print("=== Statistik deskriptif (level faktur, setelah pemetaan) ===")
print(f"Faktur unik: {len(inv)}")
print(f"qty per faktur: median {inv['qty'].median():.0f}, mean {inv['qty'].mean():.1f}, max {inv['qty'].max():.0f}")
print(f"item per faktur: median {inv['n_items'].median():.0f}, mean {inv['n_items'].mean():.2f}, max {inv['n_items'].max():.0f}")
print(f"produk per faktur: median {inv['n_prod'].median():.0f}, mean {inv['n_prod'].mean():.2f}, max {inv['n_prod'].max():.0f}")
print(f"faktur 1 item: {(inv['n_items']==1).mean():.1%}")
print(f"faktur 1 produk: {(inv['n_prod']==1).mean():.1%}")

# --- Uji sinyal pada hari non-zero, periode test ---
# PENTING: fitur qty harus LAGGED (yang diketahui model saat prediksi),
# bukan qty hari target (itu leakage - ditentukan faktur yang sama).
print("\n=== Uji sinyal fitur qty LAGGED (hari non-zero, periode test) ===")
y = np.log1p(val)
mask = (val > 0) & te_mask[:, None]
print(f"hari non-zero test: {mask.sum()} dari {te_mask.sum()}")

# Fitur lagged: mean 30-hari riwayat qty/items/prod (rolling, shift 1)
W = 30
def lagged_mean(P):
    m = pd.DataFrame(P).rolling(W).mean().shift(1).values
    return m[W:]  # sejajar dengan target mulai hari W

qty_l = lagged_mean(P_qty)
items_l = lagged_mean(P_items)
prod_l = lagged_mean(P_prod)
hist_val = lagged_mean(P_val)  # riwayat nilai (proxy sinyal model)

y_l = y[W:]
m_hist = (val[W:] > 0) & te_mask[W:, None]

for name, f in [("qty_lag", qty_l), ("items_lag", items_l), ("prod_lag", prod_l)]:
    r = np.corrcoef(f[m_hist], y_l[m_hist])[0, 1]
    print(f"  corr({name}, log1p(nilai)) = {r:.4f}")

X_q = StandardScaler().fit_transform(np.stack([qty_l[m_hist], items_l[m_hist], prod_l[m_hist]], axis=1))
r2_q = LinearRegression().fit(X_q, y_l[m_hist]).score(X_q, y_l[m_hist])
print(f"  R2 log1p(nilai) ~ [qty, items, prod] lagged saja: {r2_q:.4f}")

X_h = StandardScaler().fit_transform(hist_val[m_hist].reshape(-1, 1))
r2_h = LinearRegression().fit(X_h, y_l[m_hist]).score(X_h, y_l[m_hist])
print(f"  R2 log1p(nilai) ~ mean 30-hari riwayat nilai: {r2_h:.4f}")

X_hq = np.hstack([X_h, X_q])
r2_hq = LinearRegression().fit(X_hq, y_l[m_hist]).score(X_hq, y_l[m_hist])
print(f"  R2 + fitur qty lagged: {r2_hq:.4f}  (incremental {r2_hq - r2_h:+.4f})")

# Robustness: semua hari non-zero (train+val+test)
print("\n=== Robustness: semua hari non-zero ===")
m_all = val[W:] > 0
r2_q_all = LinearRegression().fit(
    StandardScaler().fit_transform(np.stack([qty_l[m_all], items_l[m_all], prod_l[m_all]], axis=1)),
    y_l[m_all]).score(
    StandardScaler().fit_transform(np.stack([qty_l[m_all], items_l[m_all], prod_l[m_all]], axis=1)),
    y_l[m_all])
print(f"  R2 fitur qty lagged saja: {r2_q_all:.4f}")
X_h_all = StandardScaler().fit_transform(hist_val[m_all].reshape(-1, 1))
r2_h_all = LinearRegression().fit(X_h_all, y_l[m_all]).score(X_h_all, y_l[m_all])
X_hq_all = np.hstack([X_h_all, StandardScaler().fit_transform(np.stack([qty_l[m_all], items_l[m_all], prod_l[m_all]], axis=1))])
r2_hq_all = LinearRegression().fit(X_hq_all, y_l[m_all]).score(X_hq_all, y_l[m_all])
print(f"  R2 mean 30-hari riwayat nilai: {r2_h_all:.4f}")
print(f"  R2 + fitur qty lagged: {r2_hq_all:.4f}  (incremental {r2_hq_all - r2_h_all:+.4f})")

# --- Statistik merek untuk analisis deskriptif ---
print("\n=== Merek (kategori_nama) top-10 pada faktur terpeta ===")
inv_brand = df[df["wilayah"].isin(CITY_MAP)].groupby("d_jual_nofak")["kategori_nama"].first()
vc = inv_brand.value_counts()
for k, v in vc.head(10).items():
    print(f"  {k}: {v} faktur ({v/len(inv_brand):.1%})")
print(f"  merek unik: {vc.size}")