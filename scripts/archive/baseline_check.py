# -*- coding: utf-8 -*-
"""Baseline: naive (last value) dan mean predictor pada test set."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

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
df = df.drop_duplicates("d_jual_nofak")
df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                        "jual_total_fak": "jual_total", "latitude": "Latitude"})
df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
df["region"] = df["pelanggan_kota"].map(CITY_MAP)
df["month"] = df["jual_tanggal"].dt.to_period("M")
panel = df.pivot_table(index="month", columns="region", values="jual_total", aggfunc="sum")
panel = panel.reindex(columns=REGIONS).fillna(0.0).sort_index()

WINDOW = 12
train_end = pd.Period("2023-12", "M")
val_end = pd.Period("2024-12", "M")

X_all, y_all, t_all = [], [], []
for i in range(len(panel) - WINDOW):
    X_all.append(panel.iloc[i:i + WINDOW].values)
    y_all.append(panel.iloc[i + WINDOW].values)
    t_all.append(panel.index[i + WINDOW])
X_all = np.array(X_all)
y_all = np.array(y_all)

tr_mask = np.array([m <= train_end for m in t_all])
te_mask = np.array([m > val_end for m in t_all])
Xtr, ytr = X_all[tr_mask], y_all[tr_mask]
Xte, yte = X_all[te_mask], y_all[te_mask]

print("test shape:", yte.shape, "| train shape:", ytr.shape)

# Baseline 1: naive (last value of window)
naive = Xte[:, -1, :]
# Baseline 2: per-region train mean
train_mean = ytr.mean(axis=0)
mean_pred = np.tile(train_mean, (len(yte), 1))
# Baseline 3: pooled train mean
pooled_mean = np.full_like(yte, ytr.mean())

def m(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

for name, p in [("Naive (last value)", naive),
                ("Per-region train mean", mean_pred),
                ("Pooled train mean", pooled_mean)]:
    rmse, mae, r2 = m(yte, p)
    print(f"{name}: RMSE={rmse:,.0f} MAE={mae:,.0f} R2={r2:.4f}")

# Per-region naive R2
print("\nPer-region naive (last value) R2:")
for i, reg in enumerate(REGIONS):
    r2 = r2_score(yte[:, i], naive[:, i])
    print(f"  {reg}: {r2:.3f}")

# Per-region train-mean R2
print("\nPer-region train-mean R2:")
for i, reg in enumerate(REGIONS):
    r2 = r2_score(yte[:, i], mean_pred[:, i])
    print(f"  {reg}: {r2:.3f}")

# What variance does Palembang contribute?
print("\nPalembang share of test SS_tot:",
      round(np.sum((yte[:, REGIONS.index('Palembang')] - yte.mean()) ** 2) / np.sum((yte - yte.mean()) ** 2), 3))