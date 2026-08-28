# -*- coding: utf-8 -*-
"""Baseline sederhana (always-zero, naive, train-mean) untuk panel 16 region."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

DATA = "E:/Download/Jurnal/view_penjualan_detail data hingga oktober.xlsx"
CITY_MAP = {
    "Palembang": "Palembang", "Lahat": "Lahat", "Baturaja": "Ogan Komering Ulu",
    "Oku (Ogan Komering Ulu)": "Ogan Komering Ulu Timur", "Musi Banyuasin": "Musi Banyuasin",
    "Muara Enim": "Muara Enim", "Tanjung Enim": "Muara Enim", "Muara Lawai": "Muara Enim",
    "Lubuklinggau": "Lubuk Linggau", "Ogan Ilir": "Ogan Ilir", "Oku Selatan": "Ogan Komering Ulu Selatan",
    "Prabumulih": "Prabumulih", "Prabumulih Timur": "Prabumulih",
    "Pali (Penukal Abab Lematang Ilir)": "Penukal Abab Lematang Ilir", "Talang Ubi": "Penukal Abab Lematang Ilir",
    "Musi Rawas": "Musi Rawas", "Oku Timur": "Ogan Komering Ulu Timur", "Martapura": "Ogan Komering Ulu Timur",
    "Empat Lawang": "Empat Lawang", "Banyuasin": "Banyuasin", "Pagaralam": "Pagar Alam",
    "Muara Rupit": "Musi Rawas", "Rupit": "Musi Rawas",
    "Oki (Ogan Komering Ilir)": "Ogan Komering Ilir",
}
REGIONS = [
    "Banyuasin", "Empat Lawang", "Lahat", "Lubuk Linggau", "Muara Enim",
    "Musi Banyuasin", "Musi Rawas", "Ogan Ilir",
    "Ogan Komering Ilir", "Ogan Komering Ulu", "Ogan Komering Ulu Selatan",
    "Ogan Komering Ulu Timur", "Pagar Alam", "Palembang",
    "Penukal Abab Lematang Ilir", "Prabumulih",
]
WINDOW = 30

df = pd.read_excel(DATA)
df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
df = df.drop_duplicates("d_jual_nofak")
df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                        "jual_total_fak": "jual_total"})
df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
df["region"] = df["pelanggan_kota"].map(CITY_MAP)
df["period"] = df["jual_tanggal"].dt.to_period("D")

panel = df.pivot_table(index="period", columns="region", values="jual_total", aggfunc="sum")
panel = panel.reindex(columns=REGIONS).fillna(0.0).sort_index()
n_days = len(panel)

X_all, y_all, t_all = [], [], []
for i in range(n_days - WINDOW):
    X_all.append(panel.iloc[i:i + WINDOW].values)
    y_all.append(panel.iloc[i + WINDOW].values)
    t_all.append(panel.index[i + WINDOW])
X_all = np.array(X_all)
y_all = np.array(y_all)

tr_mask = np.array([m.start_time <= pd.Timestamp("2023-12-31") for m in t_all])
te_mask = np.array([m.start_time > pd.Timestamp("2024-12-31") for m in t_all])
ytr, yte = y_all[tr_mask], y_all[te_mask]

pred_zero = np.zeros_like(yte)
pred_naive = X_all[te_mask][:, -1, :]
pred_mean = np.tile(ytr.mean(axis=0), (len(yte), 1))


def m(y_true, y_pred):
    return {"RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
            "MAE": float(np.mean(np.abs(y_true - y_pred))),
            "R2": float(r2_score(y_true, y_pred))}


print("always_zero:", m(yte, pred_zero))
print("naive_last :", m(yte, pred_naive))
print("train_mean :", m(yte, pred_mean))
print(f"zero actuals test: {(yte == 0).sum()} dari {yte.size} ({(yte == 0).mean():.2%})")