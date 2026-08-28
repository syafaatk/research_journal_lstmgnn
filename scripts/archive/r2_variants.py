# -*- coding: utf-8 -*-
"""Cek varian R2: per-region normalized, log1p, dsb. untuk prediksi HybridTuned."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

with open(r"C:\Users\ADMLAY~1\AppData\Local\Temp\opencode\experiment_results.json", encoding="utf-8") as f:
    r = json.load(f)

preds = np.array(r["models"]["HybridTuned"]["preds_original"])  # (10, 17)
regions = list(r["panel"]["desc"].keys())

# rebuild test actuals
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
df = pd.read_excel(DATA)
df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
df = df.drop_duplicates("d_jual_nofak")
df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                        "jual_total_fak": "jual_total", "latitude": "Latitude"})
df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
df["region"] = df["pelanggan_kota"].map(CITY_MAP)
df["month"] = df["jual_tanggal"].dt.to_period("M")
panel = df.pivot_table(index="month", columns="region", values="jual_total", aggfunc="sum")
panel = panel.reindex(columns=regions).fillna(0.0).sort_index()

WINDOW = 12
train_end = pd.Period("2023-12", "M")
val_end = pd.Period("2024-12", "M")
X_all, y_all, t_all = [], [], []
for i in range(len(panel) - WINDOW):
    X_all.append(panel.iloc[i:i + WINDOW].values)
    y_all.append(panel.iloc[i + WINDOW].values)
    t_all.append(panel.index[i + WINDOW])
y_all = np.array(y_all)
tr_mask = np.array([m <= train_end for m in t_all])
te_mask = np.array([m > val_end for m in t_all])
ytr, yte = y_all[tr_mask], y_all[te_mask]

print("=== R2 varian (HybridTuned preds) ===")
# 1. pooled original
print("1. pooled original:      R2 =", round(r2_score(yte, preds), 4))
# 2. per-region min-max (fit train)
def minmax_fit(ytr):
    lo = ytr.min(axis=0); hi = ytr.max(axis=0)
    return lo, hi
lo, hi = minmax_fit(ytr)
def minmax(y, lo, hi):
    return (y - lo) / (hi - lo)
yte_mm = minmax(yte, lo, hi)
preds_mm = minmax(preds, lo, hi)
print("2. per-region minmax:    R2 =", round(r2_score(yte_mm, preds_mm), 4))
# 3. per-region z-score (fit train)
mu = ytr.mean(axis=0); sd = ytr.std(axis=0)
yte_z = (yte - mu) / sd
preds_z = (preds - mu) / sd
print("3. per-region zscore:    R2 =", round(r2_score(yte_z, preds_z), 4))
# 4. log1p pooled
print("4. log1p pooled:         R2 =", round(r2_score(np.log1p(yte), np.log1p(np.clip(preds, 0, None))), 4))
# 5. per-region log1p zscore
yte_lz = (np.log1p(yte) - np.log1p(ytr).mean(axis=0)) / np.log1p(ytr).std(axis=0)
preds_lz = (np.log1p(np.clip(preds, 0, None)) - np.log1p(ytr).mean(axis=0)) / np.log1p(ytr).std(axis=0)
print("5. per-region log1p-z:   R2 =", round(r2_score(yte_lz, preds_lz), 4))
# 6. mean per-region R2 (average of per-region R2)
per_r2 = [r2_score(yte[:, i], preds[:, i]) for i in range(len(regions))]
print("6. mean per-region R2:   R2 =", round(np.mean(per_r2), 4))
# 7. weighted per-region R2 by train variance
w = ytr.var(axis=0)
w_r2 = np.average(per_r2, weights=w)
print("7. var-weighted R2:      R2 =", round(w_r2, 4))

# what about the thesis's transformed-scale claim? compute R2 on PowerTransformer scale
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method="yeo-johnson")
pt.fit(ytr.reshape(-1, 1))
yte_t = pt.transform(yte.reshape(-1, 1)).reshape(yte.shape)
preds_t = pt.transform(preds.reshape(-1, 1)).reshape(preds.shape)
print("8. PowerTransformer R2:  R2 =", round(r2_score(yte_t, preds_t), 4))

# per-region R2 detail
print("\nPer-region R2 (original scale):")
for i, reg in enumerate(regions):
    print(f"  {reg}: {per_r2[i]:.3f}")