# -*- coding: utf-8 -*-
"""
Baseline ARIMA dan XGBoost untuk protokol harian zero-inflated 16 region.

Konsisten dengan experiment_zi_geocoded.py:
- Panel harian 16 region, split kronologis (train <= 2023-12-31, val 2024, test 2025)
- Metrik dihitung pooled (region x hari) pada skala asli IDR
- ARIMA: fit per region pada sekuens train+val (1264 hari), forecast 231 langkah
  (test), transformasi log1p -> expm1, clip >= 0. Order dipilih via AIC
  (grid p,q in {0,1,2}, d in {0,1}).
- XGBoost: fitur = riwayat 30 hari (log1p) + flag COVID per (sampel, region),
  target log1p(y), early stopping pada validasi (sama seperti protokol DL),
  expm1 -> clip >= 0.

Output: skrip/baselines_arima_xgb.json + cetak ringkasan.
"""
import json
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DATA = r"E:\Download\Jurnal\view_penjualan_detail data hingga oktober.xlsx"
GEO = r"E:\Download\Jurnal\skrip\customers_geocoded_final.csv"
OUT = r"E:\Download\Jurnal\skrip\baselines_arima_xgb.json"

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
N = len(REGIONS)
WINDOW = 30
COVID_END = pd.Timestamp("2022-12-31")

# ---------------------------------------------------------------------------
# 1. Load dan agregasi harian (identik dengan experiment_zi_geocoded.py)
# ---------------------------------------------------------------------------
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
print(f"Panel: {n_days} hari ({panel.index[0]} s.d. {panel.index[-1]}), {N} region", flush=True)

covid_flag = np.array([1.0 if m.start_time <= COVID_END else 0.0 for m in panel.index])

# ---------------------------------------------------------------------------
# 2. Split kronologis (sama dengan eksperimen utama)
# ---------------------------------------------------------------------------
X_all, y_all, t_all = [], [], []
for i in range(n_days - WINDOW):
    sales = panel.iloc[i:i + WINDOW].values
    flags = np.tile(covid_flag[i:i + WINDOW][:, None], (1, N))
    X_all.append(np.stack([sales, flags], axis=-1))
    y_all.append(panel.iloc[i + WINDOW].values)
    t_all.append(panel.index[i + WINDOW])
X_all = np.array(X_all).transpose(0, 2, 1, 3)  # (B, N, T, 2)
y_all = np.array(y_all)

tr_mask = np.array([m.start_time <= pd.Timestamp("2023-12-31") for m in t_all])
va_mask = np.array([(m.start_time > pd.Timestamp("2023-12-31")) & (m.start_time <= pd.Timestamp("2024-12-31")) for m in t_all])
te_mask = np.array([m.start_time > pd.Timestamp("2024-12-31") for m in t_all])
Xtr, ytr = X_all[tr_mask], y_all[tr_mask]
Xva, yva = X_all[va_mask], y_all[va_mask]
Xte, yte = X_all[te_mask], y_all[te_mask]
print(f"Sampel: train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}", flush=True)
print(f"Zero actuals test: {(yte == 0).sum()} dari {yte.size} ({(yte == 0).mean():.2%})", flush=True)


def metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"RMSE": rmse, "MAE": mae, "R2": float(r2_score(y_true, y_pred))}


# ---------------------------------------------------------------------------
# 3. ARIMA per region: fit train+val (1264 hari), forecast 231 langkah
# ---------------------------------------------------------------------------
n_test = len(yte)
train_val = panel.iloc[:n_days - n_test]          # hari <= 2024-12-31
test_vals = panel.iloc[n_days - n_test:].values   # (231, 16) == yte

GRID = [(p, d, q) for p in range(3) for d in range(2) for q in range(3)]
arima_orders = {}
pred_arima = np.zeros_like(test_vals)
for r, region in enumerate(REGIONS):
    series = np.log1p(train_val[region].values)
    best_aic, best_order, best_res = np.inf, None, None
    for order in GRID:
        try:
            res = ARIMA(series, order=order).fit(method_kwargs={"maxiter": 200})
        except Exception:
            continue
        if res.aic < best_aic:
            best_aic, best_order, best_res = res.aic, order, res
    fc = best_res.forecast(n_test)
    pred_arima[:, r] = np.clip(np.expm1(fc), 0, None)
    arima_orders[region] = best_order
    print(f"ARIMA {region}: order {best_order} AIC {best_aic:.1f}", flush=True)

m_arima = metrics(yte, pred_arima)
print(f"ARIMA: R2 {m_arima['R2']:.4f} RMSE {m_arima['RMSE']:,.0f} MAE {m_arima['MAE']:,.0f}", flush=True)

# ---------------------------------------------------------------------------
# 4. XGBoost: fitur 30 lag log1p + flag COVID, early stopping pada validasi
# ---------------------------------------------------------------------------
def build_features(X):
    B, Nn, T, F = X.shape
    sales = np.log1p(X[:, :, :, 0])          # (B, N, T)
    covid = X[:, :, -1, 1]                    # (B, N)
    feats = np.concatenate([sales.reshape(B * Nn, T), covid.reshape(B * Nn, 1)], axis=1)
    return feats

Xtr_f = build_features(Xtr)
Xva_f = build_features(Xva)
Xte_f = build_features(Xte)
ytr_f = np.log1p(ytr).reshape(-1)
yva_f = np.log1p(yva).reshape(-1)

xgb = XGBRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=42,
    early_stopping_rounds=20,
)
xgb.fit(Xtr_f, ytr_f, eval_set=[(Xva_f, yva_f)], verbose=False)
p_xgb = np.expm1(xgb.predict(Xte_f, iteration_range=(0, xgb.best_iteration + 1)))
p_xgb = np.clip(p_xgb.reshape(n_test, N), 0, None)

m_xgb = metrics(yte, p_xgb)
print(f"XGBoost: R2 {m_xgb['R2']:.4f} RMSE {m_xgb['RMSE']:,.0f} MAE {m_xgb['MAE']:,.0f} "
      f"(best_iter {xgb.best_iteration})", flush=True)

# ---------------------------------------------------------------------------
# 5. Simpan
# ---------------------------------------------------------------------------
out = {
    "ARIMA": m_arima,
    "XGBoost": m_xgb,
    "arima_orders": arima_orders,
    "config": {
        "panel_days": n_days, "regions": N, "window": WINDOW,
        "train": len(Xtr), "val": len(Xva), "test": len(Xte),
        "arima_grid": GRID, "arima_fit": "train+val, forecast test",
        "xgb": {"n_estimators": 500, "max_depth": 6, "lr": 0.05,
                "early_stopping_rounds": 20, "best_iteration": int(xgb.best_iteration)},
    },
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print("Tersimpan:", OUT)