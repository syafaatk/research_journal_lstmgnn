# -*- coding: utf-8 -*-
"""
Hitung ulang DM test, bootstrap CI, per-wilayah, dan residual dengan
Hybrid sebagai model referensi, memakai prediksi ensemble (mean seeds)
yang tersimpan di experiment_results_zi.json. Tanpa training.
"""
import json
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score

OUT = r"E:\Download\Jurnal\results\experiment_results_zi_geocoded.json"
REGIONS = [
    "Banyuasin", "Empat Lawang", "Lahat", "Lubuk Linggau", "Muara Enim",
    "Musi Banyuasin", "Musi Rawas", "Ogan Ilir",
    "Ogan Komering Ilir", "Ogan Komering Ulu", "Ogan Komering Ulu Selatan",
    "Ogan Komering Ulu Timur", "Pagar Alam", "Palembang",
    "Penukal Abab Lematang Ilir", "Prabumulih",
]

d = json.load(open(OUT, encoding="utf-8"))
n_test = d["split"]["test"]
yte = None  # tidak tersimpan; rekonstruksi dari metrik tidak mungkin.
# Prediksi ensemble tersimpan; yte direkonstruksi dari preds_mean + residual? Tidak.
# Solusi: muat ulang yte dari pipeline data (cepat, tanpa training).
import pandas as pd

DATA = r"E:\Download\Jurnal\data\view_penjualan_detail data hingga oktober.xlsx"
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
N = len(REGIONS)

df = pd.read_excel(DATA)
df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
df = df.drop_duplicates("d_jual_nofak")
df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                        "jual_total_fak": "jual_total", "latitude": "Latitude"})
df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
df["region"] = df["pelanggan_kota"].map(CITY_MAP)
df["period"] = df["jual_tanggal"].dt.to_period("D")
panel = df.pivot_table(index="period", columns="region", values="jual_total", aggfunc="sum")
panel = panel.reindex(columns=REGIONS).fillna(0.0).sort_index()

WINDOW = 30
y_all, t_all = [], []
for i in range(len(panel) - WINDOW):
    y_all.append(panel.iloc[i + WINDOW].values)
    t_all.append(panel.index[i + WINDOW])
y_all = np.array(y_all)
te_mask = np.array([m.start_time > pd.Timestamp("2024-12-31") for m in t_all])
yte = y_all[te_mask]
print(f"yte shape: {yte.shape}")

preds = {k: np.array(v["preds_mean"]) for k, v in d["models"].items()}
print("R2 ensemble:")
for k, p in preds.items():
    print(f"  {k}: {r2_score(yte, p):.4f}")

def dm_test(e1, e2):
    dd = e1 ** 2 - e2 ** 2
    n = len(dd)
    dbar = dd.mean()
    var = np.var(dd, ddof=1) / n
    dm = dbar / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)

print("\nDM test vs Hybrid (ensemble):")
dm = {}
for name in ["LSTM", "GNN", "HybridTuned"]:
    dmv, pv = dm_test((yte - preds[name]).flatten(), (yte - preds["Hybrid"]).flatten())
    dm[name] = {"DM": dmv, "p": pv}
    print(f"  {name}: DM={dmv:.3f}, p={pv:.4f}")

def bootstrap_rmse_ci(y_true, y_pred, n_boot=1000, alpha=0.05):
    rng = np.random.default_rng(42)
    errs = (y_true - y_pred).flatten()
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(errs), len(errs))
        boots.append(np.sqrt(np.mean(errs[idx] ** 2)))
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))

print("\nBootstrap CI RMSE (ensemble):")
boot = {}
for name, p in preds.items():
    lo, hi = bootstrap_rmse_ci(yte, p)
    boot[name] = [lo, hi]
    print(f"  {name}: [{lo:,.0f}, {hi:,.0f}]")

def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else float("nan")
    return {"RMSE": float(rmse), "MAE": float(mae), "MAPE": float(mape), "R2": float(r2_score(y_true, y_pred))}

print("\nPer wilayah (Hybrid ensemble):")
per_region = {}
for r in range(N):
    per_region[REGIONS[r]] = metrics(yte[:, r], preds["Hybrid"][:, r])
    print(f"  {REGIONS[r]}: R2 {per_region[REGIONS[r]]['R2']:.4f} RMSE {per_region[REGIONS[r]]['RMSE']:,.0f} "
          f"MAE {per_region[REGIONS[r]]['MAE']:,.0f}")

resid = (yte - preds["Hybrid"]).flatten()
def ljung_box(x, lags=10):
    n = len(x)
    x = x - x.mean()
    acf = np.array([np.corrcoef(x[:-l], x[l:])[0, 1] if l < n else 0 for l in range(1, lags + 1)])
    acf = np.nan_to_num(acf)
    q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, lags + 1)))
    p = 1 - stats.chi2.cdf(q, lags)
    return float(q), float(p)

q_lb, p_lb = ljung_box(resid)
print(f"\nResidual (Hybrid ensemble): mean {resid.mean():,.0f}, Ljung-Box Q={q_lb:.3f}, p={p_lb:.4f}")

# Simpan ke JSON
d["dm_vs_hybrid"] = dm
d["bootstrap_ci_ensemble"] = boot
d["per_region_hybrid"] = per_region
d["residual_hybrid"] = {"mean": float(resid.mean()), "ljung_box_q": q_lb, "ljung_box_p": p_lb}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2, default=str)
print("\nDisimpan ke", OUT)