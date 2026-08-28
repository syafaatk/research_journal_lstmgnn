"""
Aggregation analysis v3: Using the ACTUAL model predictions from experiment_results_zi_geocoded.json.
The preds_mean field contains ensemble predictions (231 x 16).
"""
import numpy as np, pandas as pd, json, os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = r"E:\Download\Jurnal"
DATA = os.path.join(ROOT, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(ROOT, "results", "exp_aggregation_v3.json")

CITY_MAP = {
    "Palembang": "Palembang", "Lahat": "Lahat", "Baturaja": "Ogan Komering Ulu",
    "Oku (Ogan Komering Ulu)": "Ogan Komering Ulu Timur", "Musi Banyuasin": "Musi Banyuasin",
    "Muara Enim": "Muara Enim", "Tanjung Enim": "Muara Enim", "Muara Lawai": "Muara Enim",
    "Lubuklinggau": "Lubuk Linggau", "Ogan Ilir": "Ogan Ilir",
    "Oku Selatan": "Ogan Komering Ulu Selatan",
    "Prabumulih": "Prabumulih", "Prabumulih Timur": "Prabumulih",
    "Pali (Penukal Abab Lematang Ilir)": "Penukal Abab Lematang Ilir",
    "Talang Ubi": "Penukal Abab Lematang Ilir",
    "Musi Rawas": "Musi Rawas", "Oku Timur": "Ogan Komering Ulu Timur",
    "Martapura": "Ogan Komering Ulu Timur",
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

# Load main experiment results
with open(os.path.join(ROOT, "results", "experiment_results_zi_geocoded.json")) as f:
    res = json.load(f)

# The actual model predictions (ensemble)
preds_list = res['models']['HybridTuned']['preds_mean']
preds = np.array(preds_list)  # (231, 16)
print(f"Loaded preds from JSON: shape={preds.shape}")

# Rebuild yte
df = pd.read_excel(DATA)
df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
df = df.drop_duplicates("d_jual_nofak")
df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota", "jual_total_fak": "jual_total"})
df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
df["region"] = df["pelanggan_kota"].map(CITY_MAP)
df["period"] = df["jual_tanggal"].dt.to_period("D")

panel = df.pivot_table(index="period", columns="region", values="jual_total", aggfunc="sum")
panel = panel.reindex(columns=REGIONS).fillna(0.0).sort_index()
n_days = len(panel)
COVID_END = pd.Timestamp("2022-12-31")
covid_flag = np.array([1.0 if m.start_time <= COVID_END else 0.0 for m in panel.index])

WINDOW = 30
y_all, t_all = [], []
for i in range(n_days - WINDOW):
    y_all.append(panel.iloc[i + WINDOW].values)
    t_all.append(panel.index[i + WINDOW])
y_all = np.array(y_all)
te_mask = np.array([m.start_time > pd.Timestamp("2024-12-31") for m in t_all])
yte = y_all[te_mask]
t_test = [t for t, m in zip(t_all, te_mask) if m]

tr_mask = np.array([m.start_time <= pd.Timestamp("2023-12-31") for m in t_all])
ytr = y_all[tr_mask]
train_mean = ytr.mean(axis=0)

print(f"yte shape: {yte.shape}")
print(f"preds shape: {preds.shape}")
assert yte.shape == preds.shape

# Verify main model metrics match
r2_main = r2_score(yte.flatten(), preds.flatten())
rmse_main = np.sqrt(mean_squared_error(yte.flatten(), preds.flatten()))
print(f"\nMain model (HybridTuned ensemble) verification:")
print(f"  R2 (sklearn pooled): {r2_main:.6f}")
print(f"  RMSE: {rmse_main:,.0f}")

# Verify per-region matches JSON
print("\nPer-region verification (vs JSON):")
for j, reg in enumerate(REGIONS):
    r2_j = r2_score(yte[:,j], preds[:,j]) if np.var(yte[:,j]) > 0 else (1.0 if np.all(preds[:,j] == 0) and np.all(yte[:,j] == 0) else np.nan)
    rmse_j = np.sqrt(mean_squared_error(yte[:,j], preds[:,j]))
    json_rmse = res['per_region_hybrid'][reg]['RMSE']
    json_r2 = res['per_region_hybrid'][reg]['R2']
    match = "OK" if abs(rmse_j - json_rmse) < 1 else "MISMATCH"
    print(f"  {reg:<35} my_rmse={rmse_j:>14,.0f} json_rmse={json_rmse:>14,.0f} [{match}]")

# ============ AGGREGATION ============
def calc(y_true, y_pred, label=""):
    yt, yp = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = yt > 0
    r2 = r2_score(yt, yp) if len(yt) > 1 else np.nan
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    return {'label': label, 'n': len(yt), 'r2': round(r2, 6), 'rmse': round(rmse, 0), 'mae': round(mae, 0), 'n_nz': int(mask.sum())}

# Build test df
rows = []
for i, tdate in enumerate(t_test):
    for j, reg in enumerate(REGIONS):
        rows.append({'date': tdate.start_time, 'region': reg, 'y_true': yte[i,j], 'y_pred': preds[i,j]})
test_df = pd.DataFrame(rows)
test_df['date'] = pd.to_datetime(test_df['date'])

# Training mean baseline
tr_pred = np.tile(train_mean, (yte.shape[0], 1))
test_df['y_baseline'] = tr_pred.flatten()[:len(test_df)]

# ============ DAILY ============
d_model = calc(yte.flatten(), preds.flatten(), "Daily - Model")
d_base = calc(yte.flatten(), tr_pred.flatten(), "Daily - TrainMean")

print("\n" + "="*70)
print("DAILY (pooled)")
print("="*70)
for m in [d_model, d_base]:
    print(f"  {m['label']}: R2={m['r2']:.4f}, RMSE={m['rmse']:,.0f}")

# Per-region daily
print("\n  Per-region daily:")
for j, reg in enumerate(REGIONS):
    r2j = r2_score(yte[:,j], preds[:,j]) if np.var(yte[:,j]) > 0 else float('nan')
    rmsej = np.sqrt(mean_squared_error(yte[:,j], preds[:,j]))
    nz = int((yte[:,j] > 0).sum())
    nz_pred = int((preds[:,j] > 0).sum())
    print(f"    {reg:<35} R2={r2j:>8.4f} RMSE={rmsej:>12,.0f} n_nz_true={nz:>3} n_nz_pred={nz_pred:>3}")

# ============ WEEKLY ============
test_df['week'] = test_df['date'].dt.to_period('W').astype(str)
wk = test_df.groupby('week').agg(y_true=('y_true','sum'), y_pred=('y_pred','sum'), y_base=('y_baseline','sum')).reset_index()
w_model = calc(wk['y_true'], wk['y_pred'], "Weekly - Model")
w_base = calc(wk['y_true'], wk['y_base'], "Weekly - TrainMean")

print("\n" + "="*70)
print("WEEKLY (pooled)")
print("="*70)
for m in [w_model, w_base]:
    print(f"  {m['label']}: R2={m['r2']:.4f}, RMSE={m['rmse']:,.0f}")

# ============ MONTHLY ============
test_df['month'] = test_df['date'].dt.to_period('M').astype(str)
mo = test_df.groupby('month').agg(y_true=('y_true','sum'), y_pred=('y_pred','sum'), y_base=('y_baseline','sum')).reset_index()
m_model = calc(mo['y_true'], mo['y_pred'], "Monthly - Model")
m_base = calc(mo['y_true'], mo['y_base'], "Monthly - TrainMean")

print("\n" + "="*70)
print("MONTHLY (pooled)")
print("="*70)
for m in [m_model, m_base]:
    print(f"  {m['label']}: R2={m['r2']:.4f}, RMSE={m['rmse']:,.0f}")

# ============ ZERO FRACTIONS ============
daily_zf = (yte.flatten() == 0).mean()
wk_zf = (wk['y_true'] == 0).mean()
mo_zf = (mo['y_true'] == 0).mean()
print(f"\nZero fractions: daily={daily_zf:.1%}, weekly={wk_zf:.1%}, monthly={mo_zf:.1%}")

# ============ PREDICTION SPARSITY ============
pred_nonzero = (preds > 0).sum()
pred_nonzero_frac = pred_nonzero / preds.size
print(f"\nPrediction sparsity: {pred_nonzero_frac:.1%} non-zero out of {preds.size} predictions")
print(f"  Actual non-zero: {(yte > 0).sum()} out of {yte.size} ({(yte > 0).mean():.1%})")

# Which regions does the model predict as non-zero?
print(f"\nPredicted non-zero counts per region:")
for j, reg in enumerate(REGIONS):
    n_pred = (preds[:,j] > 0).sum()
    n_true = (yte[:,j] > 0).sum()
    print(f"  {reg:<35} predicted={n_pred:>3} actual={n_true:>3}")

# ============ SAVE ============
results = {
    'daily': d_model, 'daily_baseline': d_base,
    'weekly': w_model, 'weekly_baseline': w_base,
    'monthly': m_model, 'monthly_baseline': m_base,
    'zero_fraction': {'daily': daily_zf, 'weekly': wk_zf, 'monthly': mo_zf},
    'prediction_sparsity': pred_nonzero_frac,
}
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved: {OUT}")
