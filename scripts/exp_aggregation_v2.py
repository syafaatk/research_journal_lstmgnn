"""
Aggregation experiment v2:
- Use mean-based R2 (matching pipeline's sklearn formula)
- Compute what the TRAINING-MEAN baseline would give at each granularity
- Shows whether model adds value beyond trivial baseline
"""
import numpy as np, pandas as pd, json, os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = r"E:\Download\Jurnal"
DATA = os.path.join(ROOT, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(ROOT, "results", "exp_aggregation_v2.json")

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

# Build panel
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

tr_mask = np.array([m.start_time <= pd.Timestamp("2023-12-31") for m in t_all])
va_mask = np.array([(m.start_time > pd.Timestamp("2023-12-31")) & (m.start_time <= pd.Timestamp("2024-12-31")) for m in t_all])
te_mask = np.array([m.start_time > pd.Timestamp("2024-12-31") for m in t_all])
yte = y_all[te_mask]
t_test = [t for t, m in zip(t_all, te_mask) if m]

# Load V3 ensemble predictions
preds_42 = np.load(os.path.join(ROOT, "results", "v3_preds_seed42.npy"))
preds_7 = np.load(os.path.join(ROOT, "results", "v3_preds_seed7.npy"))
preds_123 = np.load(os.path.join(ROOT, "results", "v3_preds_seed123.npy"))
preds = (preds_42 + preds_7 + preds_123) / 3.0

# Training mean per region (for baseline comparison)
ytr = y_all[tr_mask]
train_mean = ytr.mean(axis=0)  # (16,)

print(f"Test: {yte.shape[0]} days x {N} regions")
print(f"V3 model R2 (sklearn, pooled): {r2_score(yte.flatten(), preds.flatten()):.6f}")
print(f"V3 model RMSE: {np.sqrt(np.mean((yte - preds)**2)):,.0f}")
print(f"Train mean per region:")
for j, r in enumerate(REGIONS):
    print(f"  {r}: {train_mean[j]:>12,.0f}")

# ============ PER-GRANULARITY ANALYSIS ============
def metrics(y_true, y_pred, label=""):
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    mask = yt > 0
    if len(yt) < 2:
        return {'label': label, 'n': len(yt), 'r2': None, 'rmse': float(np.sqrt(np.mean((yt-yp)**2))), 'mae': float(np.mean(np.abs(yt-yp)))}
    r2 = r2_score(yt, yp)
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

# Training mean baseline (pooled across regions)
train_mean_pred = np.tile(train_mean, (yte.shape[0], 1))

# ============ DAILY ============
d_model = metrics(yte.flatten(), preds.flatten(), "Daily - Model")
d_baseline = metrics(yte.flatten(), train_mean_pred.flatten(), "Daily - TrainMean baseline")

print("\n" + "="*70)
print("DAILY (pooled across 16 regions)")
print("="*70)
for m in [d_model, d_baseline]:
    r2s = f"{m['r2']:.4f}" if m['r2'] is not None else "N/A"
    print(f"  {m['label']}: R2={r2s}, RMSE={m['rmse']:,.0f}, MAE={m['mae']:,.0f}")

# Per-region daily
print("\n  Per-region daily (model):")
for j, reg in enumerate(REGIONS):
    r2 = r2_score(yte[:,j], preds[:,j])
    rmse = np.sqrt(mean_squared_error(yte[:,j], preds[:,j]))
    nz = (yte[:,j] > 0).sum()
    print(f"    {reg:<35} R2={r2:>7.4f} RMSE={rmse:>12,.0f} n_nz={nz}")

# ============ WEEKLY ============
test_df['week'] = test_df['date'].dt.to_period('W').astype(str)
test_df['month'] = test_df['date'].dt.to_period('M').astype(str)

# Weekly pooled
wk = test_df.groupby('week').agg(y_true=('y_true','sum'), y_pred=('y_pred','sum')).reset_index()
wk_base = test_df.groupby('week').agg(y_true=('y_true','sum')).reset_index()
wk_base['y_pred'] = train_mean.sum()  # weekly baseline = sum of daily train means * num regions

w_model = metrics(wk['y_true'], wk['y_pred'], "Weekly - Model")
w_base_row = wk_base.copy()
# Better weekly baseline: average daily train_mean * 7 days * N_regions (but sum across regions)
weekly_baseline_daily = np.tile(train_mean, (yte.shape[0], 1))
test_df['y_baseline'] = weekly_baseline_daily.flatten()[:len(test_df)]
wk_b = test_df.groupby('week').agg(y_true=('y_true','sum'), y_base=('y_baseline','sum')).reset_index()
w_baseline = metrics(wk_b['y_true'], wk_b['y_base'], "Weekly - TrainMean baseline")

print("\n" + "="*70)
print("WEEKLY (pooled)")
print("="*70)
for m in [w_model, w_baseline]:
    r2s = f"{m['r2']:.4f}" if m['r2'] is not None else "N/A"
    print(f"  {m['label']}: R2={r2s}, RMSE={m['rmse']:,.0f}, MAE={m['mae']:,.0f}")

# ============ MONTHLY ============
mo = test_df.groupby('month').agg(y_true=('y_true','sum'), y_pred=('y_pred','sum')).reset_index()
mo_b = test_df.groupby('month').agg(y_true=('y_true','sum'), y_base=('y_baseline','sum')).reset_index()
m_model = metrics(mo['y_true'], mo['y_pred'], "Monthly - Model")
m_baseline = metrics(mo_b['y_true'], mo_b['y_base'], "Monthly - TrainMean baseline")

print("\n" + "="*70)
print("MONTHLY (pooled)")
print("="*70)
for m in [m_model, m_baseline]:
    r2s = f"{m['r2']:.4f}" if m['r2'] is not None else "N/A"
    print(f"  {m['label']}: R2={r2s}, RMSE={m['rmse']:,.0f}, MAE={m['mae']:,.0f}")

# ============ ZERO FRACTION PER GRANULARITY ============
daily_zero = (yte.flatten() == 0).mean()
wk_zero = (wk['y_true'] == 0).mean()
mo_zero = (mo['y_true'] == 0).mean()
print(f"\nZero fraction: daily={daily_zero:.1%}, weekly={wk_zero:.1%}, monthly={mo_zero:.1%}")

# ============ KEY FINDING ============
print("\n" + "="*70)
print("KEY FINDING")
print("="*70)
print(f"Model R2 degrades at coarser granularity:")
print(f"  Daily:   R2={d_model['r2']:.4f} (baseline R2={d_baseline['r2']:.4f})")
print(f"  Weekly:  R2={w_model['r2']:.4f} (baseline R2={w_baseline['r2']:.4f})")
print(f"  Monthly: R2={m_model['r2']:.4f} (baseline R2={m_baseline['r2']:.4f})")
print(f"\nZero fraction drops: {daily_zero:.1%} -> {wk_zero:.1%} -> {mo_zero:.1%}")
print("The model's main advantage is predicting daily zero vs non-zero.")
print("At coarser granularity, magnitude errors compound and zero-prediction advantage is lost.")

# Save
results = {
    'daily': d_model, 'daily_baseline': d_baseline,
    'weekly': w_model, 'weekly_baseline': w_baseline,
    'monthly': m_model, 'monthly_baseline': m_baseline,
    'zero_fraction': {'daily': daily_zero, 'weekly': wk_zero, 'monthly': mo_zero},
}
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved: {OUT}")
