"""
Eksperimen agregasi mingguan/bulanan:
Rebuild panel persis seperti pipeline utama, lalu:
1. Load prediksi V3 dari npy cache
2. Aggregate y_true dan y_pred ke weekly dan monthly
3. Hitung R2/RMSE/MAE di setiap level
4. Bandingkan dengan harian

Juga: per-region breakdown untuk framing di Discussion.
"""
import numpy as np, pandas as pd, json, os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

ROOT = r"E:\Download\Jurnal"
DATA = os.path.join(ROOT, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT_AGG = os.path.join(ROOT, "results", "exp_aggregation.json")

# ====== 1. Load & build panel (same as experiment_zi_geocoded.py) ======
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
days = panel.index.astype(str).tolist()
n_days = len(panel)
print(f"Panel: {n_days} hari ({days[0]} s.d. {days[-1]}), {N} region")

COVID_END = pd.Timestamp("2022-12-31")
covid_flag = np.array([1.0 if m.start_time <= COVID_END else 0.0 for m in panel.index])

# ====== 2. Build windowed samples (same as pipeline) ======
WINDOW = 30
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

ytr, yva, yte = y_all[tr_mask], y_all[va_mask], y_all[te_mask]
t_test = [t for t, m in zip(t_all, te_mask) if m]
print(f"Sampel: train {len(ytr)}, val {len(yva)}, test {len(yte)}")
print(f"Test target dates: {t_test[0]} s.d. {t_test[-1]}")

# ====== 3. Load predictions (V3 ensemble) ======
preds_42 = np.load(os.path.join(ROOT, "results", "v3_preds_seed42.npy"))
preds_7 = np.load(os.path.join(ROOT, "results", "v3_preds_seed7.npy"))
preds_123 = np.load(os.path.join(ROOT, "results", "v3_preds_seed123.npy"))
preds_ensemble = (preds_42 + preds_7 + preds_123) / 3.0
print(f"Predictions: shape={preds_ensemble.shape}, yte shape={yte.shape}")

assert preds_ensemble.shape == yte.shape, f"Shape mismatch: preds={preds_ensemble.shape} vs yte={yte.shape}"

# ====== 4. Build test DataFrame ======
rows = []
for i, tdate in enumerate(t_test):
    for j, reg in enumerate(REGIONS):
        rows.append({
            'date': tdate.start_time,
            'region': reg,
            'y_true': yte[i, j],
            'y_pred': preds_ensemble[i, j],
        })
test_df = pd.DataFrame(rows)
test_df['date'] = pd.to_datetime(test_df['date'])
print(f"Test DataFrame: {len(test_df)} rows ({len(t_test)} days x {N} regions)")

# ====== 5. Metrics helpers ======
def calc_metrics(y_true, y_pred, label=""):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true > 0
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    if mask.sum() > 1:
        r2_nz = r2_score(y_true[mask], y_pred[mask])
        rmse_nz = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    else:
        r2_nz, rmse_nz = np.nan, np.nan
    return {
        'label': label,
        'n': len(y_true),
        'n_nonzero': int(mask.sum()),
        'r2_all': round(r2, 6) if not np.isnan(r2) else None,
        'rmse': round(float(rmse), 0),
        'mae': round(float(mae), 0),
        'r2_nonzero': round(float(r2_nz), 6) if not np.isnan(r2_nz) else None,
        'rmse_nonzero': round(float(rmse_nz), 0) if not np.isnan(rmse_nz) else None,
    }

# ====== 6. DAILY metrics (pooled + per-region) ======
daily_metrics = calc_metrics(test_df['y_true'], test_df['y_pred'], "Daily (pooled)")
daily_per_region = {}
for reg in REGIONS:
    sub = test_df[test_df['region'] == reg]
    daily_per_region[reg] = calc_metrics(sub['y_true'], sub['y_pred'], f"Daily-{reg}")

# ====== 7. WEEKLY aggregation ======
test_df['week'] = test_df['date'].dt.isocalendar().year.astype(str) + '-W' + test_df['date'].dt.isocalendar().week.astype(str).str.zfill(2)
weekly_pooled = test_df.groupby('week').agg(y_true=('y_true', 'sum'), y_pred=('y_pred', 'sum')).reset_index()
weekly_metrics = calc_metrics(weekly_pooled['y_true'], weekly_pooled['y_pred'], "Weekly (pooled)")

weekly = test_df.groupby(['week', 'region']).agg(y_true=('y_true', 'sum'), y_pred=('y_pred', 'sum')).reset_index()
weekly_per_region = {}
for reg in REGIONS:
    sub = weekly[weekly['region'] == reg]
    weekly_per_region[reg] = calc_metrics(sub['y_true'], sub['y_pred'], f"Weekly-{reg}")

# ====== 8. MONTHLY aggregation ======
test_df['month'] = test_df['date'].dt.to_period('M').astype(str)
monthly_pooled = test_df.groupby('month').agg(y_true=('y_true', 'sum'), y_pred=('y_pred', 'sum')).reset_index()
monthly_metrics = calc_metrics(monthly_pooled['y_true'], monthly_pooled['y_pred'], "Monthly (pooled)")

monthly = test_df.groupby(['month', 'region']).agg(y_true=('y_true', 'sum'), y_pred=('y_pred', 'sum')).reset_index()
monthly_per_region = {}
for reg in REGIONS:
    sub = monthly[monthly['region'] == reg]
    monthly_per_region[reg] = calc_metrics(sub['y_true'], sub['y_pred'], f"Monthly-{reg}")

# ====== 9. Print summary ======
print("\n" + "=" * 90)
print("METRICS PER GRANULARITAS (POOLED)")
print("=" * 90)
for m in [daily_metrics, weekly_metrics, monthly_metrics]:
    print(f"\n  {m['label']}:")
    print(f"    n={m['n']}, non-zero={m['n_nonzero']}")
    print(f"    R2 (all):   {m['r2_all']}")
    print(f"    RMSE:       {m['rmse']:>14,.0f}")
    print(f"    MAE:        {m['mae']:>14,.0f}")
    if m['r2_nonzero'] is not None:
        print(f"    R2 (nz):    {m['r2_nonzero']}")
        print(f"    RMSE (nz):  {m['rmse_nonzero']:>14,.0f}")

print("\n" + "=" * 90)
print("PER-REGION DAILY")
print("=" * 90)
print(f"{'Region':<35} | {'R2':>8} | {'RMSE':>14} | {'MAE':>14} | {'n_nz':>5}")
print("-" * 90)
for reg in REGIONS:
    m = daily_per_region[reg]
    r2s = f"{m['r2_all']:.4f}" if m['r2_all'] is not None else "N/A"
    print(f"{reg:<35} | {r2s:>8} | {m['rmse']:>14,.0f} | {m['mae']:>14,.0f} | {m['n_nonzero']:>5}")

print("\n" + "=" * 90)
print("PER-REGION WEEKLY")
print("=" * 90)
print(f"{'Region':<35} | {'R2':>8} | {'RMSE':>14} | {'MAE':>14}")
print("-" * 90)
for reg in REGIONS:
    m = weekly_per_region[reg]
    r2s = f"{m['r2_all']:.4f}" if m['r2_all'] is not None else "N/A"
    print(f"{reg:<35} | {r2s:>8} | {m['rmse']:>14,.0f} | {m['mae']:>14,.0f}")

print("\n" + "=" * 90)
print("PER-REGION MONTHLY")
print("=" * 90)
print(f"{'Region':<35} | {'R2':>8} | {'RMSE':>14} | {'MAE':>14}")
print("-" * 90)
for reg in REGIONS:
    m = monthly_per_region[reg]
    r2s = f"{m['r2_all']:.4f}" if m['r2_all'] is not None else "N/A"
    print(f"{reg:<35} | {r2s:>8} | {m['rmse']:>14,.0f} | {m['mae']:>14,.0f}")

# ====== 10. Region ranking by total sales ======
region_sales = test_df.groupby('region')['y_true'].sum().sort_values(ascending=False)
print("\n" + "=" * 90)
print("REGION RANKING BY TEST-PERIOD SALES")
print("=" * 90)
for i, (reg, total) in enumerate(region_sales.items(), 1):
    frac = total / region_sales.sum() * 100
    m = daily_per_region[reg]
    r2s = f"{m['r2_all']:.4f}" if m['r2_all'] is not None else "N/A"
    print(f"  {i:2d}. {reg:<35} | {total/1e6:>10,.1f} M ({frac:>5.1f}%) | R2={r2s}")

# ====== 11. Zone grouping: vs vs small regions ======
big_regions = region_sales.head(4).index.tolist()  # Top 4 by sales
small_regions = region_sales.tail(12).index.tolist()

big_y_true = test_df[test_df['region'].isin(big_regions)].groupby('date')['y_true'].sum()
big_y_pred = test_df[test_df['region'].isin(big_regions)].groupby('date')['y_pred'].sum()
small_y_true = test_df[test_df['region'].isin(small_regions)].groupby('date')['y_true'].sum()
small_y_pred = test_df[test_df['region'].isin(small_regions)].groupby('date')['y_pred'].sum()

big_metrics = calc_metrics(big_y_true.values, big_y_pred.values, "Top-4 regions (daily)")
small_metrics = calc_metrics(small_y_true.values, small_y_pred.values, "Bottom-12 regions (daily)")

print("\n" + "=" * 90)
print(f"ZONE AGGREGATION: Top-4 ({', '.join(big_regions)}) vs Bottom-12")
print("=" * 90)
for m in [big_metrics, small_metrics]:
    print(f"\n  {m['label']}:")
    print(f"    R2 (all):  {m['r2_all']}")
    print(f"    RMSE:      {m['rmse']:>14,.0f}")
    print(f"    MAE:       {m['mae']:>14,.0f}")

# Weekly zone aggregation
weekly_big = test_df[test_df['region'].isin(big_regions)].groupby('week').agg(
    y_true=('y_true', 'sum'), y_pred=('y_pred', 'sum')).reset_index()
weekly_small = test_df[test_df['region'].isin(small_regions)].groupby('week').agg(
    y_true=('y_true', 'sum'), y_pred=('y_pred', 'sum')).reset_index()
big_w = calc_metrics(weekly_big['y_true'], weekly_big['y_pred'], "Top-4 (weekly)")
small_w = calc_metrics(weekly_small['y_true'], weekly_small['y_pred'], "Bottom-12 (weekly)")
print("\nWeekly aggregation:")
for m in [big_w, small_w]:
    print(f"  {m['label']}: R2={m['r2_all']}, RMSE={m['rmse']:,.0f}")

# Monthly zone aggregation
monthly_big = test_df[test_df['region'].isin(big_regions)].groupby('month').agg(
    y_true=('y_true', 'sum'), y_pred=('y_pred', 'sum')).reset_index()
monthly_small = test_df[test_df['region'].isin(small_regions)].groupby('month').agg(
    y_true=('y_true', 'sum'), y_pred=('y_pred', 'sum')).reset_index()
big_m = calc_metrics(monthly_big['y_true'], monthly_big['y_pred'], "Top-4 (monthly)")
small_m = calc_metrics(monthly_small['y_true'], monthly_small['y_pred'], "Bottom-12 (monthly)")
print("\nMonthly aggregation:")
for m in [big_m, small_m]:
    print(f"  {m['label']}: R2={m['r2_all']}, RMSE={m['rmse']:,.0f}")

# ====== 12. Save ======
results = {
    'pooled': {
        'daily': daily_metrics,
        'weekly': weekly_metrics,
        'monthly': monthly_metrics,
    },
    'per_region': {
        'daily': daily_per_region,
        'weekly': weekly_per_region,
        'monthly': monthly_per_region,
    },
    'zone': {
        'big4': {
            'regions': big_regions,
            'daily': big_metrics, 'weekly': big_w, 'monthly': big_m,
        },
        'small12': {
            'regions': small_regions,
            'daily': small_metrics, 'weekly': small_w, 'monthly': small_m,
        },
    },
    'region_sales_ranking': {
        reg: {'total_sales': float(total), 'share_pct': round(float(total / region_sales.sum() * 100), 1)}
        for reg, total in region_sales.items()
    },
}

with open(OUT_AGG, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"\nSaved: {OUT_AGG}")
