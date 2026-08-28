"""
Aggregation analysis v4: Comprehensive framing for Discussion.
Also includes a FIXED model baseline: per-region mean prediction 
to show what R2 SHOULD look like if the model captured regional patterns.
"""
import numpy as np, pandas as pd, json, os
from sklearn.metrics import r2_score, mean_squared_error

ROOT = r"E:\Download\Jurnal"
DATA = os.path.join(ROOT, "data", "view_penjualan_detail data hingga oktober.xlsx")

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

with open(os.path.join(ROOT, "results", "experiment_results_zi_geocoded.json")) as f:
    res = json.load(f)

preds = np.array(res['models']['HybridTuned']['preds_mean'])

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

# Build test df
rows = []
for i, tdate in enumerate(t_test):
    for j, reg in enumerate(REGIONS):
        rows.append({'date': tdate.start_time, 'region': reg, 'y_true': yte[i,j], 'y_pred': preds[i,j]})
test_df = pd.DataFrame(rows)
test_df['date'] = pd.to_datetime(test_df['date'])

# Summary stats
print("=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

# Prediction pattern
for j, reg in enumerate(REGIONS):
    n_pred_nz = (preds[:,j] > 0).sum()
    n_true_nz = (yte[:,j] > 0).sum()
    mean_pred = preds[:,j].mean()
    mean_true = yte[:,j].mean()
    print(f"  {reg:<35} pred_nz={n_pred_nz:>3}/231  true_nz={n_true_nz:>3}/231  pred_mean={mean_pred:>12,.0f}  true_mean={mean_true:>12,.0f}")

# Total test sales
total_true = yte.sum(axis=0)
total_pred = preds.sum(axis=0)
print(f"\n  Total test sales (actual): {yte.sum():,.0f}")
print(f"  Total test sales (pred):   {preds.sum():,.0f}")
print(f"  Ratio pred/actual:         {preds.sum()/yte.sum():.2f}")

# R2 decomposition: how much of pooled R2 comes from Palembang?
# Remove Palembang and recalculate
idx_palembang = REGIONS.index("Palembang")
mask_non_pal = np.ones(N, dtype=bool)
mask_non_pal[idx_palembang] = False

r2_all = r2_score(yte.flatten(), preds.flatten())
r2_no_pal = r2_score(yte[:, mask_non_pal].flatten(), preds[:, mask_non_pal].flatten())
r2_pal = r2_score(yte[:, idx_palembang], preds[:, idx_palembang])
rmse_pal = np.sqrt(mean_squared_error(yte[:, idx_palembang], preds[:, idx_palembang]))
rmse_no_pal = np.sqrt(mean_squared_error(yte[:, mask_non_pal].flatten(), preds[:, mask_non_pal].flatten()))

print(f"\n  R2 (all 16 regions):     {r2_all:.6f}")
print(f"  R2 (Palembang only):     {r2_pal:.6f}")
print(f"  R2 (15 other regions):   {r2_no_pal:.6f}")
print(f"  RMSE (Palembang):        {rmse_pal:,.0f}")
print(f"  RMSE (15 others):        {rmse_no_pal:,.0f}")

# What would a trivial per-region mean model give?
tr_pred = np.tile(train_mean, (yte.shape[0], 1))
r2_trivial_all = r2_score(yte.flatten(), tr_pred.flatten())
r2_trivial_pal = r2_score(yte[:, idx_palembang], tr_pred[:, idx_palembang])
r2_trivial_no_pal = r2_score(yte[:, mask_non_pal].flatten(), tr_pred[:, mask_non_pal].flatten())
rmse_trivial = np.sqrt(mean_squared_error(yte.flatten(), tr_pred.flatten()))

print(f"\n  TRIVIAL BASELINE (train_mean):")
print(f"  R2 (all):     {r2_trivial_all:.6f}")
print(f"  R2 (Pal):     {r2_trivial_pal:.6f}")
print(f"  R2 (15 oth):  {r2_trivial_no_pal:.6f}")
print(f"  RMSE:         {rmse_trivial:,.0f}")
print(f"\n  Model RMSE:   {np.sqrt(mean_squared_error(yte.flatten(), preds.flatten())):,.0f}")

# Palembang prediction pattern
pal_pred = preds[:, idx_palembang]
pal_true = yte[:, idx_palembang]
print(f"\n  PALEMBANG prediction analysis:")
print(f"    Pred std:   {pal_pred.std():,.0f}")
print(f"    True std:   {pal_true.std():,.0f}")
print(f"    Pred range: {pal_pred.min():,.0f} - {pal_pred.max():,.0f}")
print(f"    True range: {pal_true.min():,.0f} - {pal_true.max():,.0f}")
print(f"    Pred all same? {np.all(pal_pred == pal_pred[0])}")

# Zero-inflation classification accuracy
true_zero = (yte == 0).flatten()
pred_zero = (preds == 0).flatten()
acc = (true_zero == pred_zero).mean()
# True positive: correctly predict non-zero when actually non-zero
tp = ((~true_zero) & (~pred_zero)).sum()
fn = ((~true_zero) & pred_zero).sum()
fp = (true_zero & (~pred_zero)).sum()
tn = (true_zero & pred_zero).sum()
print(f"\n  Zero-inflation classification:")
print(f"    Accuracy: {acc:.1%}")
print(f"    TP (correct non-zero): {tp}")
print(f"    FN (missed non-zero):  {fn} ({fn/(tp+fn)*100:.0f}% of actual non-zero)")
print(f"    FP (false alarm):      {fp}")
print(f"    TN (correct zero):     {tn}")
