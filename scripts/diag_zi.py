"""
Diagnostic: Why does the model predict zero for 15/16 regions?
1. Analyze ZI classifier (pclf) predictions per region
2. Test different thresholds
3. Test regression-only (no ZI gating)
4. Check if weighted BCE helps
"""
import numpy as np, pandas as pd, json, os, sys
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score

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

# Load clf probabilities
pclf_42 = np.load(os.path.join(ROOT, "results", "v3_pclf_seed42.npy"))
# Reload main experiment results for preds
with open(os.path.join(ROOT, "results", "experiment_results_zi_geocoded.json")) as f:
    res = json.load(f)
preds = np.array(res['models']['HybridTuned']['preds_mean'])

# Rebuild yte and p_amt (regression-only predictions)
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
ybin_te = (yte > 0).astype(float)

# pclf is from v3 model, but preds is from the main HybridTuned model
# They should be similar. Let's analyze pclf per region
print("=" * 70)
print("ZI CLASSIFIER (p_clf) ANALYSIS")
print("=" * 70)
print(f"p_clf shape: {pclf_42.shape}, yte shape: {yte.shape}")

for j, reg in enumerate(REGIONS):
    pc = pclf_42[:, j]
    yb = ybin_te[:, j]
    nz_true = int(yb.sum())
    nz_pred_05 = (pc > 0.5).sum()
    nz_pred_03 = (pc > 0.3).sum()
    nz_pred_01 = (pc > 0.1).sum()
    mean_pc = pc.mean()
    max_pc = pc.max()
    min_pc = pc.min()
    auc_j = roc_auc_score(yb, pc) if len(np.unique(yb)) > 1 else float('nan')
    print(f"  {reg:<35} nz_true={nz_true:>3} mean_pclf={mean_pc:.4f} max={max_pc:.4f} "
          f"pred>0.5={nz_pred_05:>3} pred>0.3={nz_pred_03:>3} pred>0.1={nz_pred_01:>3} AUC={auc_j:.3f}")

# Threshold sweep for the whole panel
print("\n" + "=" * 70)
print("THRESHOLD SWEEP (pooled)")
print("=" * 70)
for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    gated = (pclf_42 > thr).astype(float) * preds
    r2 = r2_score(yte.flatten(), gated.flatten())
    rmse = np.sqrt(mean_squared_error(yte.flatten(), gated.flatten()))
    nz_pred = (gated > 0).sum()
    print(f"  thr={thr:.1f}  R2={r2:.4f}  RMSE={rmse:>14,.0f}  n_nz_pred={nz_pred}")

# Regression-only (no ZI gating): just use p_amt directly
# We need to reconstruct p_amt = preds / (p_clf > 0.5) but that's undefined when p_clf <= 0.5
# Instead: p_amt is the raw regression output. We know preds = (p_clf > 0.5) * p_amt
# So p_amt[i,j] = preds[i,j] / 1.0 where p_clf > 0.5, and unknown otherwise.
# But we can also load the v3 model predictions which might give us p_amt separately.

# Let's try: what if we bypass the ZI gate entirely?
# Use pclf as soft weights instead of hard threshold
print("\n" + "=" * 70)
print("SOFT GATING (p_clf as continuous weight)")
print("=" * 70)
for scale in [0.5, 1.0, 1.5, 2.0, 3.0]:
    soft_pred = pclf_42 * preds  # soft gate: weight by probability
    r2 = r2_score(yte.flatten(), soft_pred.flatten())
    rmse = np.sqrt(mean_squared_error(yte.flatten(), soft_pred.flatten()))
    print(f"  scale=1.0*  R2={r2:.4f}  RMSE={rmse:>14,.0f}")

# What about using train_mean as regression (trivial baseline for p_amt)?
train_end_mask = np.array([m.start_time <= pd.Timestamp("2023-12-31") for m in t_all])
ytr = y_all[train_end_mask]
train_mean = ytr.mean(axis=0)  # per-region mean
print("\n" + "=" * 70)
print("COMBINATION: pclf + train_mean (what if regression was perfect?)")
print("=" * 70)
tr_pred_by_region = np.tile(train_mean, (yte.shape[0], 1))
for thr in [0.1, 0.2, 0.3, 0.5, 0.7]:
    combo = (pclf_42 > thr).astype(float) * tr_pred_by_region
    r2 = r2_score(yte.flatten(), combo.flatten())
    rmse = np.sqrt(mean_squared_error(yte.flatten(), combo.flatten()))
    print(f"  thr={thr:.1f}  R2={r2:.4f}  RMSE={rmse:>14,.0f}")

# What if we use train_mean without ZI gating?
r2_direct = r2_score(yte.flatten(), tr_pred_by_region.flatten())
rmse_direct = np.sqrt(mean_squared_error(yte.flatten(), tr_pred_by_region.flatten()))
print(f"\n  Direct train_mean (no gate): R2={r2_direct:.4f}  RMSE={rmse_direct:,.0f}")
