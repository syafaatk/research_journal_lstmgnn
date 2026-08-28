"""
Eksperimen perbaikan ZI model:
1. Weighted BCE (pos_weight = n_neg/n_pos) untuk classifier
2. Combined loss: BCE + lambda * masked-MSE (joint training)
3. Threshold tuning: test optimal threshold on validation set

Root cause: classifier only outputs p > 0.5 for Palembang (1/16 regions).
84.8% zero fraction causes BCE to dominate and collapse to majority class.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, json, os
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score

ROOT = r"E:\Download\Jurnal"
DATA = os.path.join(ROOT, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(ROOT, "results", "exp_zi_fix.json")

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

# ---------------------------------------------------------------------------
# 1. Build panel (same as main pipeline)
# ---------------------------------------------------------------------------
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

# Build windowed data
WINDOW = 30
X_all, y_all, t_all = [], [], []
for i in range(n_days - WINDOW):
    sales = panel.iloc[i:i + WINDOW].values
    flags = np.tile(covid_flag[i:i + WINDOW][:, None], (1, N))
    X_all.append(np.stack([sales, flags], axis=-1))
    y_all.append(panel.iloc[i + WINDOW].values)
    t_all.append(panel.index[i + WINDOW])
X_all = np.array(X_all).transpose(0, 2, 1, 3)
y_all = np.array(y_all)

tr_mask = np.array([m.start_time <= pd.Timestamp("2023-12-31") for m in t_all])
va_mask = np.array([(m.start_time > pd.Timestamp("2023-12-31")) & (m.start_time <= pd.Timestamp("2024-12-31")) for m in t_all])
te_mask = np.array([m.start_time > pd.Timestamp("2024-12-31") for m in t_all])
Xtr, ytr = X_all[tr_mask], y_all[tr_mask]
Xva, yva = X_all[va_mask], y_all[va_mask]
Xte, yte = X_all[te_mask], y_all[te_mask]

# Normalize
mu = np.log1p(Xtr[:, :, :, 0]).mean()
sd = np.log1p(Xtr[:, :, :, 0]).std()
X_all_n = X_all.copy()
X_all_n[:, :, :, 0] = (np.log1p(X_all[:, :, :, 0]) - mu) / sd
Xtr_n, Xva_n, Xte_n = X_all_n[tr_mask], X_all_n[va_mask], X_all_n[te_mask]

ybin_tr = (ytr > 0).astype(np.float32)
ybin_va = (yva > 0).astype(np.float32)
ybin_te = (yte > 0).astype(np.float32)

# Adjacency (distance k=3)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

geo = pd.read_csv(os.path.join(ROOT, "data", "customers_geocoded_final.csv"))
geo = geo[geo["status"].isin(["FOUND_GEO", "FOUND_SIRS"])].copy()
inv_per_cust = df.groupby("pelanggan_nama")["d_jual_nofak"].nunique().rename("n_inv_data")
geo = geo.merge(inv_per_cust, left_on="pelanggan_nama", right_index=True, how="left")
geo["n_inv_data"] = geo["n_inv_data"].fillna(0)
w = geo.groupby("region").apply(
    lambda g: pd.Series({
        "lat": (g["lat"] * g["n_inv_data"]).sum() / g["n_inv_data"].sum(),
        "lon": (g["lon"] * g["n_inv_data"]).sum() / g["n_inv_data"].sum(),
    }), include_groups=False)
coords = w[["lat", "lon"]].reindex(REGIONS)

D = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        D[i, j] = haversine(coords.iloc[i].lat, coords.iloc[i].lon,
                            coords.iloc[j].lat, coords.iloc[j].lon)
K = 3
knn_dists = np.array([np.sort(D[i])[1:K+1] for i in range(N)])
THRESHOLD = float(knn_dists.max())
adj = np.zeros((N, N))
for i in range(N):
    order = np.argsort(D[i])
    for j in order[1:K+1]:
        if D[i, j] <= THRESHOLD:
            adj[i, j] = 1
            adj[j, i] = 1
np.fill_diagonal(adj, 0)

def build_adj(adj_matrix):
    A_tilde = adj_matrix + np.eye(N)
    D_tilde = A_tilde.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D_tilde))
    return torch.tensor(D_inv_sqrt @ A_tilde @ D_inv_sqrt, dtype=torch.float32)

A_norm_t = build_adj(adj)

# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
class HybridModel(nn.Module):
    def __init__(self, hidden_lstm=128, hidden_gnn=64, dropout=0.0, in_features=2):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_lstm, batch_first=True)
        self.gcn = nn.Linear(hidden_lstm, hidden_gnn)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_lstm + hidden_gnn, 1)

    def forward(self, x):
        B, N_, T, F = x.shape
        h, _ = self.lstm(x.reshape(B * N_, T, F))
        h_lstm = h[:, -1, :].view(B, N_, -1)
        h_gcn = torch.relu(self.gcn(h_lstm))
        h_gcn = A_norm_t @ h_gcn
        fused = torch.cat([h_lstm, h_gcn], dim=-1)
        return self.fc(self.drop(fused)).view(B, N_)

def make_model():
    return HybridModel(hidden_lstm=128, hidden_gnn=64, dropout=0.2, in_features=2)

# ---------------------------------------------------------------------------
# 3. Training functions
# ---------------------------------------------------------------------------
def train_clf(model, Xtr, ytr, Xva, yva, pos_weight, lr=0.001, bs=64, epochs=60, patience=8, seed=42):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pw = torch.tensor([pos_weight], dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    yv = torch.tensor(yva, dtype=torch.float32)
    best_loss, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xv), yv).item()
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model

def train_amt(model, Xtr, ytr, Xva, yva, lr=0.001, bs=64, epochs=60, patience=8, seed=42):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt_log = torch.tensor(np.log1p(ytr), dtype=torch.float32)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    yv_log = torch.tensor(np.log1p(yva), dtype=torch.float32)
    m_tr = torch.tensor((ytr > 0).astype(float), dtype=torch.float32)
    m_va = torch.tensor((yva > 0).astype(float), dtype=torch.float32)
    best_loss, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            out = model(Xt[idx])
            m = m_tr[idx]
            loss = ((out - yt_log[idx]) ** 2 * m).sum() / m.sum()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out_v = model(Xv)
            vloss = ((out_v - yv_log) ** 2 * m_va).sum() / m_va.sum().item()
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model

def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"R2": r2, "RMSE": rmse, "MAE": mae}

# ---------------------------------------------------------------------------
# 4. Experiments
# ---------------------------------------------------------------------------
pos_neg = (ybin_tr == 0).sum() / max((ybin_tr == 1).sum(), 1)
print(f"Train zero fraction: {(ybin_tr == 0).mean():.1%}")
print(f"pos_weight (n_neg/n_pos): {pos_neg:.2f}")
print(f"Train: {len(ytr)}, Val: {len(yva)}, Test: {len(yte)}")

results = {}
SEEDS = [42, 7, 123]

# === Experiment A: Baseline (original) ===
print("\n" + "=" * 70)
print("A. BASELINE (original BCE, threshold 0.5)")
print("=" * 70)
preds_a = []
for seed in SEEDS:
    clf = make_model()
    clf = train_clf(clf, Xtr_n, ybin_tr, Xva_n, ybin_va, pos_weight=1.0, seed=seed)
    p_clf = 1 / (1 + np.exp(-predict(clf, Xte_n)))
    amt = make_model()
    amt = train_amt(amt, Xtr_n, ytr, Xva_n, yva, seed=seed)
    p_amt = np.expm1(predict(amt, Xte_n))
    p_amt = np.clip(p_amt, 0, None)
    pred = (p_clf > 0.5).astype(float) * p_amt
    m = metrics(yte, pred)
    preds_a.append(pred)
    print(f"  seed={seed} R2={m['R2']:.4f} RMSE={m['RMSE']:,.0f}")
ens_a = np.mean(preds_a, axis=0)
m_ens_a = metrics(yte, ens_a)
print(f"  Ensemble: R2={m_ens_a['R2']:.4f} RMSE={m_ens_a['RMSE']:,.0f}")
results['A_baseline'] = m_ens_a

# === Experiment B: Weighted BCE (pos_weight = n_neg/n_pos) ===
print("\n" + "=" * 70)
print("B. WEIGHTED BCE (pos_weight=%.1f)" % pos_neg)
print("=" * 70)
preds_b = []
pclf_b_all = []
for seed in SEEDS:
    clf = make_model()
    clf = train_clf(clf, Xtr_n, ybin_tr, Xva_n, ybin_va, pos_weight=pos_neg, seed=seed)
    p_clf = 1 / (1 + np.exp(-predict(clf, Xte_n)))
    pclf_b_all.append(p_clf)
    amt = make_model()
    amt = train_amt(amt, Xtr_n, ytr, Xva_n, yva, seed=seed)
    p_amt = np.expm1(predict(amt, Xte_n))
    p_amt = np.clip(p_amt, 0, None)
    pred = (p_clf > 0.5).astype(float) * p_amt
    m = metrics(yte, pred)
    preds_b.append(pred)
    print(f"  seed={seed} R2={m['R2']:.4f} RMSE={m['RMSE']:,.0f}")
ens_b = np.mean(preds_b, axis=0)
m_ens_b = metrics(yte, ens_b)
print(f"  Ensemble: R2={m_ens_b['R2']:.4f} RMSE={m_ens_b['RMSE']:,.0f}")
results['B_weighted_bce'] = m_ens_b

# Analyze pclf from experiment B
pclf_b_avg = np.mean(pclf_b_all, axis=0)
print("\n  Per-region clf analysis (weighted BCE):")
for j, reg in enumerate(REGIONS):
    pc = pclf_b_avg[:, j]
    nz_true = int(ybin_te[:, j].sum())
    nz_pred_05 = (pc > 0.5).sum()
    nz_pred_03 = (pc > 0.3).sum()
    mean_pc = pc.mean()
    max_pc = pc.max()
    auc_j = roc_auc_score(ybin_te[:, j], pc) if len(np.unique(ybin_te[:, j])) > 1 else float('nan')
    print(f"    {reg:<35} nz_true={nz_true:>3} mean_pclf={mean_pc:.4f} max={max_pc:.4f} "
          f"pred>0.5={nz_pred_05:>3} pred>0.3={nz_pred_03:>3} AUC={auc_j:.3f}")

# === Experiment B2: Weighted BCE + threshold tuning on val ===
print("\n" + "=" * 70)
print("B2. WEIGHTED BCE + THRESHOLD TUNING (find best thr on val)")
print("=" * 70)
# Re-train one seed for threshold search
clf_tune = make_model()
clf_tune = train_clf(clf_tune, Xtr_n, ybin_tr, Xva_n, ybin_va, pos_weight=pos_neg, seed=42)
p_clf_val = 1 / (1 + np.exp(-predict(clf_tune, Xva_n)))
p_clf_test = 1 / (1 + np.exp(-predict(clf_tune, Xte_n)))
amt_tune = make_model()
amt_tune = train_amt(amt_tune, Xtr_n, ytr, Xva_n, yva, seed=42)
p_amt_val = np.expm1(predict(amt_tune, Xva_n))
p_amt_val = np.clip(p_amt_val, 0, None)
p_amt_test = np.expm1(predict(amt_tune, Xte_n))
p_amt_test = np.clip(p_amt_test, 0, None)

best_thr, best_r2 = 0.5, -999
for thr in np.arange(0.05, 0.95, 0.05):
    pred_val = (p_clf_val > thr).astype(float) * p_amt_val
    r2_val = r2_score(yva.flatten(), pred_val.flatten())
    if r2_val > best_r2:
        best_r2 = r2_val
        best_thr = thr
print(f"  Best threshold on val: {best_thr:.2f} (val R2={best_r2:.4f})")
pred_b2 = (p_clf_test > best_thr).astype(float) * p_amt_test
m_b2 = metrics(yte, pred_b2)
print(f"  Test: R2={m_b2['R2']:.4f} RMSE={m_b2['RMSE']:,.0f}")
results['B2_weighted_thr_tuned'] = m_b2

# === Experiment C: Joint training (BCE + masked-MSE combined) ===
print("\n" + "=" * 70)
print("C. JOINT TRAINING (BCE + lambda * masked MSE)")
print("=" * 70)

def train_joint(model, Xtr, ytr, ybin, Xva, yva, ybin_va,
                pos_weight, lam=0.5, lr=0.001, bs=64, epochs=60, patience=8, seed=42):
    """Joint training: loss = BCE + lam * masked_MSE (on log1p scale)"""
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pw = torch.tensor([pos_weight], dtype=torch.float32)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt_bin = torch.tensor(ybin, dtype=torch.float32)
    yt_log = torch.tensor(np.log1p(ytr), dtype=torch.float32)
    mask_tr = torch.tensor((ytr > 0).astype(float), dtype=torch.float32)
    Xv = torch.tensor(Xva, dtype=torch.float32)
    yv_bin = torch.tensor(ybin_va, dtype=torch.float32)
    yv_log = torch.tensor(np.log1p(yva), dtype=torch.float32)
    mask_va = torch.tensor((yva > 0).astype(float), dtype=torch.float32)
    best_loss, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            out = model(Xt[idx])
            l_bce = bce(out, yt_bin[idx])
            m = mask_tr[idx]
            l_mse = ((out - yt_log[idx]) ** 2 * m).sum() / m.sum()
            loss = l_bce + lam * l_mse
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out_v = model(Xv)
            vb = bce(out_v, yv_bin).item()
            vm = ((out_v - yv_log) ** 2 * mask_va).sum() / mask_va.sum().item()
            vloss = vb + lam * vm
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model

for lam in [0.1, 0.5, 1.0]:
    print(f"\n  lambda={lam}:")
    preds_c = []
    for seed in SEEDS:
        m_joint = make_model()
        m_joint = train_joint(m_joint, Xtr_n, ytr, ybin_tr, Xva_n, yva, ybin_va,
                             pos_weight=pos_neg, lam=lam, seed=seed)
        p_clf = 1 / (1 + np.exp(-predict(m_joint, Xte_n)))
        p_amt = np.expm1(predict(m_joint, Xte_n))
        p_amt = np.clip(p_amt, 0, None)
        pred = (p_clf > 0.5).astype(float) * p_amt
        m = metrics(yte, pred)
        preds_c.append(pred)
        print(f"    seed={seed} R2={m['R2']:.4f} RMSE={m['RMSE']:,.0f}")
    ens_c = np.mean(preds_c, axis=0)
    m_ens_c = metrics(yte, ens_c)
    print(f"    Ensemble: R2={m_ens_c['R2']:.4f} RMSE={m_ens_c['RMSE']:,.0f}")
    results[f'C_joint_lam{lam}'] = m_ens_c

# === Experiment D: Regression only (no ZI gate) ===
print("\n" + "=" * 70)
print("D. REGRESSION ONLY (no ZI gating, direct prediction)")
print("=" * 70)
preds_d = []
for seed in SEEDS:
    amt = make_model()
    amt = train_amt(amt, Xtr_n, ytr, Xva_n, yva, seed=seed)
    p_amt = np.expm1(predict(amt, Xte_n))
    p_amt = np.clip(p_amt, 0, None)
    m = metrics(yte, p_amt)
    preds_d.append(p_amt)
    print(f"  seed={seed} R2={m['R2']:.4f} RMSE={m['RMSE']:,.0f}")
ens_d = np.mean(preds_d, axis=0)
m_ens_d = metrics(yte, ens_d)
print(f"  Ensemble: R2={m_ens_d['R2']:.4f} RMSE={m_ens_d['RMSE']:,.0f}")
results['D_regr_only'] = m_ens_d

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
for name, m in sorted(results.items()):
    print(f"  {name:<35} R2={m['R2']:.4f}  RMSE={m['RMSE']:>14,.0f}")

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {OUT}")
