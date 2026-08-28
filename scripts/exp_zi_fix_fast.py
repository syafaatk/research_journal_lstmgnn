"""Minimal test: weighted BCE vs baseline, single seed, reuse cached data."""
import numpy as np, torch, torch.nn as nn, json, os, time
from sklearn.metrics import r2_score, mean_squared_error

t0 = time.time()
ROOT = r"E:\Download\Jurnal"
OUT = os.path.join(ROOT, "results", "exp_zi_fix_quick.json")

# Load cached npy data if available, otherwise rebuild
CACHE = os.path.join(ROOT, "results", "_cache_panel.npz")
if os.path.exists(CACHE):
    print("Loading cached panel...")
    d = np.load(CACHE, allow_pickle=True)
    X_all_n, y_all = d['X_all_n'], d['y_all']
    tr_mask, va_mask, te_mask = d['tr_mask'], d['va_mask'], d['te_mask']
    ybin_tr, ybin_va, ybin_te = d['ybin_tr'], d['ybin_va'], d['ybin_te']
    yte = d['yte']
    ytr = d['ytr']; yva = d['yva']
    print(f"Loaded: X_all_n={X_all_n.shape}, yte={yte.shape}")
else:
    print("Building panel from scratch (first run)...")
    import pandas as pd
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
    DATA = os.path.join(ROOT, "data", "view_penjualan_detail data hingga oktober.xlsx")
    df = pd.read_excel(DATA); print(f"Excel loaded: {time.time()-t0:.1f}s")
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
    X_all, y_all_list, t_all = [], [], []
    for i in range(n_days - WINDOW):
        sales = panel.iloc[i:i + WINDOW].values
        flags = np.tile(covid_flag[i:i + WINDOW][:, None], (1, N))
        X_all.append(np.stack([sales, flags], axis=-1))
        y_all_list.append(panel.iloc[i + WINDOW].values)
        t_all.append(panel.index[i + WINDOW])
    X_all = np.array(X_all).transpose(0, 2, 1, 3)
    y_all = np.array(y_all_list)
    tr_mask = np.array([m.start_time <= pd.Timestamp("2023-12-31") for m in t_all])
    va_mask = np.array([(m.start_time > pd.Timestamp("2023-12-31")) & (m.start_time <= pd.Timestamp("2024-12-31")) for m in t_all])
    te_mask = np.array([m.start_time > pd.Timestamp("2024-12-31") for m in t_all])
    Xtr = X_all[tr_mask]; mu = np.log1p(Xtr[:, :, :, 0]).mean(); sd = np.log1p(Xtr[:, :, :, 0]).std()
    X_all_n = X_all.copy()
    X_all_n[:, :, :, 0] = (np.log1p(X_all[:, :, :, 0]) - mu) / sd
    ybin_tr = (y_all[tr_mask] > 0).astype(np.float32)
    ybin_va = (y_all[va_mask] > 0).astype(np.float32)
    ybin_te = (y_all[te_mask] > 0).astype(np.float32)
    ytr = y_all[tr_mask]; yva = y_all[va_mask]; yte = y_all[te_mask]
    np.savez_compressed(CACHE, X_all_n=X_all_n, y_all=y_all, tr_mask=tr_mask, va_mask=va_mask, te_mask=te_mask,
                        ybin_tr=ybin_tr, ybin_va=ybin_va, ybin_te=ybin_te, yte=yte, ytr=ytr, yva=yva)
    print(f"Cached: {time.time()-t0:.1f}s")

# Subset
Xtr_n = X_all_n[tr_mask]; Xva_n = X_all_n[va_mask]; Xte_n = X_all_n[te_mask]
N = 16

# Adjacency (distance k=3)
GEO_CACHE = os.path.join(ROOT, "results", "_cache_adj.npy")
if os.path.exists(GEO_CACHE):
    adj = np.load(GEO_CACHE)
else:
    import pandas as pd
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dp = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
        a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))
    DATA = os.path.join(ROOT, "data", "view_penjualan_detail data hingga oktober.xlsx")
    geo = pd.read_csv(os.path.join(ROOT, "data", "customers_geocoded_final.csv"))
    geo = geo[geo["status"].isin(["FOUND_GEO", "FOUND_SIRS"])].copy()
    df2 = pd.read_excel(DATA, usecols=["d_jual_nofak", "pelanggan_nama"])
    df2 = df2.dropna(subset=["d_jual_nofak"])
    inv_per_cust = df2.groupby("pelanggan_nama")["d_jual_nofak"].nunique().rename("n_inv_data")
    geo = geo.merge(inv_per_cust, left_on="pelanggan_nama", right_index=True, how="left")
    geo["n_inv_data"] = geo["n_inv_data"].fillna(0)
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
    REGIONS = ["Banyuasin","Empat Lawang","Lahat","Lubuk Linggau","Muara Enim","Musi Banyuasin","Musi Rawas","Ogan Ilir","Ogan Komering Ilir","Ogan Komering Ulu","Ogan Komering Ulu Selatan","Ogan Komering Ulu Timur","Pagar Alam","Palembang","Penukal Abab Lematang Ilir","Prabumulih"]
    w = geo.groupby("region").apply(lambda g: pd.Series({"lat": (g["lat"]*g["n_inv_data"]).sum()/g["n_inv_data"].sum(), "lon": (g["lon"]*g["n_inv_data"]).sum()/g["n_inv_data"].sum()}), include_groups=False)
    coords = w[["lat","lon"]].reindex(REGIONS)
    D = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            D[i,j] = haversine(coords.iloc[i].lat, coords.iloc[i].lon, coords.iloc[j].lat, coords.iloc[j].lon)
    knn_dists = np.array([np.sort(D[i])[1:4] for i in range(N)])
    THRESHOLD = float(knn_dists.max())
    adj = np.zeros((N, N))
    for i in range(N):
        order = np.argsort(D[i])
        for j in order[1:4]:
            if D[i,j] <= THRESHOLD: adj[i,j] = 1; adj[j,i] = 1
    np.fill_diagonal(adj, 0)
    np.save(GEO_CACHE, adj)

def build_adj(m):
    A = m + np.eye(N); D = A.sum(axis=1)
    D_inv = np.diag(1.0 / np.sqrt(D))
    return torch.tensor(D_inv @ A @ D_inv, dtype=torch.float32)

A_norm_t = build_adj(adj)

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 128, batch_first=True)
        self.gcn = nn.Linear(128, 64)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(128 + 64, 1)
    def forward(self, x):
        B, N_, T, F = x.shape
        h, _ = self.lstm(x.reshape(B * N_, T, F))
        h_lstm = h[:, -1, :].view(B, N_, -1)
        h_gcn = torch.relu(self.gcn(h_lstm))
        h_gcn = A_norm_t @ h_gcn
        return self.fc(self.drop(torch.cat([h_lstm, h_gcn], dim=-1))).view(B, N_)

def md(): return HybridModel()
def pr(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()

pos_neg = float((ybin_tr == 0).sum() / max((ybin_tr == 1).sum(), 1))
print(f"pos_weight: {pos_neg:.2f}")

def train_clf(pw, seed):
    model = md(); torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
    Xt = torch.tensor(Xtr_n, dtype=torch.float32); yt = torch.tensor(ybin_tr, dtype=torch.float32)
    Xv = torch.tensor(Xva_n, dtype=torch.float32); yv = torch.tensor(ybin_va, dtype=torch.float32)
    best, bs, bad = float("inf"), None, 0
    for _ in range(60):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i+64]; opt.zero_grad()
            lossf(model(Xt[idx]), yt[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = lossf(model(Xv), yv).item()
        if vl < best: best, bs, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8: break
    if bs: model.load_state_dict(bs)
    return model

def train_amt(seed):
    model = md(); torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    Xt = torch.tensor(Xtr_n, dtype=torch.float32); yt = torch.tensor(np.log1p(ytr), dtype=torch.float32)
    Xv = torch.tensor(Xva_n, dtype=torch.float32); yv = torch.tensor(np.log1p(yva), dtype=torch.float32)
    mt = torch.tensor((ytr > 0).astype(float), dtype=torch.float32)
    mv = torch.tensor((yva > 0).astype(float), dtype=torch.float32)
    best, bs, bad = float("inf"), None, 0
    for _ in range(60):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i+64]; opt.zero_grad()
            out = model(Xt[idx]); m_ = mt[idx]
            (((out - yt[idx]) ** 2 * m_).sum() / m_.sum()).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out_v = model(Xv)
            vl = ((out_v - yv) ** 2 * mv).sum() / mv.sum().item()
        if vl < best: best, bs, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 8: break
    if bs: model.load_state_dict(bs)
    return model

# ---- Run experiments (3 seeds each) ----
SEEDS = [42, 7, 123]
results = {}

# A: Baseline
print("\n=== A: Baseline (pw=1.0) ===")
preds_a = []
for s in SEEDS:
    t1 = time.time()
    c = train_clf(1.0, s); a = train_amt(s)
    pc = 1/(1+np.exp(-pr(c, Xte_n))); pa = np.clip(np.expm1(pr(a, Xte_n)), 0, None)
    p = (pc > 0.5).astype(float) * pa
    r2 = r2_score(yte, p); rmse = float(np.sqrt(mean_squared_error(yte, p)))
    preds_a.append(p); print(f"  seed={s} R2={r2:.4f} RMSE={rmse:,.0f} ({time.time()-t1:.0f}s)")
ens = np.mean(preds_a, axis=0)
r2, rmse = r2_score(yte, ens), float(np.sqrt(mean_squared_error(yte, ens)))
print(f"  ENSEMBLE: R2={r2:.4f} RMSE={rmse:,.0f}")
results['A_baseline'] = {'R2': round(r2, 6), 'RMSE': round(rmse, 0)}

# B: Weighted BCE
print(f"\n=== B: Weighted BCE (pw={pos_neg:.1f}) ===")
preds_b = []
for s in SEEDS:
    t1 = time.time()
    c = train_clf(pos_neg, s); a = train_amt(s)
    pc = 1/(1+np.exp(-pr(c, Xte_n))); pa = np.clip(np.expm1(pr(a, Xte_n)), 0, None)
    p = (pc > 0.5).astype(float) * pa
    r2 = r2_score(yte, p); rmse = float(np.sqrt(mean_squared_error(yte, p)))
    preds_b.append(p); print(f"  seed={s} R2={r2:.4f} RMSE={rmse:,.0f} ({time.time()-t1:.0f}s)")
ens = np.mean(preds_b, axis=0)
r2, rmse = r2_score(yte, ens), float(np.sqrt(mean_squared_error(yte, ens)))
print(f"  ENSEMBLE: R2={r2:.4f} RMSE={rmse:,.0f}")
results['B_weighted'] = {'R2': round(r2, 6), 'RMSE': round(rmse, 0)}

# D: Regression only
print(f"\n=== D: Regression only ===")
preds_d = []
for s in SEEDS:
    t1 = time.time()
    a = train_amt(s)
    pa = np.clip(np.expm1(pr(a, Xte_n)), 0, None)
    r2 = r2_score(yte, pa); rmse = float(np.sqrt(mean_squared_error(yte, pa)))
    preds_d.append(pa); print(f"  seed={s} R2={r2:.4f} RMSE={rmse:,.0f} ({time.time()-t1:.0f}s)")
ens = np.mean(preds_d, axis=0)
r2, rmse = r2_score(yte, ens), float(np.sqrt(mean_squared_error(yte, ens)))
print(f"  ENSEMBLE: R2={r2:.4f} RMSE={rmse:,.0f}")
results['D_regr_only'] = {'R2': round(r2, 6), 'RMSE': round(rmse, 0)}

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for name, m in sorted(results.items()):
    print(f"  {name:<30} R2={m['R2']:.4f}  RMSE={m['RMSE']:>14,.0f}")
print(f"\nTotal: {time.time()-t0:.0f}s")

with open(OUT, 'w') as f:
    json.dump(results, f, indent=2)
