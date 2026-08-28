"""Ultra-minimal: 1 seed, 30 epochs, cached data."""
import numpy as np, torch, torch.nn as nn, json, os, time
from sklearn.metrics import r2_score, mean_squared_error

t0 = time.time()
ROOT = r"E:\Download\Jurnal"
CACHE = os.path.join(ROOT, "results", "_cache_panel.npz")
GEO_CACHE = os.path.join(ROOT, "results", "_cache_adj.npy")

d = np.load(CACHE, allow_pickle=True)
X_all_n, y_all = d['X_all_n'], d['y_all']
tr_mask, va_mask, te_mask = d['tr_mask'], d['va_mask'], d['te_mask']
ybin_tr, ybin_va, ybin_te = d['ybin_tr'], d['ybin_va'], d['ybin_te']
yte, ytr, yva = d['yte'], d['ytr'], d['yva']
Xtr_n = X_all_n[tr_mask]; Xva_n = X_all_n[va_mask]; Xte_n = X_all_n[te_mask]
N = 16
adj = np.load(GEO_CACHE)
print(f"Data loaded: {time.time()-t0:.1f}s, Xtr={Xtr_n.shape}")

def build_adj(m):
    A = m + np.eye(N); D_ = A.sum(axis=1)
    D_inv = np.diag(1.0 / np.sqrt(D_))
    return torch.tensor(D_inv @ A @ D_inv, dtype=torch.float32)
A_norm_t = build_adj(adj)

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 128, batch_first=True)
        self.gcn = nn.Linear(128, 64)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(192, 1)
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

EPOCHS = 30; PAT = 8; SEED = 42

def train_clf(pw):
    model = md(); torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
    Xt = torch.tensor(Xtr_n, dtype=torch.float32); yt = torch.tensor(ybin_tr, dtype=torch.float32)
    Xv = torch.tensor(Xva_n, dtype=torch.float32); yv = torch.tensor(ybin_va, dtype=torch.float32)
    best, bs, bad = float("inf"), None, 0
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i+64]; opt.zero_grad()
            lossf(model(Xt[idx]), yt[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = lossf(model(Xv), yv).item()
        if vl < best: best, bs, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else: bad += 1
        if bad >= PAT: break
    if bs: model.load_state_dict(bs)
    return model

def train_amt():
    model = md(); torch.manual_seed(SEED)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    Xt = torch.tensor(Xtr_n, dtype=torch.float32); yt = torch.tensor(np.log1p(ytr), dtype=torch.float32)
    Xv = torch.tensor(Xva_n, dtype=torch.float32); yv = torch.tensor(np.log1p(yva), dtype=torch.float32)
    mt = torch.tensor((ytr > 0).astype(float), dtype=torch.float32)
    mv = torch.tensor((yva > 0).astype(float), dtype=torch.float32)
    best, bs, bad = float("inf"), None, 0
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 64):
            idx = perm[i:i+64]; opt.zero_grad()
            out = model(Xt[idx]); m_ = mt[idx]
            (((out - yt[idx]) ** 2 * m_).sum() / m_.sum()).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out_v = model(Xv)
            vl = ((out_v - yv) ** 2 * mv).sum() / mv.sum().item()
        if vl < best: best, bs, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else: bad += 1
        if bad >= PAT: break
    if bs: model.load_state_dict(bs)
    return model

# A: Baseline
print(f"\n=== A: Baseline (pw=1.0) ===", flush=True)
t1 = time.time()
c = train_clf(1.0); a = train_amt()
pc = 1/(1+np.exp(-pr(c, Xte_n))); pa = np.clip(np.expm1(pr(a, Xte_n)), 0, None)
p_a = (pc > 0.5).astype(float) * pa
r2a = r2_score(yte, p_a); rmsea = float(np.sqrt(mean_squared_error(yte, p_a)))
print(f"  R2={r2a:.4f} RMSE={rmsea:,.0f} ({time.time()-t1:.0f}s)", flush=True)

# B: Weighted BCE
print(f"\n=== B: Weighted BCE (pw={pos_neg:.1f}) ===", flush=True)
t1 = time.time()
c = train_clf(pos_neg); a = train_amt()
pc = 1/(1+np.exp(-pr(c, Xte_n))); pa = np.clip(np.expm1(pr(a, Xte_n)), 0, None)
p_b = (pc > 0.5).astype(float) * pa
r2b = r2_score(yte, p_b); rmseb = float(np.sqrt(mean_squared_error(yte, p_b)))
print(f"  R2={r2b:.4f} RMSE={rmseb:,.0f} ({time.time()-t1:.0f}s)", flush=True)

# pclf analysis for B
print("  pclf per-region (weighted BCE):")
for j, reg in enumerate(["Banyuasin","Empat Lawang","Lahat","Lubuk Linggau","Muara Enim","Musi Banyuasin","Musi Rawas","Ogan Ilir","Ogan Komering Ilir","Ogan Komering Ulu","Ogan Komering Ulu Selatan","Ogan Komering Ulu Timur","Pagar Alam","Palembang","Penukal Abab Lematang Ilir","Prabumulih"]):
    nz_true = int(ybin_te[:, j].sum())
    nz_pred = (pc[:, j] > 0.5).sum()
    print(f"    {reg:<35} nz_true={nz_true:>3} nz_pred>0.5={nz_pred:>3} mean_pclf={pc[:,j].mean():.4f} max={pc[:,j].max():.4f}")

# D: Regression only
print(f"\n=== D: Regression only ===", flush=True)
t1 = time.time()
a = train_amt()
pa = np.clip(np.expm1(pr(a, Xte_n)), 0, None)
r2d = r2_score(yte, pa); rmsed = float(np.sqrt(mean_squared_error(yte, pa)))
print(f"  R2={r2d:.4f} RMSE={rmsed:,.0f} ({time.time()-t1:.0f}s)", flush=True)

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"  A (baseline pw=1.0):     R2={r2a:.4f}  RMSE={rmsea:>14,.0f}")
print(f"  B (weighted pw={pos_neg:.1f}): R2={r2b:.4f}  RMSE={rmseb:>14,.0f}")
print(f"  D (regression only):     R2={r2d:.4f}  RMSE={rmsed:>14,.0f}")
print(f"Total: {time.time()-t0:.0f}s")

results = {
    'A_baseline': {'R2': round(r2a, 6), 'RMSE': round(rmsea, 0)},
    'B_weighted': {'R2': round(r2b, 6), 'RMSE': round(rmseb, 0)},
    'D_regr_only': {'R2': round(r2d, 6), 'RMSE': round(rmsed, 0)},
    'pos_weight': pos_neg,
}
with open(os.path.join(ROOT, "results", "exp_zi_fix_quick.json"), 'w') as f:
    json.dump(results, f, indent=2)
