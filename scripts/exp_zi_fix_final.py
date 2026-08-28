"""3-seed ensemble for best configs: A (baseline) vs C1 (wt+fullreg)."""
import numpy as np, torch, torch.nn as nn, json, os, time
from sklearn.metrics import r2_score, mean_squared_error

t0 = time.time()
ROOT = r"E:\Download\Jurnal"
d = np.load(os.path.join(ROOT, "results", "_cache_panel.npz"), allow_pickle=True)
X_all_n = d['X_all_n']; y_all = d['y_all']
tr_mask, va_mask, te_mask = d['tr_mask'], d['va_mask'], d['te_mask']
ybin_tr, ybin_va, ybin_te = d['ybin_tr'], d['ybin_va'], d['ybin_te']
yte, ytr, yva = d['yte'], d['ytr'], d['yva']
Xtr_n, Xva_n, Xte_n = X_all_n[tr_mask], X_all_n[va_mask], X_all_n[te_mask]
N = 16
adj = np.load(os.path.join(ROOT, "results", "_cache_adj.npy"))
pos_neg = float((ybin_tr == 0).sum() / max((ybin_tr == 1).sum(), 1))
print(f"Data: {time.time()-t0:.1f}s  pw={pos_neg:.2f}")

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

Xt_t = torch.tensor(Xtr_n, dtype=torch.float32)
Xv_t = torch.tensor(Xva_n, dtype=torch.float32)
yt_clf_t = torch.tensor(ybin_tr, dtype=torch.float32)
yv_clf_t = torch.tensor(ybin_va, dtype=torch.float32)
yt_amt_t = torch.tensor(np.log1p(ytr), dtype=torch.float32)
yv_amt_t = torch.tensor(np.log1p(yva), dtype=torch.float32)
mt_amt = torch.tensor((ytr > 0).astype(float), dtype=torch.float32)
mv_amt = torch.tensor((yva > 0).astype(float), dtype=torch.float32)

EPOCHS, PAT = 40, 10

def train_clf(pw, seed):
    torch.manual_seed(seed); model = md()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
    best, bs, bad = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xt_t))
        for i in range(0, len(Xt_t), 64):
            idx = perm[i:i+64]; opt.zero_grad()
            lossf(model(Xt_t[idx]), yt_clf_t[idx]).backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl = lossf(model(Xv_t), yv_clf_t).item()
        if vl < best: best, bs, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else: bad += 1
        if bad >= PAT: break
    if bs: model.load_state_dict(bs)
    return model

def train_amt(seed, mask_all=False):
    torch.manual_seed(seed); model = md()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    mt_ = torch.ones_like(yt_amt_t) if mask_all else mt_amt
    mv_ = torch.ones_like(yv_amt_t) if mask_all else mv_amt
    best, bs, bad = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xt_t))
        for i in range(0, len(Xt_t), 64):
            idx = perm[i:i+64]; opt.zero_grad()
            out = model(Xt_t[idx]); m_ = mt_[idx]
            (((out - yt_amt_t[idx]) ** 2 * m_).sum() / m_.sum().clamp(min=1)).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out_v = model(Xv_t)
            vl = ((out_v - yv_amt_t) ** 2 * mv_).sum() / mv_.sum().clamp(min=1).item()
        if vl < best: best, bs, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else: bad += 1
        if bad >= PAT: break
    if bs: model.load_state_dict(bs)
    return model

SEEDS = [42, 7, 123]

# === A: Baseline (base clf + masked amt, thr=0.5) ===
print("\n=== A: Baseline (3 seeds) ===", flush=True)
preds_a = []
for s in SEEDS:
    t1 = time.time()
    c = train_clf(1.0, s); a = train_amt(s, mask_all=False)
    pc = 1/(1+np.exp(-pr(c, Xte_n))); pa = np.clip(np.expm1(pr(a, Xte_n)), 0, None)
    p = (pc > 0.5).astype(float) * pa
    r2 = r2_score(yte, p); rmse = float(np.sqrt(mean_squared_error(yte, p)))
    preds_a.append(p); print(f"  seed={s} R2={r2:.4f} RMSE={rmse:,.0f} ({time.time()-t1:.0f}s)")
ens_a = np.mean(preds_a, axis=0)
r2a, rmsea = r2_score(yte, ens_a), float(np.sqrt(mean_squared_error(yte, ens_a)))
print(f"  ENSEMBLE: R2={r2a:.4f} RMSE={rmsea:,.0f}")

# === C1: Weighted BCE + full regression, thr=0.5 ===
print(f"\n=== C1: Weighted + fullreg (3 seeds) ===", flush=True)
preds_c = []
for s in SEEDS:
    t1 = time.time()
    c = train_clf(pos_neg, s); a = train_amt(s, mask_all=True)
    pc = 1/(1+np.exp(-pr(c, Xte_n))); pa = np.clip(np.expm1(pr(a, Xte_n)), 0, None)
    p = (pc > 0.5).astype(float) * pa
    r2 = r2_score(yte, p); rmse = float(np.sqrt(mean_squared_error(yte, p)))
    preds_c.append(p); print(f"  seed={s} R2={r2:.4f} RMSE={rmse:,.0f} ({time.time()-t1:.0f}s)")
ens_c = np.mean(preds_c, axis=0)
r2c, rmsec = r2_score(yte, ens_c), float(np.sqrt(mean_squared_error(yte, ens_c)))
print(f"  ENSEMBLE: R2={r2c:.4f} RMSE={rmsec:,.0f}")

# === C3: Base clf + full regression, thr=0.5 (to check if clf weight matters) ===
print(f"\n=== C3: Base clf + fullreg (3 seeds) ===", flush=True)
preds_c3 = []
for s in SEEDS:
    t1 = time.time()
    c = train_clf(1.0, s); a = train_amt(s, mask_all=True)
    pc = 1/(1+np.exp(-pr(c, Xte_n))); pa = np.clip(np.expm1(pr(a, Xte_n)), 0, None)
    p = (pc > 0.5).astype(float) * pa
    r2 = r2_score(yte, p); rmse = float(np.sqrt(mean_squared_error(yte, p)))
    preds_c3.append(p); print(f"  seed={s} R2={r2:.4f} RMSE={rmse:,.0f} ({time.time()-t1:.0f}s)")
ens_c3 = np.mean(preds_c3, axis=0)
r2c3, rmsec3 = r2_score(yte, ens_c3), float(np.sqrt(mean_squared_error(yte, ens_c3)))
print(f"  ENSEMBLE: R2={r2c3:.4f} RMSE={rmsec3:,.0f}")

print(f"\n{'='*60}")
print(f"FINAL COMPARISON (3-seed ensemble)")
print(f"{'='*60}")
print(f"  A  base+masked:  R2={r2a:.4f}  RMSE={rmsea:,.0f}")
print(f"  C1 wt+fullreg:   R2={r2c:.4f}  RMSE={rmsec:,.0f}  (delta R2={r2c-r2a:+.4f})")
print(f"  C3 base+fullreg: R2={r2c3:.4f}  RMSE={rmsec3:,.0f}  (delta R2={r2c3-r2a:+.4f})")
print(f"Total: {time.time()-t0:.0f}s")

results = {
    'A_baseline_3seed': {'R2': round(r2a, 6), 'RMSE': round(rmsea, 0)},
    'C1_wt_fullreg_3seed': {'R2': round(r2c, 6), 'RMSE': round(rmsec, 0)},
    'C3_base_fullreg_3seed': {'R2': round(r2c3, 6), 'RMSE': round(rmsec3, 0)},
}
with open(os.path.join(ROOT, "results", "exp_zi_fix_final.json"), 'w') as f:
    json.dump(results, f, indent=2)
