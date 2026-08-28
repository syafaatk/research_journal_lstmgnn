"""Single-seed, all experiments in one pass. Fixed mask bug."""
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
print(f"Data: {time.time()-t0:.1f}s  pos_weight={pos_neg:.2f}  Xtr={Xtr_n.shape}  Xva={Xva_n.shape}")

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

EPOCHS, PAT, SEED = 40, 10, 42
torch.manual_seed(SEED)

# Pre-convert tensors
Xt_t = torch.tensor(Xtr_n, dtype=torch.float32)
Xv_t = torch.tensor(Xva_n, dtype=torch.float32)
yt_clf_t = torch.tensor(ybin_tr, dtype=torch.float32)
yv_clf_t = torch.tensor(ybin_va, dtype=torch.float32)
yt_amt_t = torch.tensor(np.log1p(ytr), dtype=torch.float32)
yv_amt_t = torch.tensor(np.log1p(yva), dtype=torch.float32)
mt_amt = torch.tensor((ytr > 0).astype(float), dtype=torch.float32)
mv_amt = torch.tensor((yva > 0).astype(float), dtype=torch.float32)

def train_clf(pw):
    torch.manual_seed(SEED); model = md()
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

def train_amt(mask_all=False):
    torch.manual_seed(SEED); model = md()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    mt_ = torch.ones_like(yt_amt_t) if mask_all else mt_amt
    mv_ = torch.ones_like(yv_amt_t) if mask_all else mv_amt
    best, bs, bad = float("inf"), None, 0
    for _ in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xt_t))
        for i in range(0, len(Xt_t), 64):
            idx = perm[i:i+64]; opt.zero_grad()
            out = model(Xt_t[idx])
            m_ = mt_[idx]
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

# Train models
print("Training classifiers...", flush=True)
t1 = time.time()
c_base = train_clf(1.0); print(f"  base clf: {time.time()-t1:.0f}s", flush=True)
t1 = time.time()
c_wt = train_clf(pos_neg); print(f"  weighted clf: {time.time()-t1:.0f}s", flush=True)

print("Training regressors...", flush=True)
t1 = time.time()
a_masked = train_amt(mask_all=False); print(f"  masked amt: {time.time()-t1:.0f}s", flush=True)
t1 = time.time()
a_full = train_amt(mask_all=True); print(f"  full amt: {time.time()-t1:.0f}s", flush=True)

# Predictions
print("Predictions...", flush=True)
pc_base_te = 1/(1+np.exp(-pr(c_base, Xte_n)))
pc_wt_te = 1/(1+np.exp(-pr(c_wt, Xte_n)))
pc_base_va = 1/(1+np.exp(-pr(c_base, Xva_n)))
pc_wt_va = 1/(1+np.exp(-pr(c_wt, Xva_n)))
pa_masked_te = np.clip(np.expm1(pr(a_masked, Xte_n)), 0, None)
pa_full_te = np.clip(np.expm1(pr(a_full, Xte_n)), 0, None)
pa_masked_va = np.clip(np.expm1(pr(a_masked, Xva_n)), 0, None)
pa_full_va = np.clip(np.expm1(pr(a_full, Xva_n)), 0, None)
print(f"  done: {time.time()-t0:.0f}s", flush=True)

# Threshold sweep (on validation only)
def sweep(pc_va, pa_va, y_val, label):
    best_thr, best_r2 = 0.5, -999
    for thr in np.arange(0.05, 0.96, 0.05):
        p = (pc_va > thr).astype(float) * pa_va
        try: r2 = r2_score(y_val, p)
        except: r2 = -999
        if r2 > best_r2: best_r2, best_thr = r2, thr
    print(f"  {label}: best_thr={best_thr:.2f} val_R2={best_r2:.4f}")
    return best_thr

results = {}
print("\nThreshold sweeps:", flush=True)
thr_b2 = sweep(pc_wt_va, pa_masked_va, yva, "B2 (wt+masked)")
thr_c2 = sweep(pc_wt_va, pa_full_va, yva, "C2 (wt+full)")
thr_c3 = sweep(pc_base_va, pa_full_va, yva, "C3 (base+full)")

# All experiments
def eval_exp(pc, pa, thr, label):
    p = (pc > thr).astype(float) * pa
    r2 = r2_score(yte, p); rmse = float(np.sqrt(mean_squared_error(yte, p)))
    print(f"  {label}: R2={r2:.4f} RMSE={rmse:,.0f}")
    return {'R2': round(r2, 6), 'RMSE': round(rmse, 0)}

print("\n" + "="*60)
results['A_base_thr05'] = eval_exp(pc_base_te, pa_masked_te, 0.5, "A  base pw=1 thr=0.5")
results['B1_wt_thr05'] = eval_exp(pc_wt_te, pa_masked_te, 0.5, "B1 wt pw=4.1 thr=0.5")
results['B2_wt_tuned'] = {**eval_exp(pc_wt_te, pa_masked_te, thr_b2, f"B2 wt tuned thr={thr_b2:.2f}"), 'threshold': round(thr_b2, 2)}
results['C1_wt_full_thr05'] = eval_exp(pc_wt_te, pa_full_te, 0.5, "C1 wt+full thr=0.5")
results['C2_wt_full_tuned'] = {**eval_exp(pc_wt_te, pa_full_te, thr_c2, f"C2 wt+full tuned thr={thr_c2:.2f}"), 'threshold': round(thr_c2, 2)}
results['C3_base_full_tuned'] = {**eval_exp(pc_base_te, pa_full_te, thr_c3, f"C3 base+full tuned thr={thr_c3:.2f}"), 'threshold': round(thr_c3, 2)}
r2d = r2_score(yte, pa_masked_te); rmsed = float(np.sqrt(mean_squared_error(yte, pa_masked_te)))
print(f"  D  masked only:  R2={r2d:.4f} RMSE={rmsed:,.0f}")
results['D_masked_only'] = {'R2': round(r2d, 6), 'RMSE': round(rmsed, 0)}
r2d2 = r2_score(yte, pa_full_te); rmsed2 = float(np.sqrt(mean_squared_error(yte, pa_full_te)))
print(f"  D2 full only:    R2={r2d2:.4f} RMSE={rmsed2:,.0f}")
results['D2_full_only'] = {'R2': round(r2d2, 6), 'RMSE': round(rmsed2, 0)}

print(f"\n{'='*60}")
print(f"SUMMARY (seed={SEED}, {EPOCHS}ep)")
for name in sorted(results):
    m = results[name]
    thr_s = f" thr={m['threshold']}" if 'threshold' in m else ""
    print(f"  {name:<30} R2={m['R2']:>8.4f}  RMSE={m['RMSE']:>14,.0f}{thr_s}")
print(f"Total: {time.time()-t0:.0f}s")

with open(os.path.join(ROOT, "results", "exp_zi_fix_v3.json"), 'w') as f:
    json.dump(results, f, indent=2)
