# -*- coding: utf-8 -*-
"""
Eksperimen final: Spatiotemporal Prediction of Medical Device Sales
Using Hybrid LSTM-GNN in South Sumatra - protokol HARIAN ZERO-INFLATED
dengan fitur eksogen COVID-19.

Data: view_penjualan_detail data hingga oktober.xlsx
- Target: jual_total (nilai faktur) per kab/kota per HARI
- 17 kab/kota Sumsel (pemetaan dari 34 kota mentah)
- Split 3 arah: training <= 2023-12-31, validasi 2024, testing 2025 (Jan-Okt)
- Window 30 hari, horizon 1
- Fitur COVID-19: indikator biner periode pandemi (Jul 2020 - Des 2022),
  ditambahkan sebagai kanal input kedua (F=2)
- Graf: Haversine, k=3, threshold 95.82 km, undirected, tanpa self-loop
- Model zero-inflated dua tahap:
    (1) Klasifikasi ada/tidak penjualan (BCE)
    (2) Regresi jumlah pada sampel non-zero (masked MSE, target log1p)
  Prediksi akhir = threshold 0.5 pada klasifikasi x jumlah
- 4 arsitektur: LSTM, GNN, Hybrid, HybridTuned; seeds 3/5
- Metrik skala asli: R2, RMSE, MAE + akurasi klasifikasi, AUC
- Diebold-Mariano, bootstrap 95% CI, ablasi graf + ablasi fitur COVID,
  per wilayah, residual
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, roc_auc_score
import torch
import torch.nn as nn

torch.set_num_threads(4)
np.random.seed(42)

DATA = r"E:\Download\Jurnal\view_penjualan_detail data hingga oktober.xlsx"
OUT = r"E:\Download\Jurnal\skrip\experiment_results_zi.json"

CITY_MAP = {
    "Palembang": "Palembang", "Lahat": "Lahat", "Baturaja": "Ogan Komering Ulu",
    "Oku (Ogan Komering Ulu)": "Ogan Komering Ulu Timur", "Musi Banyuasin": "Musi Banyuasin",
    "Muara Enim": "Muara Enim", "Tanjung Enim": "Muara Enim", "Muara Lawai": "Muara Enim",
    "Lubuklinggau": "Lubuk Linggau", "Ogan Ilir": "Ogan Ilir", "Oku Selatan": "Ogan Komering Ulu Selatan",
    "Prabumulih": "Prabumulih", "Prabumulih Timur": "Prabumulih",
    "Pali (Penukal Abab Lematang Ilir)": "Penukal Abab Lematang Ilir", "Talang Ubi": "Penukal Abab Lematang Ilir",
    "Musi Rawas": "Musi Rawas", "Oku Timur": "Ogan Komering Ulu Timur", "Martapura": "Ogan Komering Ulu Timur",
    "Empat Lawang": "Empat Lawang", "Banyuasin": "Banyuasin", "Pagaralam": "Pagar Alam",
    "Muara Rupit": "Musi Rawas Utara", "Rupit": "Musi Rawas Utara",
    "Oki (Ogan Komering Ilir)": "Ogan Komering Ilir",
}
REGIONS = [
    "Banyuasin", "Empat Lawang", "Lahat", "Lubuk Linggau", "Muara Enim",
    "Musi Banyuasin", "Musi Rawas", "Musi Rawas Utara", "Ogan Ilir",
    "Ogan Komering Ilir", "Ogan Komering Ulu", "Ogan Komering Ulu Selatan",
    "Ogan Komering Ulu Timur", "Pagar Alam", "Palembang",
    "Penukal Abab Lematang Ilir", "Prabumulih",
]
N = len(REGIONS)

# ---------------------------------------------------------------------------
# 1. Load dan agregasi harian
# ---------------------------------------------------------------------------
df = pd.read_excel(DATA)
df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
df = df.drop_duplicates("d_jual_nofak")
df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                        "jual_total_fak": "jual_total", "latitude": "Latitude"})
df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
df["region"] = df["pelanggan_kota"].map(CITY_MAP)
df["period"] = df["jual_tanggal"].dt.to_period("D")

coords = df.groupby("region").agg(lat=("Latitude", "mean"), lon=("longitude", "mean"))
panel = df.pivot_table(index="period", columns="region", values="jual_total", aggfunc="sum")
panel = panel.reindex(columns=REGIONS).fillna(0.0).sort_index()
days = panel.index.astype(str).tolist()
n_days = len(panel)
print(f"Panel: {n_days} hari ({days[0]} s.d. {days[-1]}), {N} region", flush=True)
print(f"Faktur unik setelah pemetaan: {len(df)}", flush=True)
zero_frac = (panel == 0).mean(axis=0)
desc = panel.describe().loc[["mean", "std", "min", "max"]].to_dict()

# Fitur COVID-19: indikator biner periode pandemi (kasus pertama Mar 2020 -
# pencabutan PPKM 30 Des 2022; data dimulai Jul 2020, jadi flag=1 s.d. Des 2022)
COVID_END = pd.Timestamp("2022-12-31")
covid_flag = np.array([1.0 if m.start_time <= COVID_END else 0.0 for m in panel.index])
print(f"COVID flag: {int(covid_flag.sum())} hari aktif "
      f"({(covid_flag == 1).mean():.1%} dari panel)", flush=True)

# ---------------------------------------------------------------------------
# 2. Split 3 arah (kronologis)
# ---------------------------------------------------------------------------
WINDOW = 30
X_all, y_all, t_all = [], [], []
for i in range(n_days - WINDOW):
    sales = panel.iloc[i:i + WINDOW].values            # (T, N)
    flags = np.tile(covid_flag[i:i + WINDOW][:, None], (1, N))  # (T, N)
    X_all.append(np.stack([sales, flags], axis=-1))    # (T, N, 2)
    y_all.append(panel.iloc[i + WINDOW].values)        # (N,)
    t_all.append(panel.index[i + WINDOW])
X_all = np.array(X_all).transpose(0, 2, 1, 3)          # (B, N, T, 2)
y_all = np.array(y_all)

tr_mask = np.array([m.start_time <= pd.Timestamp("2023-12-31") for m in t_all])
va_mask = np.array([(m.start_time > pd.Timestamp("2023-12-31")) & (m.start_time <= pd.Timestamp("2024-12-31")) for m in t_all])
te_mask = np.array([m.start_time > pd.Timestamp("2024-12-31") for m in t_all])
Xtr, ytr = X_all[tr_mask], y_all[tr_mask]
Xva, yva = X_all[va_mask], y_all[va_mask]
Xte, yte = X_all[te_mask], y_all[te_mask]
print(f"Sampel: train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}", flush=True)
print(f"Zero actuals test: {(yte == 0).sum()} dari {yte.size} ({(yte == 0).mean():.2%})", flush=True)

# Normalisasi kanal penjualan (log1p pooled); kanal COVID dibiarkan 0/1
mu = np.log1p(X_all[:, :, :, 0]).mean(); sd = np.log1p(X_all[:, :, :, 0]).std()
X_all_n = X_all.copy()
X_all_n[:, :, :, 0] = (np.log1p(X_all[:, :, :, 0]) - mu) / sd
Xtr_n, Xva_n, Xte_n = X_all_n[tr_mask], X_all_n[va_mask], X_all_n[te_mask]

# Dataset tanpa fitur COVID (untuk ablasi)
X_all_1 = X_all[:, :, :, :1]
mu1 = np.log1p(X_all_1[:, :, :, 0]).mean(); sd1 = np.log1p(X_all_1[:, :, :, 0]).std()
X_all_1n = X_all_1.copy()
X_all_1n[:, :, :, 0] = (np.log1p(X_all_1[:, :, :, 0]) - mu1) / sd1
Xtr_n1, Xva_n1, Xte_n1 = X_all_1n[tr_mask], X_all_1n[va_mask], X_all_1n[te_mask]

ybin_tr = (ytr > 0).astype(np.float32)
ybin_va = (yva > 0).astype(np.float32)
ybin_te = (yte > 0).astype(np.float32)

# ---------------------------------------------------------------------------
# 3. Graf: Haversine, k=3, threshold 95.82 km
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

D = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        D[i, j] = haversine(coords.iloc[i].lat, coords.iloc[i].lon,
                            coords.iloc[j].lat, coords.iloc[j].lon)

THRESHOLD = 95.82
K = 3
adj = np.zeros((N, N))
for i in range(N):
    order = np.argsort(D[i])
    for j in order[1:K + 1]:
        if D[i, j] <= THRESHOLD:
            adj[i, j] = 1
            adj[j, i] = 1
np.fill_diagonal(adj, 0)
n_edges = int(adj.sum() / 2)
degrees = adj.sum(axis=1)
print(f"Graf: {n_edges} edge, derajat min {degrees.min():.0f} max {degrees.max():.0f} "
      f"rata {degrees.mean():.2f}, isolated {(degrees == 0).sum()}", flush=True)

knn_dists = []
for i in range(N):
    order = np.argsort(D[i])
    knn_dists.append(D[i, order[1:K + 1]])
knn_dists = np.array(knn_dists)
print(f"Jarak 3-NN: min {knn_dists.min():.2f}, median {np.median(knn_dists):.2f}, "
      f"max {knn_dists.max():.2f} km", flush=True)

def build_adj(adj_matrix):
    A_tilde = adj_matrix + np.eye(N)
    D_tilde = A_tilde.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D_tilde))
    return torch.tensor(D_inv_sqrt @ A_tilde @ D_inv_sqrt, dtype=torch.float32)

A_norm_t = build_adj(adj)

# ---------------------------------------------------------------------------
# 4. Model zero-inflated (klasifikasi + jumlah), input F=2 (dengan COVID)
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, hidden=128, dropout=0.0, in_features=2):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        B, N, T, F = x.shape
        h, _ = self.lstm(x.reshape(B * N, T, F))
        return self.fc(self.drop(h[:, -1, :])).view(B, N)


class GNNModel(nn.Module):
    def __init__(self, hidden=128, dropout=0.0, in_features=2):
        super().__init__()
        self.gcn = nn.Linear(in_features, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        B, N, T, F = x.shape
        x_last = x[:, :, -1, :]
        h = torch.relu(self.gcn(x_last))
        h = A_norm_t @ h
        return self.fc(self.drop(h)).view(B, N)


class HybridModel(nn.Module):
    def __init__(self, hidden_lstm=128, hidden_gnn=64, dropout=0.0, in_features=2):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_lstm, batch_first=True)
        self.gcn = nn.Linear(hidden_lstm, hidden_gnn)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_lstm + hidden_gnn, 1)

    def forward(self, x):
        B, N, T, F = x.shape
        h, _ = self.lstm(x.reshape(B * N, T, F))
        h_lstm = h[:, -1, :].view(B, N, -1)
        h_gcn = torch.relu(self.gcn(h_lstm))
        h_gcn = A_norm_t @ h_gcn
        fused = torch.cat([h_lstm, h_gcn], dim=-1)
        return self.fc(self.drop(fused)).view(B, N)


def make_model(name, in_features=2):
    if name == "LSTM":
        return LSTMModel(hidden=128, dropout=0.0, in_features=in_features)
    if name == "GNN":
        return GNNModel(hidden=128, dropout=0.0, in_features=in_features)
    if name == "Hybrid":
        return HybridModel(hidden_lstm=128, hidden_gnn=128, dropout=0.0, in_features=in_features)
    return HybridModel(hidden_lstm=128, hidden_gnn=64, dropout=0.2, in_features=in_features)  # HybridTuned


def train(model, X, y, binary, lr, batch, epochs, seed, Xva_, yva_, mask=None, mask_va=None, patience=8):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss() if binary else nn.MSELoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    Xv = torch.tensor(Xva_, dtype=torch.float32)
    yv = torch.tensor(yva_, dtype=torch.float32)
    n = len(Xt)
    best_loss = float("inf")
    best_state = None
    bad = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out = model(Xt[idx])
            if mask is not None:
                m = torch.tensor(mask[idx], dtype=torch.float32)
                loss = ((out - yt[idx]) ** 2 * m).sum() / m.sum()
            else:
                loss = lossf(out, yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vout = model(Xv)
            if mask_va is not None:
                mv = torch.tensor(mask_va, dtype=torch.float32)
                vloss = ((vout - yv) ** 2 * mv).sum() / mv.sum()
            else:
                vloss = lossf(vout, yv).item()
        if vloss < best_loss:
            best_loss = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


def fit_zi(name, seed, adj_matrix=None, use_covid=True):
    """Latih pasangan (klasifikasi, jumlah) dan kembalikan prediksi test."""
    global A_norm_t
    if adj_matrix is not None:
        A_norm_t = build_adj(adj_matrix)
    if use_covid:
        Xtr_, Xva_, Xte_ = Xtr_n, Xva_n, Xte_n
        in_f = 2
    else:
        Xtr_, Xva_, Xte_ = Xtr_n1, Xva_n1, Xte_n1
        in_f = 1
    clf = make_model(name, in_f)
    train(clf, Xtr_, ybin_tr, True, 0.001, 64, 60, seed, Xva_, ybin_va)
    p_clf = 1 / (1 + np.exp(-predict(clf, Xte_)))
    amt = make_model(name, in_f)
    train(amt, Xtr_, np.log1p(ytr), False, 0.001, 64, 60, seed,
          Xva_, np.log1p(yva), mask=(ytr > 0), mask_va=(yva > 0))
    p_amt = np.expm1(predict(amt, Xte_))
    p_amt = np.clip(p_amt, 0, None)
    pred = (p_clf > 0.5).astype(float) * p_amt
    return pred, p_clf, p_amt


def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.sum() > 0 else float("nan")
    return {"RMSE": float(rmse), "MAE": float(mae), "MAPE": float(mape), "R2": float(r2_score(y_true, y_pred))}


# ---------------------------------------------------------------------------
# 5. Loop utama: 4 model x seeds
# ---------------------------------------------------------------------------
CONFIGS = {"LSTM": {}, "GNN": {}, "Hybrid": {}, "HybridTuned": {}}
SEEDS = {"LSTM": [42, 7, 123], "GNN": [42, 7, 123], "Hybrid": [42, 7, 123],
         "HybridTuned": [42, 7, 123, 2024, 99]}
results = {}
preds_ref = {}
for name in CONFIGS:
    r2s, rmses, maes, accs, aucs = [], [], [], [], []
    preds_all = []
    for seed in SEEDS[name]:
        pred, p_clf, _ = fit_zi(name, seed)
        m = metrics(yte, pred)
        acc = float(((p_clf > 0.5) == ybin_te).mean())
        auc = float(roc_auc_score(ybin_te.ravel(), p_clf.ravel()))
        r2s.append(m["R2"]); rmses.append(m["RMSE"]); maes.append(m["MAE"])
        accs.append(acc); aucs.append(auc)
        preds_all.append(pred)
        if seed == 42:
            preds_ref[name] = pred
        print(f"  [{name} seed {seed}] R2 {m['R2']:.4f} RMSE {m['RMSE']:,.0f} ACC {acc:.4f}", flush=True)
    results[name] = {
        "R2": r2s, "RMSE": rmses, "MAE": maes, "ACC": accs, "AUC": aucs,
        "preds_mean": np.mean(preds_all, axis=0).tolist(),
    }
    print(f"{name}: R2 {np.mean(r2s):.4f} +- {np.std(r2s):.4f} | RMSE {np.mean(rmses):,.0f} | "
          f"MAE {np.mean(maes):,.0f} | ACC {np.mean(accs):.4f} | AUC {np.mean(aucs):.4f}", flush=True)

# Baseline
pred_zero = np.zeros_like(yte)
pred_naive = X_all[te_mask][:, :, -1, 0]
pred_mean = np.tile(ytr.mean(axis=0), (len(yte), 1))
baselines = {
    "always_zero": metrics(yte, pred_zero),
    "naive_last": metrics(yte, pred_naive),
    "train_mean": metrics(yte, pred_mean),
}
print(f"Baseline selalu-nol: R2 {baselines['always_zero']['R2']:.4f} | "
      f"naive: R2 {baselines['naive_last']['R2']:.4f} | mean: R2 {baselines['train_mean']['R2']:.4f}", flush=True)

# Checkpoint parsial: simpan hasil loop utama dulu
partial = {
    "panel": {"n_days": n_days, "days": days, "n_regions": N,
              "n_records_mapped": int(len(df)),
              "zero_frac": {k: float(v) for k, v in zero_frac.items()},
              "desc": desc},
    "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte),
              "train_end": "2023-12-31", "val_end": "2024-12-31",
              "test_start": "2025-01-01"},
    "covid": {"flag_active_days": int(covid_flag.sum()), "end": str(COVID_END.date())},
    "graph": {"n_edges": n_edges, "degrees": degrees.tolist(),
              "knn_dists": knn_dists.tolist(), "threshold": THRESHOLD,
              "coords": coords.to_dict()},
    "zero_actuals_test": int((yte == 0).sum()),
    "models": results,
    "baselines": baselines,
    "configs": {"window": WINDOW, "seeds": SEEDS, "threshold": THRESHOLD, "k": K},
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(partial, f, indent=2, default=str)
print("Checkpoint utama disimpan.", flush=True)

# ---------------------------------------------------------------------------
# 6. Diebold-Mariano (seed 42, skala asli)
# ---------------------------------------------------------------------------
def dm_test(e1, e2, h=1):
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    var = np.var(d, ddof=1) / n
    if var == 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)

e_ref = yte - preds_ref["HybridTuned"]
dm_results = {}
for name in ["LSTM", "GNN", "Hybrid"]:
    e_other = yte - preds_ref[name]
    dm, p = dm_test(e_other.flatten(), e_ref.flatten())
    dm_results[name] = {"DM": dm, "p": p}
    print(f"DM {name} vs HybridTuned: DM={dm:.3f}, p={p:.4f}", flush=True)

# Bootstrap 95% CI RMSE
def bootstrap_rmse_ci(y_true, y_pred, n_boot=1000, alpha=0.05):
    rng = np.random.default_rng(42)
    errs = (y_true - y_pred).flatten()
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(errs), len(errs))
        boots.append(np.sqrt(np.mean(errs[idx] ** 2)))
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))

boot_ci = {}
for name in CONFIGS:
    lo, hi = bootstrap_rmse_ci(yte, preds_ref[name])
    boot_ci[name] = [lo, hi]
    print(f"Bootstrap RMSE CI {name}: [{lo:,.0f}, {hi:,.0f}]", flush=True)

# ---------------------------------------------------------------------------
# 7. Ablasi graf + ablasi fitur COVID (HybridTuned, seed 42)
# ---------------------------------------------------------------------------
def run_ablation(adj_matrix, label):
    pred, _, _ = fit_zi("HybridTuned", 42, adj_matrix)
    m = metrics(yte, pred)
    print(f"Ablasi {label}: RMSE {m['RMSE']:,.0f}, MAE {m['MAE']:,.0f}, R2 {m['R2']:.4f}", flush=True)
    return m

ablation = {}
ablation["Identity adjacency"] = run_ablation(np.eye(N), "Identity")
rand_metrics = []
for perm_seed in range(5):
    rng = np.random.default_rng(perm_seed)
    perm = rng.permutation(N)
    m = run_ablation(adj[perm][:, perm], f"Random {perm_seed}")
    rand_metrics.append(m)
ablation["Random graph (mean of 5)"] = {
    "RMSE": float(np.mean([x["RMSE"] for x in rand_metrics])),
    "MAE": float(np.mean([x["MAE"] for x in rand_metrics])),
    "R2": float(np.mean([x["R2"] for x in rand_metrics])),
}
ablation["Distance graph k=3"] = run_ablation(adj, "Distance")
corr = np.corrcoef(panel.values.T)
corr_adj = (np.abs(corr) > 0.5).astype(float)
np.fill_diagonal(corr_adj, 0)
ablation["Correlation graph"] = run_ablation(corr_adj, "Correlation")
# Ablasi fitur COVID: model tanpa kanal COVID (F=1)
pred_nocovid, _, _ = fit_zi("HybridTuned", 42, use_covid=False)
ablation["Tanpa fitur COVID"] = metrics(yte, pred_nocovid)
print(f"Ablasi Tanpa fitur COVID: RMSE {ablation['Tanpa fitur COVID']['RMSE']:,.0f}, "
      f"R2 {ablation['Tanpa fitur COVID']['R2']:.4f}", flush=True)

# ---------------------------------------------------------------------------
# 8. Per wilayah (HybridTuned, seed 42) - reuse prediksi seed 42
# ---------------------------------------------------------------------------
pred = preds_ref["HybridTuned"]
per_region = {}
for r in range(N):
    m = metrics(yte[:, r], pred[:, r])
    per_region[REGIONS[r]] = m
best = min(per_region, key=lambda k: per_region[k]["RMSE"])
worst = max(per_region, key=lambda k: per_region[k]["RMSE"])
print(f"Per wilayah terbaik: {best} ({per_region[best]['RMSE']:,.0f}), "
      f"terburuk: {worst} ({per_region[worst]['RMSE']:,.0f})", flush=True)

# ---------------------------------------------------------------------------
# 9. Residual (HybridTuned, seed 42)
# ---------------------------------------------------------------------------
resid = (yte - pred).flatten()
def ljung_box(x, lags=10):
    n = len(x)
    x = x - x.mean()
    acf = np.array([np.corrcoef(x[:-l], x[l:])[0, 1] if l < n else 0 for l in range(1, lags + 1)])
    acf = np.nan_to_num(acf)
    q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, lags + 1)))
    p = 1 - stats.chi2.cdf(q, lags)
    return float(q), float(p)

q_lb, p_lb = ljung_box(resid)
print(f"Residual: mean {resid.mean():,.0f}, Ljung-Box Q={q_lb:.3f}, p={p_lb:.4f}", flush=True)

# ---------------------------------------------------------------------------
# 10. Simpan final
# ---------------------------------------------------------------------------
partial.update({
    "dm": dm_results,
    "bootstrap_ci": boot_ci,
    "ablation": ablation,
    "per_region": per_region,
    "residual": {"mean": float(resid.mean()), "ljung_box_q": q_lb, "ljung_box_p": p_lb},
})
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(partial, f, indent=2, default=str)
print("Selesai. Hasil disimpan ke", OUT, flush=True)