# -*- coding: utf-8 -*-
"""
Lanjutan experiment_zi_final.py: menyelesaikan ablasi (Random 4, Distance,
Correlation, COVID), per-wilayah, dan residual; menggabungkan dengan
checkpoint parsial. DM dan bootstrap diambil dari log run sebelumnya.
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score
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

# --- Pipeline data (sama dengan experiment_zi_final.py) ---
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
n_days = len(panel)

COVID_END = pd.Timestamp("2022-12-31")
covid_flag = np.array([1.0 if m.start_time <= COVID_END else 0.0 for m in panel.index])

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

mu = np.log1p(X_all[:, :, :, 0]).mean(); sd = np.log1p(X_all[:, :, :, 0]).std()
X_all_n = X_all.copy()
X_all_n[:, :, :, 0] = (np.log1p(X_all[:, :, :, 0]) - mu) / sd
Xtr_n, Xva_n, Xte_n = X_all_n[tr_mask], X_all_n[va_mask], X_all_n[te_mask]

X_all_1 = X_all[:, :, :, :1]
mu1 = np.log1p(X_all_1[:, :, :, 0]).mean(); sd1 = np.log1p(X_all_1[:, :, :, 0]).std()
X_all_1n = X_all_1.copy()
X_all_1n[:, :, :, 0] = (np.log1p(X_all_1[:, :, :, 0]) - mu1) / sd1
Xtr_n1, Xva_n1, Xte_n1 = X_all_1n[tr_mask], X_all_1n[va_mask], X_all_1n[te_mask]

ybin_tr = (ytr > 0).astype(np.float32)
ybin_va = (yva > 0).astype(np.float32)
ybin_te = (yte > 0).astype(np.float32)

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

def build_adj(adj_matrix):
    A_tilde = adj_matrix + np.eye(N)
    D_tilde = A_tilde.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D_tilde))
    return torch.tensor(D_inv_sqrt @ A_tilde @ D_inv_sqrt, dtype=torch.float32)

A_norm_t = build_adj(adj)

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
    return HybridModel(hidden_lstm=128, hidden_gnn=64, dropout=0.2, in_features=in_features)


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


# --- Load checkpoint parsial ---
with open(OUT, "r", encoding="utf-8") as f:
    data = json.load(f)

# --- DM dan bootstrap dari log run sebelumnya (tidak butuh retrain) ---
data["dm"] = {
    "LSTM": {"DM": 7.849, "p": 0.0000},
    "GNN": {"DM": 9.779, "p": 0.0000},
    "Hybrid": {"DM": -7.575, "p": 0.0000},
}
data["bootstrap_ci"] = {
    "LSTM": [6912061, 9250907],
    "GNN": [7785301, 10137078],
    "Hybrid": [6639512, 8960272],
    "HybridTuned": [6769161, 9104539],
}

# --- Ablasi: 5 sudah selesai (nilai dari log), retrain 4 sisanya ---
ablation = {
    "Identity adjacency": {"RMSE": 7836588, "MAE": 1787482, "R2": 0.1055, "MAPE": float("nan")},
    "Random graph (mean of 5)": None,  # diisi setelah Random 4
}
rand_done = [
    {"RMSE": 7985281, "MAE": 1826755, "R2": 0.1000},
    {"RMSE": 7936956, "MAE": 1812827, "R2": 0.1018},
    {"RMSE": 7518001, "MAE": 1719956, "R2": 0.1168},
    {"RMSE": 7830697, "MAE": 1785617, "R2": 0.1057},
]

print("Retrain Random 4...", flush=True)
rng = np.random.default_rng(4)
perm4 = rng.permutation(N)
pred, _, _ = fit_zi("HybridTuned", 42, adj[perm4][:, perm4])
m4 = metrics(yte, pred)
rand_done.append(m4)
print(f"Random 4: R2 {m4['R2']:.4f}", flush=True)
ablation["Random graph (mean of 5)"] = {
    "RMSE": float(np.mean([x["RMSE"] for x in rand_done])),
    "MAE": float(np.mean([x["MAE"] for x in rand_done])),
    "R2": float(np.mean([x["R2"] for x in rand_done])),
    "MAPE": float("nan"),
}

print("Retrain Distance graph...", flush=True)
pred_dist, _, _ = fit_zi("HybridTuned", 42, adj)
m_dist = metrics(yte, pred_dist)
ablation["Distance graph k=3"] = m_dist
print(f"Distance: R2 {m_dist['R2']:.4f}", flush=True)

print("Retrain Correlation graph...", flush=True)
corr = np.corrcoef(panel.values.T)
corr_adj = (np.abs(corr) > 0.5).astype(float)
np.fill_diagonal(corr_adj, 0)
pred, _, _ = fit_zi("HybridTuned", 42, corr_adj)
m_corr = metrics(yte, pred)
ablation["Correlation graph"] = m_corr
print(f"Correlation: R2 {m_corr['R2']:.4f}", flush=True)

print("Retrain tanpa fitur COVID...", flush=True)
pred, _, _ = fit_zi("HybridTuned", 42, use_covid=False)
m_nocovid = metrics(yte, pred)
ablation["Tanpa fitur COVID"] = m_nocovid
print(f"Tanpa COVID: R2 {m_nocovid['R2']:.4f}", flush=True)
data["ablation"] = ablation

# --- Per wilayah + residual: pakai prediksi model utama (Distance graph, seed 42) ---
pred_main = pred_dist
per_region = {}
for r in range(N):
    per_region[REGIONS[r]] = metrics(yte[:, r], pred_main[:, r])
best = min(per_region, key=lambda k: per_region[k]["RMSE"])
worst = max(per_region, key=lambda k: per_region[k]["RMSE"])
print(f"Per wilayah terbaik: {best} ({per_region[best]['RMSE']:,.0f}), "
      f"terburuk: {worst} ({per_region[worst]['RMSE']:,.0f})", flush=True)
data["per_region"] = per_region

resid = (yte - pred_main).flatten()
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
data["residual"] = {"mean": float(resid.mean()), "ljung_box_q": q_lb, "ljung_box_p": p_lb}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, default=str)
print("Selesai. Hasil final disimpan ke", OUT, flush=True)