# -*- coding: utf-8 -*-
"""Uji varian protokol: per-region log1p normalization + training lebih banyak."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import torch
import torch.nn as nn

torch.set_num_threads(4)
np.random.seed(42)

DATA = r"E:\Download\Jurnal\view_penjualan_detail data hingga oktober.xlsx"
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

df = pd.read_excel(DATA)
df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
df = df.drop_duplicates("d_jual_nofak")
df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                        "jual_total_fak": "jual_total", "latitude": "Latitude"})
df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
df["region"] = df["pelanggan_kota"].map(CITY_MAP)
df["month"] = df["jual_tanggal"].dt.to_period("M")
panel = df.pivot_table(index="month", columns="region", values="jual_total", aggfunc="sum")
panel = panel.reindex(columns=REGIONS).fillna(0.0).sort_index()

WINDOW = 12
X_all, y_all = [], []
for i in range(len(panel) - WINDOW):
    X_all.append(panel.iloc[i:i + WINDOW].values)
    y_all.append(panel.iloc[i + WINDOW].values)
X_all = np.array(X_all).transpose(0, 2, 1)[:, :, :, None]
y_all = np.array(y_all)
n = len(X_all)

class HybridModel(nn.Module):
    def __init__(self, hidden_lstm=128, hidden_gnn=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_lstm, batch_first=True)
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

N = len(REGIONS)
A_norm_t = torch.eye(N)

def train(model, X, y, lr=0.001, batch=32, epochs=150, seed=42):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return model

def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()

# Varian A: per-region log1p, train 42 (train+val), test 10
def run_variant(name, Xtr, ytr, Xte, yte, transform_fn, inverse_fn):
    Xtr_t = transform_fn(Xtr)
    ytr_t = transform_fn(ytr)
    Xte_t = transform_fn(Xte)
    yte_t = transform_fn(yte)
    model = HybridModel()
    train(model, Xtr_t, ytr_t)
    p_t = predict(model, Xte_t)
    p_o = inverse_fn(p_t)
    r2_t = r2_score(yte_t, p_t)
    r2_o = r2_score(yte, p_o)
    rmse_o = np.sqrt(np.mean((yte - p_o) ** 2))
    mae_o = np.mean(np.abs(yte - p_o))
    print(f"{name}: R2_trans={r2_t:.4f} R2_orig={r2_o:.4f} RMSE={rmse_o:,.0f} MAE={mae_o:,.0f}")
    return r2_o

# per-region log1p: fit pada train
def make_log1p_perregion(ytr):
    mu = np.log1p(ytr).mean(axis=0)
    sd = np.log1p(ytr).std(axis=0)
    sd[sd == 0] = 1.0
    def tf(y):
        y = np.log1p(np.clip(y, 0, None))
        if y.ndim == 4:  # (B, N, T, 1)
            return (y - mu.reshape(1, N, 1, 1)) / sd.reshape(1, N, 1, 1)
        return (y - mu) / sd
    def inv(y):
        if y.ndim == 4:
            return np.expm1(y * sd.reshape(1, N, 1, 1) + mu.reshape(1, N, 1, 1))
        return np.expm1(y * sd + mu)
    return tf, inv

# pooled log1p
def make_log1p_pooled(ytr):
    v = np.log1p(ytr).reshape(-1, 1)
    mu = v.mean(); sd = v.std()
    def tf(y):
        return (np.log1p(np.clip(y, 0, None)) - mu) / sd
    def inv(y):
        return np.expm1(y * sd + mu)
    return tf, inv

# split: train+val = 42 sampel pertama, test = 10 terakhir
tr_idx = np.arange(42)
te_idx = np.arange(42, 52)
Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
Xte, yte = X_all[te_idx], y_all[te_idx]

print("=== Varian protokol (HybridTuned, seed 42) ===")
tf, inv = make_log1p_perregion(ytr)
run_variant("A. per-region log1p, train 42", Xtr, ytr, Xte, yte, tf, inv)

tf, inv = make_log1p_pooled(ytr)
run_variant("B. pooled log1p, train 42", Xtr, ytr, Xte, yte, tf, inv)

# Varian C: per-region log1p, train 30 (protokol tesis)
tr_idx = np.arange(30)
te_idx = np.arange(42, 52)
Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
Xte, yte = X_all[te_idx], y_all[te_idx]
tf, inv = make_log1p_perregion(ytr)
run_variant("C. per-region log1p, train 30", Xtr, ytr, Xte, yte, tf, inv)

# Varian D: per-region log1p, train 42, tapi LSTM saja
class LSTMModel(nn.Module):
    def __init__(self, hidden=128, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        B, N, T, F = x.shape
        h, _ = self.lstm(x.reshape(B * N, T, F))
        return self.fc(self.drop(h[:, -1, :])).view(B, N)

tr_idx = np.arange(42)
te_idx = np.arange(42, 52)
Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
Xte, yte = X_all[te_idx], y_all[te_idx]
tf, inv = make_log1p_perregion(ytr)
Xtr_t, ytr_t = tf(Xtr), tf(ytr)
Xte_t, yte_t = tf(Xte), tf(yte)
model = LSTMModel()
train(model, Xtr_t, ytr_t)
p_o = inv(predict(model, Xte_t))
print(f"D. LSTM per-region log1p, train 42: R2_orig={r2_score(yte, p_o):.4f} RMSE={np.sqrt(np.mean((yte - p_o) ** 2)):,.0f}")

# baseline naive untuk konteks
naive = Xte[:, -1, :]
print(f"\nNaive baseline: R2={r2_score(yte, naive):.4f} RMSE={np.sqrt(np.mean((yte - naive) ** 2)):,.0f}")