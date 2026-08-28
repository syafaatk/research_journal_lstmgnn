# -*- coding: utf-8 -*-
"""Uji: split acak (random) vs kronologis - apakah R2 melonjak karena leakage?"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
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
print("total samples:", n)

class HybridModel(nn.Module):
    def __init__(self, hidden_lstm=128, hidden_gnn=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_lstm, batch_first=True)
        self.gcn = nn.Linear(hidden_lstm, hidden_gnn)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_lstm + hidden_gnn, 1)

    def forward(self, x):
        B, N, T, F = x.shape
        h, _ = self.lstm(x.view(B * N, T, F))
        h_lstm = h[:, -1, :].view(B, N, -1)
        h_gcn = torch.relu(self.gcn(h_lstm))
        h_gcn = A_norm_t @ h_gcn
        fused = torch.cat([h_lstm, h_gcn], dim=-1)
        return self.fc(self.drop(fused)).view(B, N)

N = len(REGIONS)
A_norm_t = torch.eye(N)

def train(model, X, y, lr=0.001, batch=32, epochs=100, seed=42):
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

# RANDOM split: 30 train, 12 val, 10 test (acak dari 52 sampel)
rng = np.random.default_rng(42)
idx = rng.permutation(n)
tr_idx = idx[:30]
te_idx = idx[-10:]
Xtr, ytr = X_all[tr_idx], y_all[tr_idx]
Xte, yte = X_all[te_idx], y_all[te_idx]

pt = PowerTransformer(method="yeo-johnson")
pt.fit(ytr.reshape(-1, 1))
def transform(y):
    return pt.transform(y.reshape(-1, 1)).reshape(y.shape)
def inverse(y):
    return pt.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)
Xtr_t, ytr_t = transform(Xtr), transform(ytr)
Xte_t, yte_t = transform(Xte), transform(yte)

model = HybridModel()
train(model, Xtr_t, ytr_t)
p_t = predict(model, Xte_t)
p_o = inverse(p_t)
print("\nRANDOM SPLIT (leakage):")
print("  transformed R2:", round(r2_score(yte_t, p_t), 4))
print("  original R2:   ", round(r2_score(yte, p_o), 4))
print("  original RMSE: ", round(np.sqrt(np.mean((yte - p_o) ** 2)), 0))

# CHRONOLOGICAL split untuk perbandingan
tr_mask = np.arange(n) < 30
te_mask = np.arange(n) >= 42
Xtr, ytr = X_all[tr_mask], y_all[tr_mask]
Xte, yte = X_all[te_mask], y_all[te_mask]
pt = PowerTransformer(method="yeo-johnson")
pt.fit(ytr.reshape(-1, 1))
Xtr_t, ytr_t = transform(Xtr), transform(ytr)
Xte_t, yte_t = transform(Xte), transform(yte)
model = HybridModel()
train(model, Xtr_t, ytr_t)
p_t = predict(model, Xte_t)
p_o = inverse(p_t)
print("\nCHRONOLOGICAL SPLIT:")
print("  transformed R2:", round(r2_score(yte_t, p_t), 4))
print("  original R2:   ", round(r2_score(yte, p_o), 4))
print("  original RMSE: ", round(np.sqrt(np.mean((yte - p_o) ** 2)), 0))