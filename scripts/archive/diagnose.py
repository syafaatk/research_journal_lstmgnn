# -*- coding: utf-8 -*-
"""Diagnosa: kualitas fit pada training vs test (HybridTuned, seed 42)."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import r2_score
import torch
import torch.nn as nn

torch.set_num_threads(4)
np.random.seed(42)

DATA = r"E:\Download\Jurnal\df.xlsx"
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
df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
df["region"] = df["pelanggan_kota"].map(CITY_MAP)
df["month"] = df["jual_tanggal"].dt.to_period("M")
panel = df.pivot_table(index="month", columns="region", values="jual_total", aggfunc="sum")
panel = panel.reindex(columns=REGIONS).fillna(0.0).sort_index()

WINDOW = 12
train_end = pd.Period("2023-12", "M")
val_end = pd.Period("2024-12", "M")

X_all, y_all, t_all = [], [], []
for i in range(len(panel) - WINDOW):
    X_all.append(panel.iloc[i:i + WINDOW].values)
    y_all.append(panel.iloc[i + WINDOW].values)
    t_all.append(panel.index[i + WINDOW])
X_all = np.array(X_all).transpose(0, 2, 1)[:, :, :, None]
y_all = np.array(y_all)

tr_mask = np.array([m <= train_end for m in t_all])
va_mask = np.array([(m > train_end) & (m <= val_end) for m in t_all])
te_mask = np.array([m > val_end for m in t_all])
Xtr, ytr = X_all[tr_mask], y_all[tr_mask]
Xva, yva = X_all[va_mask], y_all[va_mask]
Xte, yte = X_all[te_mask], y_all[te_mask]

pt = PowerTransformer(method="yeo-johnson")
pt.fit(ytr.reshape(-1, 1))

def transform(y):
    return pt.transform(y.reshape(-1, 1)).reshape(y.shape)

def inverse(y):
    return pt.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)

ytr_t, yva_t, yte_t = transform(ytr), transform(yva), transform(yte)
Xtr_t, Xva_t, Xte_t = transform(Xtr), transform(Xva), transform(Xte)

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
A_norm_t = torch.eye(N)  # identity untuk diagnosa sederhana

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

model = HybridModel()
train(model, Xtr_t, ytr_t)

# fit quality on train (transformed scale)
p_tr = predict(model, Xtr_t)
print("TRAIN (transformed): R2 =", round(r2_score(ytr_t, p_tr), 4))
print("TRAIN (original):    R2 =", round(r2_score(ytr, inverse(p_tr)), 4))
p_va = predict(model, Xva_t)
print("VAL   (transformed): R2 =", round(r2_score(yva_t, p_va), 4))
print("VAL   (original):    R2 =", round(r2_score(yva, inverse(p_va)), 4))
p_te = predict(model, Xte_t)
print("TEST  (transformed): R2 =", round(r2_score(yte_t, p_te), 4))
print("TEST  (original):    R2 =", round(r2_score(yte, inverse(p_te)), 4))

# baseline: predict last value (naive) on original scale
naive = Xte[:, :, -1, 0]
print("\nNaive (last value) TEST original: R2 =", round(r2_score(yte, naive), 4),
      "RMSE =", round(np.sqrt(np.mean((yte - naive) ** 2)), 0))
# baseline: predict mean
mean_pred = np.full_like(yte, ytr.mean())
print("Mean predictor TEST original: R2 =", round(r2_score(yte, mean_pred), 4))

# per-region train R2 to see if model fits
print("\nPer-region TRAIN R2 (transformed):")
for i, reg in enumerate(REGIONS):
    r2 = r2_score(ytr_t[:, i], p_tr[:, i])
    print(f"  {reg}: {r2:.3f}")