# -*- coding: utf-8 -*-
"""Uji harian: per-region transform, baseline nol, dan zero-inflated sederhana."""
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
N = len(REGIONS)

df = pd.read_excel(DATA)
df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
df = df.drop_duplicates("d_jual_nofak")
df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                        "jual_total_fak": "jual_total", "latitude": "Latitude"})
df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
df["region"] = df["pelanggan_kota"].map(CITY_MAP)
df["period"] = df["jual_tanggal"].dt.to_period("D")
panel = df.pivot_table(index="period", columns="region", values="jual_total", aggfunc="sum")
panel = panel.reindex(columns=REGIONS).fillna(0.0).sort_index()

WINDOW = 30
X_all, y_all, t_all = [], [], []
for i in range(len(panel) - WINDOW):
    X_all.append(panel.iloc[i:i + WINDOW].values)
    y_all.append(panel.iloc[i + WINDOW].values)
    t_all.append(panel.index[i + WINDOW])
X_all = np.array(X_all).transpose(0, 2, 1)[:, :, :, None]
y_all = np.array(y_all)

tr_mask = np.array([m.start_time <= pd.Timestamp("2023-12-31") for m in t_all])
te_mask = np.array([m.start_time > pd.Timestamp("2024-12-31") for m in t_all])
Xtr, ytr = X_all[tr_mask], y_all[tr_mask]
Xte, yte = X_all[te_mask], y_all[te_mask]
print(f"train {len(Xtr)}, test {len(Xte)}, zero test {(yte == 0).mean():.2%}")

# baseline: selalu nol
zero_pred = np.zeros_like(yte)
print(f"Baseline selalu-nol: R2={r2_score(yte, zero_pred):.4f} RMSE={np.sqrt(np.mean(yte**2)):,.0f}")

# baseline: selalu nol per region yang train-nya nol, mean untuk lainnya
train_mean = ytr.mean(axis=0)
mixed = np.where(train_mean == 0, 0.0, np.tile(train_mean, (len(yte), 1)))
print(f"Baseline train-mean: R2={r2_score(yte, mixed):.4f}")


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


A_norm_t = torch.eye(N)


def train(model, X, y, lr=0.001, batch=64, epochs=150, seed=42):
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


# Varian 1: per-region PowerTransformer
def run_perregion_pt():
    pts = [PowerTransformer(method="yeo-johnson") for _ in range(N)]
    for i in range(N):
        pts[i].fit(ytr[:, i].reshape(-1, 1))
    ytr_t = np.stack([pts[i].transform(ytr[:, i].reshape(-1, 1)).ravel() for i in range(N)], axis=1)
    yte_t = np.stack([pts[i].transform(yte[:, i].reshape(-1, 1)).ravel() for i in range(N)], axis=1)
    # X: transform per region (flatten T*B)
    Xtr_flat = Xtr.reshape(-1, N)
    Xte_flat = Xte.reshape(-1, N)
    Xtr_t = np.stack([pts[i].transform(Xtr_flat[:, i].reshape(-1, 1)).ravel() for i in range(N)], axis=1).reshape(Xtr.shape)
    Xte_t = np.stack([pts[i].transform(Xte_flat[:, i].reshape(-1, 1)).ravel() for i in range(N)], axis=1).reshape(Xte.shape)
    model = HybridModel()
    train(model, Xtr_t, ytr_t)
    p_t = predict(model, Xte_t)
    p_o = np.stack([pts[i].inverse_transform(p_t[:, i].reshape(-1, 1)).ravel() for i in range(N)], axis=1)
    print(f"Per-region PT: R2_orig={r2_score(yte, p_o):.4f} RMSE={np.sqrt(np.mean((yte - p_o)**2)):,.0f} "
          f"MAE={np.mean(np.abs(yte - p_o)):,.0f}")
    return p_o


# Varian 2: pooled PT (referensi)
def run_pooled_pt():
    pt = PowerTransformer(method="yeo-johnson")
    pt.fit(ytr.reshape(-1, 1))
    def tf(y):
        return pt.transform(y.reshape(-1, 1)).reshape(y.shape)
    def inv(y):
        return pt.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)
    model = HybridModel()
    train(model, tf(Xtr), tf(ytr))
    p_o = inv(predict(model, tf(Xte)))
    print(f"Pooled PT: R2_orig={r2_score(yte, p_o):.4f} RMSE={np.sqrt(np.mean((yte - p_o)**2)):,.0f} "
          f"MAE={np.mean(np.abs(yte - p_o)):,.0f}")
    return p_o


print("\n--- Varian transformasi (daily, window 30) ---")
p1 = run_perregion_pt()
p2 = run_pooled_pt()

# per-region R2 untuk varian terbaik
per = [r2_score(yte[:, i], p1[:, i]) for i in range(N)]
print("\nPer-region R2 (per-region PT):")
for i, reg in enumerate(REGIONS):
    print(f"  {reg}: {per[i]:.3f}")