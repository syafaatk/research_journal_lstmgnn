# -*- coding: utf-8 -*-
"""Zero-inflated harian v2: model jumlah dilatih hanya pada sampel non-zero."""
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

mu = np.log1p(X_all).mean(); sd = np.log1p(X_all).std()
X_all_n = (np.log1p(X_all) - mu) / sd
Xtr_n, Xte_n = X_all_n[tr_mask], X_all_n[te_mask]

ybin_tr = (ytr > 0).astype(np.float32)
ybin_te = (yte > 0).astype(np.float32)


class LSTMBin(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        B, N, T, F = x.shape
        h, _ = self.lstm(x.reshape(B * N, T, F))
        return self.fc(h[:, -1, :]).view(B, N)


def train_bin(model, X, y, lr=0.001, batch=64, epochs=100, seed=42):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
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


def train_reg(model, X, y, lr=0.001, batch=64, epochs=100, seed=42):
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


clf = LSTMBin()
train_bin(clf, Xtr_n, ybin_tr)
p_clf = 1 / (1 + np.exp(-predict(clf, Xte_n)))
print(f"Klasifikasi akurasi: {((p_clf > 0.5) == ybin_te).mean():.4f}")

# ---- Model jumlah: hanya sampel non-zero ----
mask_tr = ytr > 0
# flatten (B, N) -> pasangan (b, n) non-zero
b_idx, n_idx = np.where(mask_tr)
Xamt = Xtr_n[b_idx, n_idx][:, None, :, :]  # (n_nz, 1, T, 1)
yamt = np.log1p(ytr[b_idx, n_idx])  # (n_nz,)
print(f"Sampel non-zero training: {len(Xamt)}")

amt = LSTMBin()  # reuse arsitektur (input 1 feature)
train_reg(amt, Xamt, yamt)

# prediksi jumlah untuk SEMUA test (b, n)
B_te, N_te, T_te, F_te = Xte_n.shape
Xte_flat = Xte_n.reshape(B_te * N_te, T_te, F_te)
p_amt_flat = np.expm1(predict(amt, Xte_flat.reshape(B_te * N_te, 1, T_te, F_te))).ravel()
p_amt = p_amt_flat.reshape(B_te, N_te)
p_amt = np.clip(p_amt, 0, None)

# gabungan
p_zi = p_clf * p_amt
p_hard = (p_clf > 0.5).astype(float) * p_amt
print(f"Zero-inflated soft: R2={r2_score(yte, p_zi):.4f} RMSE={np.sqrt(np.mean((yte - p_zi)**2)):,.0f} MAE={np.mean(np.abs(yte - p_zi)):,.0f}")
print(f"Zero-inflated hard: R2={r2_score(yte, p_hard):.4f} RMSE={np.sqrt(np.mean((yte - p_hard)**2)):,.0f} MAE={np.mean(np.abs(yte - p_hard)):,.0f}")
print(f"Selalu nol:         R2={r2_score(yte, np.zeros_like(yte)):.4f}")

# per-region
per = [r2_score(yte[:, i], p_hard[:, i]) for i in range(N)]
print("\nPer-region R2 (zero-inflated hard):")
for i, reg in enumerate(REGIONS):
    print(f"  {reg}: {per[i]:.3f}")