# -*- coding: utf-8 -*-
"""Zero-inflated mingguan: klasifikasi + model jumlah (non-zero only)."""
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


def run_zi(name, freq, window):
    print(f"\n===== {name} (window {window}) =====")
    df["period"] = df["jual_tanggal"].dt.to_period(freq)
    panel = df.pivot_table(index="period", columns="region", values="jual_total", aggfunc="sum")
    panel = panel.reindex(columns=REGIONS).fillna(0.0).sort_index()

    X_all, y_all, t_all = [], [], []
    for i in range(len(panel) - window):
        X_all.append(panel.iloc[i:i + window].values)
        y_all.append(panel.iloc[i + window].values)
        t_all.append(panel.index[i + window])
    X_all = np.array(X_all).transpose(0, 2, 1)[:, :, :, None]
    y_all = np.array(y_all)

    tr_mask = np.array([m.start_time <= pd.Timestamp("2023-12-31") for m in t_all])
    te_mask = np.array([m.start_time > pd.Timestamp("2024-12-31") for m in t_all])
    Xtr, ytr = X_all[tr_mask], y_all[tr_mask]
    Xte, yte = X_all[te_mask], y_all[te_mask]
    print(f"train {len(Xtr)}, test {len(Xte)}, zero test {(yte == 0).mean():.2%}")

    mu = np.log1p(X_all).mean(); sd = np.log1p(X_all).std()
    X_all_n = (np.log1p(X_all) - mu) / sd
    Xtr_n, Xte_n = X_all_n[tr_mask], X_all_n[te_mask]

    ybin_tr = (ytr > 0).astype(np.float32)
    ybin_te = (yte > 0).astype(np.float32)

    clf = LSTMBin()
    train_bin(clf, Xtr_n, ybin_tr)
    p_clf = 1 / (1 + np.exp(-predict(clf, Xte_n)))
    acc = ((p_clf > 0.5) == ybin_te).mean()
    print(f"Klasifikasi akurasi: {acc:.4f}")

    mask_tr = ytr > 0
    b_idx, n_idx = np.where(mask_tr)
    Xamt = Xtr_n[b_idx, n_idx][:, None, :, :]
    yamt = np.log1p(ytr[b_idx, n_idx])
    amt = LSTMBin()
    train_reg(amt, Xamt, yamt)

    B_te, N_te, T_te, F_te = Xte_n.shape
    Xte_flat = Xte_n.reshape(B_te * N_te, T_te, F_te)
    p_amt_flat = np.expm1(predict(amt, Xte_flat.reshape(B_te * N_te, 1, T_te, F_te))).ravel()
    p_amt = np.clip(p_amt_flat.reshape(B_te, N_te), 0, None)

    p_hard = (p_clf > 0.5).astype(float) * p_amt
    r2_zi = r2_score(yte, p_hard)
    rmse_zi = np.sqrt(np.mean((yte - p_hard) ** 2))
    mae_zi = np.mean(np.abs(yte - p_hard))
    r2_zero = r2_score(yte, np.zeros_like(yte))
    print(f"Zero-inflated hard: R2={r2_zi:.4f} RMSE={rmse_zi:,.0f} MAE={mae_zi:,.0f}")
    print(f"Selalu nol: R2={r2_zero:.4f}")

    per = [r2_score(yte[:, i], p_hard[:, i]) for i in range(N)]
    print(f"Per-region R2: mean={np.mean(per):.3f} best={REGIONS[int(np.argmax(per))]}({max(per):.3f}) "
          f"worst={REGIONS[int(np.argmin(per))]}({min(per):.3f})")
    return {"freq": name, "acc": acc, "R2_zi": r2_zi, "RMSE": rmse_zi, "MAE": mae_zi,
            "R2_always_zero": r2_zero, "n_train": len(Xtr), "n_test": len(Xte)}


results = []
results.append(run_zi("WEEKLY", "W", 12))
results.append(run_zi("DAILY", "D", 30))

print("\n===== RINGKASAN ZERO-INFLATED =====")
for r in results:
    print(f"{r['freq']}: train={r['n_train']} test={r['n_test']} acc={r['acc']:.4f} "
          f"R2_zi={r['R2_zi']:.4f} R2_zero={r['R2_always_zero']:.4f} RMSE={r['RMSE']:,.0f} MAE={r['MAE']:,.0f}")