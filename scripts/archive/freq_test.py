# -*- coding: utf-8 -*-
"""Uji granularitas: mingguan dan harian (HybridTuned, seed 42)."""
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


def run_freq(name, freq, window):
    print(f"\n===== {name} (window {window}) =====")
    df["period"] = df["jual_tanggal"].dt.to_period(freq)
    panel = df.pivot_table(index="period", columns="region", values="jual_total", aggfunc="sum")
    panel = panel.reindex(columns=REGIONS).fillna(0.0).sort_index()
    print(f"Panel: {len(panel)} periode ({panel.index[0]} s.d. {panel.index[-1]})")

    X_all, y_all, t_all = [], [], []
    for i in range(len(panel) - window):
        X_all.append(panel.iloc[i:i + window].values)
        y_all.append(panel.iloc[i + window].values)
        t_all.append(panel.index[i + window])
    X_all = np.array(X_all).transpose(0, 2, 1)[:, :, :, None]
    y_all = np.array(y_all)

    tr_mask = np.array([m.start_time <= pd.Timestamp("2023-12-31") for m in t_all])
    va_mask = np.array([(m.start_time > pd.Timestamp("2023-12-31")) & (m.start_time <= pd.Timestamp("2024-12-31")) for m in t_all])
    te_mask = np.array([m.start_time > pd.Timestamp("2024-12-31") for m in t_all])
    Xtr, ytr = X_all[tr_mask], y_all[tr_mask]
    Xva, yva = X_all[va_mask], y_all[va_mask]
    Xte, yte = X_all[te_mask], y_all[te_mask]
    print(f"Sampel: train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}")
    print(f"Zero actuals test: {(yte == 0).sum()} dari {yte.size}")

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
    r2_t = r2_score(yte_t, p_t)
    r2_o = r2_score(yte, p_o)
    rmse_o = np.sqrt(np.mean((yte - p_o) ** 2))
    mae_o = np.mean(np.abs(yte - p_o))
    print(f"HybridTuned: R2_trans={r2_t:.4f} R2_orig={r2_o:.4f} RMSE={rmse_o:,.0f} MAE={mae_o:,.0f}")

    naive = Xte[:, :, -1, 0]
    print(f"Naive: R2={r2_score(yte, naive):.4f} RMSE={np.sqrt(np.mean((yte - naive) ** 2)):,.0f}")

    per = [r2_score(yte[:, i], p_o[:, i]) for i in range(N)]
    best = REGIONS[int(np.argmax(per))]
    worst = REGIONS[int(np.argmin(per))]
    print(f"Per-region R2: mean={np.mean(per):.3f} best={best}({max(per):.3f}) worst={worst}({min(per):.3f})")
    return {"freq": name, "R2_trans": r2_t, "R2_orig": r2_o, "RMSE": rmse_o, "MAE": mae_o,
            "n_train": len(Xtr), "n_test": len(Xte), "zero_frac_test": float((yte == 0).mean())}


results = []
results.append(run_freq("WEEKLY", "W", 12))
results.append(run_freq("DAILY", "D", 30))

print("\n===== RINGKASAN =====")
for r in results:
    print(f"{r['freq']}: train={r['n_train']} test={r['n_test']} zero_test={r['zero_frac_test']:.2%} "
          f"R2_trans={r['R2_trans']:.4f} R2_orig={r['R2_orig']:.4f} RMSE={r['RMSE']:,.0f} MAE={r['MAE']:,.0f}")