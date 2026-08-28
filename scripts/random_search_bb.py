# -*- coding: utf-8 -*-
"""
Random search hyperparameter optimization, mengikuti Bergstra & Bengio (2012)
"Random search for hyper-parameter optimization", JMLR 13:281-305.

Prinsip yang diterapkan:
1. Random search (bukan grid): N trial mengeksplorasi N nilai berbeda per
   dimensi; unggul karena hyperparameter punya low effective dimensionality.
2. Distribusi sampling:
   - hidden_lstm, hidden_gnn: log-uniform [32, 256] (integer)
   - learning rate: log-uniform [1e-4, 1e-2]
   - dropout: uniform [0.0, 0.5]
   - batch: uniform diskrit {32, 64, 128}
3. Jumlah trial N=30: P(menemukan konfigurasi dalam top 10%) = 1-(0.9)^30
   = 0.958 (Bergstra & Bengio, Sec. 3.2).
4. Kriteria seleksi: validation loss (BCE klasifikasi + masked MSE jumlah,
   dijumlahkan). Test set TIDAK dipakai selama tuning; hanya sekali di akhir
   untuk konfigurasi terbaik.
5. Early stopping (patience 8) untuk menghemat komputasi.

Pipeline data identik dengan experiment_zi_geocoded.py (16 region geocoded,
anti-leakage: normalisasi log1p dihitung pada window training saja).

Output: skrip/random_search_bb.json + log skrip/random_search_bb.log
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import torch
import torch.nn as nn

torch.set_num_threads(4)
np.random.seed(42)

DATA = r"E:\Download\Jurnal\data\view_penjualan_detail data hingga oktober.xlsx"
GEO = r"E:\Download\Jurnal\data\customers_geocoded_final.csv"
OUT = r"E:\Download\Jurnal\results\random_search_bb.json"

CITY_MAP = {
    "Palembang": "Palembang", "Lahat": "Lahat", "Baturaja": "Ogan Komering Ulu",
    "Oku (Ogan Komering Ulu)": "Ogan Komering Ulu Timur", "Musi Banyuasin": "Musi Banyuasin",
    "Muara Enim": "Muara Enim", "Tanjung Enim": "Muara Enim", "Muara Lawai": "Muara Enim",
    "Lubuklinggau": "Lubuk Linggau", "Ogan Ilir": "Ogan Ilir", "Oku Selatan": "Ogan Komering Ulu Selatan",
    "Prabumulih": "Prabumulih", "Prabumulih Timur": "Prabumulih",
    "Pali (Penukal Abab Lematang Ilir)": "Penukal Abab Lematang Ilir", "Talang Ubi": "Penukal Abab Lematang Ilir",
    "Musi Rawas": "Musi Rawas", "Oku Timur": "Ogan Komering Ulu Timur", "Martapura": "Ogan Komering Ulu Timur",
    "Empat Lawang": "Empat Lawang", "Banyuasin": "Banyuasin", "Pagaralam": "Pagar Alam",
    "Muara Rupit": "Musi Rawas", "Rupit": "Musi Rawas",
    "Oki (Ogan Komering Ilir)": "Ogan Komering Ilir",
}
REGIONS = [
    "Banyuasin", "Empat Lawang", "Lahat", "Lubuk Linggau", "Muara Enim",
    "Musi Banyuasin", "Musi Rawas", "Ogan Ilir",
    "Ogan Komering Ilir", "Ogan Komering Ulu", "Ogan Komering Ulu Selatan",
    "Ogan Komering Ulu Timur", "Pagar Alam", "Palembang",
    "Penukal Abab Lematang Ilir", "Prabumulih",
]
N = len(REGIONS)

# ---------------------------------------------------------------------------
# 1. Pipeline data (identik dengan experiment_zi_geocoded.py)
# ---------------------------------------------------------------------------
df = pd.read_excel(DATA)
df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
df = df.drop_duplicates("d_jual_nofak")
df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                        "jual_total_fak": "jual_total", "latitude": "Latitude"})
df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
df["region"] = df["pelanggan_kota"].map(CITY_MAP)
df["period"] = df["jual_tanggal"].dt.to_period("D")

geo = pd.read_csv(GEO)
geo = geo[geo["status"].isin(["FOUND_GEO", "FOUND_SIRS"])].copy()
inv_per_cust = df.groupby("pelanggan_nama")["d_jual_nofak"].nunique().rename("n_inv_data")
geo = geo.merge(inv_per_cust, left_on="pelanggan_nama", right_index=True, how="left")
geo["n_inv_data"] = geo["n_inv_data"].fillna(0)
w = geo.groupby("region").apply(
    lambda g: pd.Series({
        "lat": (g["lat"] * g["n_inv_data"]).sum() / g["n_inv_data"].sum(),
        "lon": (g["lon"] * g["n_inv_data"]).sum() / g["n_inv_data"].sum(),
    }), include_groups=False)
coords = w[["lat", "lon"]].reindex(REGIONS)

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

mu = np.log1p(Xtr[:, :, :, 0]).mean(); sd = np.log1p(Xtr[:, :, :, 0]).std()
X_all_n = X_all.copy()
X_all_n[:, :, :, 0] = (np.log1p(X_all[:, :, :, 0]) - mu) / sd
Xtr_n, Xva_n, Xte_n = X_all_n[tr_mask], X_all_n[va_mask], X_all_n[te_mask]

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
K = 3
adj = np.zeros((N, N))
for i in range(N):
    order = np.argsort(D[i])
    for j in order[1:K + 1]:
        adj[i, j] = 1
        adj[j, i] = 1
np.fill_diagonal(adj, 0)

def build_adj(adj_matrix):
    A_tilde = adj_matrix + np.eye(N)
    D_tilde = A_tilde.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D_tilde))
    return torch.tensor(D_inv_sqrt @ A_tilde @ D_inv_sqrt, dtype=torch.float32)

A_norm_t = build_adj(adj)

# ---------------------------------------------------------------------------
# 2. Model (parameterized oleh konfigurasi)
# ---------------------------------------------------------------------------
class HybridModel(nn.Module):
    def __init__(self, hidden_lstm=128, hidden_gnn=64, dropout=0.0, in_features=2):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_lstm, batch_first=True)
        self.gcn = nn.Linear(hidden_lstm, hidden_gnn)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_lstm + hidden_gnn, 1)

    def forward(self, x):
        B, Nn, T, F = x.shape
        h, _ = self.lstm(x.reshape(B * Nn, T, F))
        h_lstm = h[:, -1, :].view(B, Nn, -1)
        h_gcn = torch.relu(self.gcn(h_lstm))
        h_gcn = A_norm_t @ h_gcn
        fused = torch.cat([h_lstm, h_gcn], dim=-1)
        return self.fc(self.drop(fused)).view(B, Nn)


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
    return model, best_loss


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


def fit_zi(cfg, seed=42):
    """Latih (klasifikasi, jumlah) dengan konfigurasi cfg; kembalikan
    (pred_test, val_loss_total). Test TIDAK dipakai untuk seleksi."""
    clf = HybridModel(hidden_lstm=cfg["hidden_lstm"], hidden_gnn=cfg["hidden_gnn"],
                      dropout=cfg["dropout"], in_features=2)
    _, vloss_clf = train(clf, Xtr_n, ybin_tr, True, cfg["lr"], cfg["batch"], 60, seed,
                         Xva_n, ybin_va)
    p_clf = 1 / (1 + np.exp(-predict(clf, Xte_n)))
    amt = HybridModel(hidden_lstm=cfg["hidden_lstm"], hidden_gnn=cfg["hidden_gnn"],
                      dropout=cfg["dropout"], in_features=2)
    _, vloss_amt = train(amt, Xtr_n, np.log1p(ytr), False, cfg["lr"], cfg["batch"], 60, seed,
                          Xva_n, np.log1p(yva), mask=(ytr > 0), mask_va=(yva > 0))
    p_amt = np.expm1(predict(amt, Xte_n))
    p_amt = np.clip(p_amt, 0, None)
    pred = (p_clf > 0.5).astype(float) * p_amt
    return pred, float(vloss_clf + vloss_amt)


def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2_score(y_true, y_pred))}


# ---------------------------------------------------------------------------
# 3. Random search (Bergstra & Bengio 2012)
# ---------------------------------------------------------------------------
N_TRIALS = 30
rng = np.random.default_rng(42)

def sample_config(rng):
    hidden_lstm = int(np.exp(rng.uniform(np.log(32), np.log(256))))   # log-uniform
    hidden_gnn = int(np.exp(rng.uniform(np.log(32), np.log(256))))    # log-uniform
    dropout = float(rng.uniform(0.0, 0.5))                            # uniform
    lr = float(np.exp(rng.uniform(np.log(1e-4), np.log(1e-2))))       # log-uniform
    batch = int(rng.choice([32, 64, 128]))                            # uniform diskrit
    return {"hidden_lstm": hidden_lstm, "hidden_gnn": hidden_gnn,
            "dropout": dropout, "lr": lr, "batch": batch}

trials = []
for t in range(N_TRIALS):
    cfg = sample_config(rng)
    pred, vloss = fit_zi(cfg)
    m = metrics(yte, pred)
    trials.append({"trial": t + 1, **cfg, "val_loss": vloss, **m})
    print(f"Trial {t + 1:02d}: hl={cfg['hidden_lstm']:3d} hg={cfg['hidden_gnn']:3d} "
          f"do={cfg['dropout']:.2f} lr={cfg['lr']:.2e} bs={cfg['batch']:3d} | "
          f"val_loss={vloss:.4f} | test R2={m['R2']:.4f} RMSE={m['RMSE']:,.0f}", flush=True)

best = min(trials, key=lambda x: x["val_loss"])
print(f"\nKonfigurasi terbaik (val loss): trial {best['trial']} "
      f"hl={best['hidden_lstm']} hg={best['hidden_gnn']} do={best['dropout']:.2f} "
      f"lr={best['lr']:.2e} bs={best['batch']} | test R2={best['R2']:.4f}", flush=True)

# Konfigurasi referensi yang sudah dievaluasi di eksperimen utama
refs = {
    "Hybrid base (128/128, do=0)": {"hidden_lstm": 128, "hidden_gnn": 128, "dropout": 0.0,
                                    "lr": 1e-3, "batch": 64},
    "HybridTuned (128/64, do=0.2)": {"hidden_lstm": 128, "hidden_gnn": 64, "dropout": 0.2,
                                     "lr": 1e-3, "batch": 64},
}

out = {
    "method": "Random search (Bergstra & Bengio, 2012)",
    "search_space": {
        "hidden_lstm": "log-uniform [32, 256]",
        "hidden_gnn": "log-uniform [32, 256]",
        "dropout": "uniform [0.0, 0.5]",
        "lr": "log-uniform [1e-4, 1e-2]",
        "batch": "uniform {32, 64, 128}",
    },
    "n_trials": N_TRIALS,
    "selection": "validation loss (BCE + masked MSE), test dipakai sekali di akhir",
    "seed": 42,
    "trials": trials,
    "best_by_val_loss": best,
    "references": refs,
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=str)
print(f"\nSelesai. Hasil disimpan ke {OUT}")