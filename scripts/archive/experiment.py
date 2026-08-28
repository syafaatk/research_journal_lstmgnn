# -*- coding: utf-8 -*-
"""
Eksperimen ulang: Spatiotemporal Prediction of Medical Device Sales
Using Hybrid LSTM-GNN in South Sumatra (df.xlsx)

Protokol mengikuti revisi.md:
- Target: jual_total (nilai faktur) per kab/kota per bulan
- 17 kab/kota Sumsel (pemetaan dari 34 kota mentah)
- Split 3 arah: training Jul 2020-Des 2023, validasi Jan-Des 2024, testing Jan-Mar 2025
- Power Transformer (Yeo-Johnson) fit training only
- Window 12, horizon 1
- Graf: Haversine, k=3, threshold 95.82 km, undirected, tanpa self-loop
- Model: LSTM, GNN, Hybrid Base, Hybrid Tuned (konfigurasi Tabel 4.12 tesis)
- Metrik pada skala asli (inverse transform) dan skala transformasi
- 5 seeds, Diebold-Mariano, bootstrap 95% CI, ablasi, per wilayah, residual
"""
import json
import math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import r2_score
import torch
import torch.nn as nn

torch.set_num_threads(4)
np.random.seed(42)

DATA = r"E:\Download\Jurnal\view_penjualan_detail data hingga oktober.xlsx"
OUT = r"C:\Users\ADMLAY~1\AppData\Local\Temp\opencode\experiment_results.json"

# ---------------------------------------------------------------------------
# 1. Pemetaan kota mentah -> 17 kab/kota Sumsel
# ---------------------------------------------------------------------------
CITY_MAP = {
    "Palembang": "Palembang",
    "Lahat": "Lahat",
    "Baturaja": "Ogan Komering Ulu",
    # Modifikasi view_penjualan_detail.xlsx: Oku (Ogan Komering Ulu) -> Oku Timur
    "Oku (Ogan Komering Ulu)": "Ogan Komering Ulu Timur",
    "Musi Banyuasin": "Musi Banyuasin",
    "Muara Enim": "Muara Enim",
    "Tanjung Enim": "Muara Enim",
    "Muara Lawai": "Muara Enim",
    "Lubuklinggau": "Lubuk Linggau",
    "Ogan Ilir": "Ogan Ilir",
    "Oku Selatan": "Ogan Komering Ulu Selatan",
    "Prabumulih": "Prabumulih",
    "Prabumulih Timur": "Prabumulih",
    "Pali (Penukal Abab Lematang Ilir)": "Penukal Abab Lematang Ilir",
    "Talang Ubi": "Penukal Abab Lematang Ilir",
    "Musi Rawas": "Musi Rawas",
    "Oku Timur": "Ogan Komering Ulu Timur",
    "Martapura": "Ogan Komering Ulu Timur",
    "Empat Lawang": "Empat Lawang",
    "Banyuasin": "Banyuasin",
    "Pagaralam": "Pagar Alam",
    "Muara Rupit": "Musi Rawas Utara",
    "Rupit": "Musi Rawas Utara",
    "Oki (Ogan Komering Ilir)": "Ogan Komering Ilir",
}
# Kota di luar Sumsel (dibuang): Bangka Belitung, Bengkulu, Bengkulu Utara,
# Jakarta Pusat, Jambi, Lebong, Metro, Pinang, Rejang Lebong, Sungailiat

REGIONS = [
    "Banyuasin", "Empat Lawang", "Lahat", "Lubuk Linggau", "Muara Enim",
    "Musi Banyuasin", "Musi Rawas", "Musi Rawas Utara", "Ogan Ilir",
    "Ogan Komering Ilir", "Ogan Komering Ulu", "Ogan Komering Ulu Selatan",
    "Ogan Komering Ulu Timur", "Pagar Alam", "Palembang",
    "Penukal Abab Lematang Ilir", "Prabumulih",
]

# ---------------------------------------------------------------------------
# 2. Load dan agregasi
# ---------------------------------------------------------------------------
df = pd.read_excel(DATA)
# Buang baris sampah (d_jual_nofak null, tanggal NaT, nilai NaN)
df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
# Satu penjualan = satu d_jual_nofak (faktur); baris lain adalah item dari faktur
# yang sama dengan jual_total_fak diulang. Agregasi harus per faktur unik.
df = df.drop_duplicates("d_jual_nofak")
# Kolom file ini: tanggal, wilayah, jual_total_fak, latitude, longitude
df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                        "jual_total_fak": "jual_total", "latitude": "Latitude"})
print(f"Faktur unik: {len(df)}")

df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
df["region"] = df["pelanggan_kota"].map(CITY_MAP)
df["month"] = df["jual_tanggal"].dt.to_period("M")

# Koordinat per region: mean lokasi pelanggan dari data
coords = df.groupby("region").agg(lat=("Latitude", "mean"), lon=("longitude", "mean"))

# Panel: region x month, sum jual_total
panel = df.pivot_table(index="month", columns="region", values="jual_total", aggfunc="sum")
panel = panel.reindex(columns=REGIONS).fillna(0.0)
panel = panel.sort_index()

months = panel.index.astype(str).tolist()
n_months = len(panel)
print(f"Panel: {n_months} bulan ({months[0]} s.d. {months[-1]}), {len(REGIONS)} region")
print(f"Total entri setelah pemetaan: {len(df)}")

# Statistik deskriptif
zero_frac = (panel == 0).mean(axis=0)
desc = panel.describe().loc[["mean", "std", "min", "max"]].to_dict()

# ---------------------------------------------------------------------------
# 3. Split 3 arah (kronologis)
# ---------------------------------------------------------------------------
train_end = pd.Period("2023-12", "M")
val_end = pd.Period("2024-12", "M")

train_months = panel.index[panel.index <= train_end]
val_months = panel.index[(panel.index > train_end) & (panel.index <= val_end)]
test_months = panel.index[panel.index > val_end]
print(f"Training: {len(train_months)} bulan, Validasi: {len(val_months)}, Testing: {len(test_months)}")

# ---------------------------------------------------------------------------
# 4. Power Transformer (fit training only)
# ---------------------------------------------------------------------------
WINDOW = 12
HORIZON = 1

def make_sequences(month_idx):
    """Window 12 -> target bulan berikutnya. Target harus dalam subset."""
    X, y = [], []
    for i in range(len(month_idx) - WINDOW):
        X.append(panel.loc[month_idx[i:i + WINDOW]].values)  # (12, 17)
        y.append(panel.loc[month_idx[i + WINDOW]].values)    # (17,)
    return np.array(X), np.array(y)

# Window digeser di seluruh panel; sampel diklasifikasikan oleh bulan target.
# Dengan 57 bulan: train 30 sampel (target Jul 2021-Des 2023), val 12 (2024),
# test 3 (Jan-Mar 2025). Window sampel val/test menjorok ke bulan sebelumnya.
X_all, y_all, t_all = [], [], []
for i in range(n_months - WINDOW):
    X_all.append(panel.iloc[i:i + WINDOW].values)
    y_all.append(panel.iloc[i + WINDOW].values)
    t_all.append(panel.index[i + WINDOW])
X_all, y_all = np.array(X_all), np.array(y_all)
# (B, T, N) -> (B, N, T, 1) sesuai bentuk input model
X_all = X_all.transpose(0, 2, 1)[:, :, :, None]

tr_mask = np.array([m <= train_end for m in t_all])
va_mask = np.array([(m > train_end) & (m <= val_end) for m in t_all])
te_mask = np.array([m > val_end for m in t_all])
Xtr, ytr = X_all[tr_mask], y_all[tr_mask]
Xva, yva = X_all[va_mask], y_all[va_mask]
Xte, yte = X_all[te_mask], y_all[te_mask]

# Fit transformer pada training (flatten per region)
pt = PowerTransformer(method="yeo-johnson")
flat_tr = ytr.reshape(-1, 1)
pt.fit(flat_tr)

def transform(y):
    return pt.transform(y.reshape(-1, 1)).reshape(y.shape)

def inverse(y):
    return pt.inverse_transform(y.reshape(-1, 1)).reshape(y.shape)

ytr_t, yva_t, yte_t = transform(ytr), transform(yva), transform(yte)
Xtr_t, Xva_t, Xte_t = transform(Xtr), transform(Xva), transform(Xte)

print(f"Sampel: train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}")
print(f"Zero actuals (test, skala asli): {(yte == 0).sum()} dari {yte.size}")

# ---------------------------------------------------------------------------
# 5. Graf: Haversine, k=3, threshold 95.82 km
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

N = len(REGIONS)
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
    for j in order[1:K + 1]:  # skip self
        if D[i, j] <= THRESHOLD:
            adj[i, j] = 1
            adj[j, i] = 1
np.fill_diagonal(adj, 0)

n_edges = int(adj.sum() / 2)
degrees = adj.sum(axis=1)
print(f"Graf: {n_edges} edge (undirected), derajat min {degrees.min():.0f}, "
      f"max {degrees.max():.0f}, rata-rata {degrees.mean():.2f}, "
      f"isolated: {(degrees == 0).sum()}")

# Justifikasi threshold: distribusi jarak tetangga terdekat
knn_dists = []
for i in range(N):
    order = np.argsort(D[i])
    knn_dists.append(D[i, order[1:K + 1]])
knn_dists = np.array(knn_dists)
print(f"Jarak 3-NN: min {knn_dists.min():.2f}, median {np.median(knn_dists):.2f}, "
      f"max {knn_dists.max():.2f} km; threshold 95.82 km memotong "
      f"{(knn_dists > THRESHOLD).sum()} dari {knn_dists.size} kandidat edge")

# Adjacency untuk GCN (A~ = A + I, D~^-1/2 A~ D~^-1/2)
A_tilde = adj + np.eye(N)
D_tilde = A_tilde.sum(axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(D_tilde))
A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt
A_norm_t = torch.tensor(A_norm, dtype=torch.float32)

# ---------------------------------------------------------------------------
# 6. Model
# ---------------------------------------------------------------------------
class LSTMModel(nn.Module):
    def __init__(self, hidden=128, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):  # (B, N, 12, 1)
        B, N, T, F = x.shape
        h, _ = self.lstm(x.view(B * N, T, F))
        h = h[:, -1, :]  # (B*N, hidden)
        return self.fc(self.drop(h)).view(B, N)


class GNNModel(nn.Module):
    def __init__(self, hidden=128, dropout=0.0):
        super().__init__()
        self.gcn = nn.Linear(1, hidden)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):  # (B, N, 12, 1) -> pakai nilai terakhir
        B, N, T, F = x.shape
        x_last = x[:, :, -1, :]  # (B, N, 1)
        h = torch.relu(self.gcn(x_last))
        h = A_norm_t @ h  # (B, N, hidden)
        return self.fc(self.drop(h)).view(B, N)


class HybridModel(nn.Module):
    def __init__(self, hidden_lstm=128, hidden_gnn=64, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden_lstm, batch_first=True)
        self.gcn = nn.Linear(hidden_lstm, hidden_gnn)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_lstm + hidden_gnn, 1)

    def forward(self, x):  # (B, N, 12, 1)
        B, N, T, F = x.shape
        h, _ = self.lstm(x.view(B * N, T, F))
        h_lstm = h[:, -1, :].view(B, N, -1)  # (B, N, hidden_lstm)
        h_gcn = torch.relu(self.gcn(h_lstm))  # (B, N, hidden_gnn)
        h_gcn = A_norm_t @ h_gcn
        fused = torch.cat([h_lstm, h_gcn], dim=-1)  # (B, N, h_l+h_g)
        return self.fc(self.drop(fused)).view(B, N)


# ---------------------------------------------------------------------------
# 7. Training
# ---------------------------------------------------------------------------
def train_model(model, X, y, lr, batch, epochs, seed, patience=10):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    n = len(Xt)
    best_loss = float("inf")
    best_state = None
    bad = 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = lossf(out, yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        # validasi
        model.eval()
        with torch.no_grad():
            vloss = lossf(model(Xt), yt).item()
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


def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = float("nan")
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


CONFIGS = {
    "LSTM": dict(cls=LSTMModel, hidden=128, dropout=0.0, lr=0.001, batch=16, epochs=50),
    "GNN": dict(cls=GNNModel, hidden=128, dropout=0.0, lr=0.001, batch=8, epochs=50),
    "Hybrid": dict(cls=HybridModel, hidden_lstm=128, hidden_gnn=128, dropout=0.0,
                   lr=0.005, batch=8, epochs=50),
    "HybridTuned": dict(cls=HybridModel, hidden_lstm=128, hidden_gnn=64, dropout=0.2,
                        lr=0.001, batch=32, epochs=100),
}

SEEDS = [42, 7, 123, 2024, 99]
results = {}

for name, cfg in CONFIGS.items():
    rmses_t, maes_t, mapes_t, r2s_t = [], [], [], []
    rmses_o, maes_o, mapes_o, r2s_o = [], [], [], []
    preds_o_all = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = cfg["cls"](**{k: v for k, v in cfg.items()
                              if k in ("hidden", "hidden_lstm", "hidden_gnn", "dropout")})
        train_model(model, Xtr_t, ytr_t, cfg["lr"], cfg["batch"], cfg["epochs"], seed)
        # skala transformasi
        p_t = predict(model, Xte_t)
        m_t = metrics(yte_t, p_t)
        # skala asli
        p_o = inverse(p_t)
        m_o = metrics(yte, p_o)
        rmses_t.append(m_t["RMSE"]); maes_t.append(m_t["MAE"])
        mapes_t.append(m_t["MAPE"]); r2s_t.append(m_t["R2"])
        rmses_o.append(m_o["RMSE"]); maes_o.append(m_o["MAE"])
        mapes_o.append(m_o["MAPE"]); r2s_o.append(m_o["R2"])
        preds_o_all.append(p_o)
    results[name] = {
        "transformed": {"RMSE": rmses_t, "MAE": maes_t, "MAPE": mapes_t, "R2": r2s_t},
        "original": {"RMSE": rmses_o, "MAE": maes_o, "MAPE": mapes_o, "R2": r2s_o},
        "preds_original": np.mean(preds_o_all, axis=0).tolist(),
    }
    print(f"{name}: RMSE {np.mean(rmses_o):.4f} +- {np.std(rmses_o):.4f} | "
          f"MAE {np.mean(maes_o):.4f} | MAPE {np.mean(mapes_o):.2f} | R2 {np.mean(r2s_o):.4f}")

# ---------------------------------------------------------------------------
# 8. Diebold-Mariano test (skala asli, seed 42)
# ---------------------------------------------------------------------------
def dm_test(e1, e2, h=1):
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    var = np.var(d, ddof=1) / n
    if var == 0:
        return float("nan")
    dm = dbar / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return dm, p

# Re-train seed 42 untuk DM
dm_results = {}
preds_ref = {}
for name, cfg in CONFIGS.items():
    torch.manual_seed(42)
    model = cfg["cls"](**{k: v for k, v in cfg.items()
                          if k in ("hidden", "hidden_lstm", "hidden_gnn", "dropout")})
    train_model(model, Xtr_t, ytr_t, cfg["lr"], cfg["batch"], cfg["epochs"], 42)
    preds_ref[name] = inverse(predict(model, Xte_t))

e_ref = yte - preds_ref["HybridTuned"]
for name in ["LSTM", "GNN", "Hybrid"]:
    e_other = yte - preds_ref[name]
    dm, p = dm_test(e_other.flatten(), e_ref.flatten())
    dm_results[name] = {"DM": dm, "p": p}
    print(f"DM {name} vs HybridTuned: DM={dm:.3f}, p={p:.4f}")

# Bootstrap 95% CI untuk RMSE (skala asli, seed 42)
def bootstrap_rmse_ci(y_true, y_pred, n_boot=1000, alpha=0.05):
    rng = np.random.default_rng(42)
    errs = (y_true - y_pred).flatten()
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(errs), len(errs))
        boots.append(np.sqrt(np.mean(errs[idx] ** 2)))
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))

boot_ci = {}
for name in ["LSTM", "GNN", "Hybrid", "HybridTuned"]:
    lo, hi = bootstrap_rmse_ci(yte, preds_ref[name])
    boot_ci[name] = [lo, hi]
    print(f"Bootstrap RMSE CI {name}: [{lo:.4f}, {hi:.4f}]")

# ---------------------------------------------------------------------------
# 9. Ablasi (seed 42, skala asli)
# ---------------------------------------------------------------------------
def run_ablation(adj_matrix, label, hidden_lstm=128, hidden_gnn=64, dropout=0.2):
    global A_norm_t
    A_ = adj_matrix + np.eye(N)
    D_ = A_.sum(axis=1)
    A_n = np.diag(1.0 / np.sqrt(D_)) @ A_ @ np.diag(1.0 / np.sqrt(D_))
    A_norm_t = torch.tensor(A_n, dtype=torch.float32)
    torch.manual_seed(42)
    model = HybridModel(hidden_lstm, hidden_gnn, dropout)
    train_model(model, Xtr_t, ytr_t, 0.001, 32, 100, 42)
    p_o = inverse(predict(model, Xte_t))
    m = metrics(yte, p_o)
    print(f"Ablasi {label}: RMSE {m['RMSE']:.4f}, MAE {m['MAE']:.4f}, R2 {m['R2']:.4f}")
    return m

ablation = {}
# LSTM only (tanpa graf): pakai HybridModel tapi gcn weight -> 0? Lebih bersih: LSTM + FC
torch.manual_seed(42)
lstm_only = LSTMModel(hidden=128, dropout=0.2)
train_model(lstm_only, Xtr_t, ytr_t, 0.001, 32, 100, 42)
m = metrics(yte, inverse(predict(lstm_only, Xte_t)))
ablation["LSTM only"] = m

ablation["Identity adjacency"] = run_ablation(np.eye(N), "Identity")
# Random graph: 10 permutasi, mean
rand_metrics = []
for perm_seed in range(10):
    rng = np.random.default_rng(perm_seed)
    perm = rng.permutation(N)
    A_rand = adj[perm][:, perm]
    m = run_ablation(A_rand, f"Random {perm_seed}")
    rand_metrics.append(m)
ablation["Random graph (mean of 10)"] = {
    "RMSE": float(np.mean([x["RMSE"] for x in rand_metrics])),
    "MAE": float(np.mean([x["MAE"] for x in rand_metrics])),
    "R2": float(np.mean([x["R2"] for x in rand_metrics])),
}
ablation["Distance graph k=3"] = run_ablation(adj, "Distance")
# Correlation graph
corr = np.corrcoef(panel.values.T)
corr_adj = (np.abs(corr) > 0.5).astype(float)
np.fill_diagonal(corr_adj, 0)
ablation["Correlation graph"] = run_ablation(corr_adj, "Correlation")

# ---------------------------------------------------------------------------
# 10. Per wilayah (HybridTuned, seed 42, skala asli)
# ---------------------------------------------------------------------------
torch.manual_seed(42)
model = HybridModel(128, 64, 0.2)
train_model(model, Xtr_t, ytr_t, 0.001, 32, 100, 42)
p_o = inverse(predict(model, Xte_t))

per_region = {}
for r in range(N):
    m = metrics(yte[:, r], p_o[:, r])
    per_region[REGIONS[r]] = {k: float(v) for k, v in m.items()}
best = min(per_region, key=lambda k: per_region[k]["RMSE"])
worst = max(per_region, key=lambda k: per_region[k]["RMSE"])
print(f"Per wilayah terbaik: {best} ({per_region[best]['RMSE']:.4f}), "
      f"terburuk: {worst} ({per_region[worst]['RMSE']:.4f})")

# ---------------------------------------------------------------------------
# 11. Residual (HybridTuned, seed 42, skala asli)
# ---------------------------------------------------------------------------
resid = (yte - p_o).flatten()
mean_resid = float(resid.mean())
# Ljung-Box (manual)
def ljung_box(x, lags=10):
    n = len(x)
    x = x - x.mean()
    acf = np.array([np.corrcoef(x[:-l], x[l:])[0, 1] if l < n else 0 for l in range(1, lags + 1)])
    acf = np.nan_to_num(acf)
    q = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, lags + 1)))
    p = 1 - stats.chi2.cdf(q, lags)
    return q, p

q_lb, p_lb = ljung_box(resid)
print(f"Residual: mean {mean_resid:.4f}, Ljung-Box Q={q_lb:.3f}, p={p_lb:.4f}")

# ---------------------------------------------------------------------------
# 12. Simpan hasil
# ---------------------------------------------------------------------------
out = {
    "panel": {"n_months": n_months, "months": months, "n_regions": N,
              "n_records_mapped": int(len(df)),
              "zero_frac": {k: float(v) for k, v in zero_frac.items()},
              "desc": desc},
    "split": {"train": len(Xtr), "val": len(Xva), "test": len(Xte),
              "train_months": [str(m) for m in train_months],
              "val_months": [str(m) for m in val_months],
              "test_months": [str(m) for m in test_months]},
    "graph": {"n_edges": n_edges, "degrees": degrees.tolist(),
              "knn_dists": knn_dists.tolist(), "threshold": THRESHOLD,
              "coords": coords.to_dict()},
    "zero_actuals_test": int((yte == 0).sum()),
    "models": results,
    "dm": dm_results,
    "bootstrap_ci": boot_ci,
    "ablation": ablation,
    "per_region": per_region,
    "residual": {"mean": mean_resid, "ljung_box_q": q_lb, "ljung_box_p": p_lb},
    "configs": {k: {kk: vv for kk, vv in v.items() if kk != "cls"} for k, v in CONFIGS.items()},
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=str)
print("Selesai. Hasil disimpan ke", OUT)