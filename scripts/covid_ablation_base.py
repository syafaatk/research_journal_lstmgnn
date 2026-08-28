# -*- coding: utf-8 -*-
"""
Ablasi fitur COVID-19 pada konfigurasi Hybrid BASE (128/128, dropout 0).

Latar belakang (temuan audit 21 Agu 2026):
- Klaim naskah "removing COVID decreases R2 from 0.0523 to 0.0470" membandingkan
  DUA arsitektur berbeda sekaligus (Hybrid base dgn COVID vs HybridTuned tanpa
  COVID) -> perbandingan tidak valid.
- Satu-satunya pembanding se-config di experiment_zi_geocoded.py adalah level
  HybridTuned seed 42: 0.0458 (dengan) vs 0.0470 (tanpa).
- Script ini mengisi sel yang hilang: Hybrid BASE tanpa COVID, 3 seed (42/7/123),
  pipeline IDENTIK dengan experiment_zi_geocoded.py (panel, split, normalisasi
  anti-leakage, graf geocoded k=3, arsitektur, training).

Output: results/covid_ablation_base.json
"""
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, roc_auc_score
import torch
import torch.nn as nn

torch.set_num_threads(4)
np.random.seed(42)

DATA = r"E:\Download\Jurnal\data\view_penjualan_detail data hingga oktober.xlsx"
GEO = r"E:\Download\Jurnal\data\customers_geocoded_final.csv"
REF = r"E:\Download\Jurnal\results\experiment_results_zi_geocoded.json"
OUT = r"E:\Download\Jurnal\results\covid_ablation_base.json"

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
# 1. Load dan agregasi harian (identik experiment_zi_geocoded.py)
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

# ---------------------------------------------------------------------------
# 2. Split 3 arah + windowing (identik)
# ---------------------------------------------------------------------------
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

X_all_1 = X_all[:, :, :, :1]
mu1 = np.log1p(X_all_1[tr_mask][:, :, :, 0]).mean()
sd1 = np.log1p(X_all_1[tr_mask][:, :, :, 0]).std()
X_all_1n = X_all_1.copy()
X_all_1n[:, :, :, 0] = (np.log1p(X_all_1[:, :, :, 0]) - mu1) / sd1
Xtr_n1, Xva_n1, Xte_n1 = X_all_1n[tr_mask], X_all_1n[va_mask], X_all_1n[te_mask]

ybin_tr = (ytr > 0).astype(np.float32)
ybin_va = (yva > 0).astype(np.float32)

# ---------------------------------------------------------------------------
# 3. Graf (identik)
# ---------------------------------------------------------------------------
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
knn_dists = np.array([np.sort(D[i])[1:K + 1] for i in range(N)])
THRESHOLD = float(knn_dists.max())
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

# ---------------------------------------------------------------------------
# 4. Model Hybrid base (identik)
# ---------------------------------------------------------------------------
class HybridModel(nn.Module):
    def __init__(self, hidden_lstm=128, hidden_gnn=64, dropout=0.0, in_features=2):
        super().__init__()
        self.lstm = nn.LSTM(in_features, hidden_lstm, batch_first=True)
        self.gcn = nn.Linear(hidden_lstm, hidden_gnn)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_lstm + hidden_gnn, 1)

    def forward(self, x):
        B, N_, T, F = x.shape
        h, _ = self.lstm(x.reshape(B * N_, T, F))
        h_lstm = h[:, -1, :].view(B, N_, -1)
        h_gcn = torch.relu(self.gcn(h_lstm))
        h_gcn = A_norm_t @ h_gcn
        fused = torch.cat([h_lstm, h_gcn], dim=-1)
        return self.fc(self.drop(fused)).view(B, N_)


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


def fit_zi(seed, use_covid=True):
    global A_norm_t
    if use_covid:
        Xtr_, Xva_, Xte_ = Xtr_n, Xva_n, Xte_n
        in_f = 2
    else:
        Xtr_, Xva_, Xte_ = Xtr_n1, Xva_n1, Xte_n1
        in_f = 1
    clf = HybridModel(hidden_lstm=128, hidden_gnn=128, dropout=0.0, in_features=in_f)
    train(clf, Xtr_, ybin_tr, True, 0.001, 64, 60, seed, Xva_, ybin_va)
    p_clf = 1 / (1 + np.exp(-predict(clf, Xte_)))
    amt = HybridModel(hidden_lstm=128, hidden_gnn=128, dropout=0.0, in_features=in_f)
    train(amt, Xtr_, np.log1p(ytr), False, 0.001, 64, 60, seed,
          Xva_, np.log1p(yva), mask=(ytr > 0), mask_va=(yva > 0))
    p_amt = np.expm1(predict(amt, Xte_))
    p_amt = np.clip(p_amt, 0, None)
    pred = (p_clf > 0.5).astype(float) * p_amt
    return pred, p_clf


def metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {"RMSE": float(rmse), "MAE": float(mae),
            "R2": float(r2_score(y_true, y_pred))}


def dm_test(e1, e2, h=1):
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dbar = d.mean()
    def acf(x, lag):
        x = x - x.mean()
        return np.corrcoef(x[:-lag], x[lag:])[0, 1] if lag < n else 0.0
    var_d = (d - dbar).var(ddof=0) / n
    for l in range(1, h + 1):
        g = np.cov(d[l:], d[:-l])[0, 1]
        var_d += 2 * g / n
    dm = dbar / np.sqrt(var_d)
    p = 2 * stats.t.sf(np.abs(dm), df=n - 1)
    return float(dm), float(p)


def bootstrap_ci_rmse(y_true, y_pred, n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.integers(0, n, size=(n_boot, n))
    rmses = np.sqrt(((y_true[idx] - y_pred[idx]) ** 2).mean(axis=1))
    return [float(np.quantile(rmses, alpha / 2)), float(np.quantile(rmses, 1 - alpha / 2))]


# ---------------------------------------------------------------------------
# 5. Jalankan: sanity retrain with-COVID seed 42, lalu tanpa-COVID 3 seed
# ---------------------------------------------------------------------------
SEEDS = [42, 7, 123]
out = {"config": {"hidden_lstm": 128, "hidden_gnn": 128, "dropout": 0.0,
                  "lr": 0.001, "batch": 64}, "seeds": SEEDS}

print("=== Sanity: Hybrid base WITH COVID, seed 42 (harus cocok JSON referensi) ===", flush=True)
pred_c42, pclf_c42 = fit_zi(42, use_covid=True)
m_c42 = metrics(yte, pred_c42)
ref = json.load(open(REF, encoding="utf-8"))
ref_r2_42 = ref["models"]["Hybrid"]["R2"][0]
print(f"retrain R2 {m_c42['R2']:.4f} vs referensi {ref_r2_42:.4f}", flush=True)
out["sanity_with_covid_seed42"] = {"retrained": m_c42, "reference_R2": ref_r2_42}

print("=== Hybrid base TANPA COVID, 3 seed ===", flush=True)
per_seed = []
preds_all = []
for seed in SEEDS:
    pred, p_clf = fit_zi(seed, use_covid=False)
    m = metrics(yte, pred)
    auc = roc_auc_score((yte > 0).astype(int).ravel(), p_clf.ravel())
    acc = ((p_clf > 0.5) == (yte > 0)).mean()
    m["ACC"] = float(acc); m["AUC"] = float(auc); m["seed"] = seed
    per_seed.append(m)
    preds_all.append(pred)
    print(f"seed {seed}: R2 {m['R2']:.4f} RMSE {m['RMSE']:,.0f} MAE {m['MAE']:,.0f} "
          f"ACC {acc:.4f} AUC {auc:.4f}", flush=True)

ens = np.mean(preds_all, axis=0)
m_ens = metrics(yte, ens)
print(f"ensemble: R2 {m_ens['R2']:.4f} RMSE {m_ens['RMSE']:,.0f}", flush=True)

# DM ensemble: tanpa-COVID vs dengan-COVID (preds_mean dari JSON referensi)
pred_with = np.array(ref["models"]["Hybrid"]["preds_mean"])
dm, p_dm = dm_test(yte - ens, yte - pred_with)
ci = bootstrap_ci_rmse(yte, ens)
print(f"DM (tanpa vs dengan COVID): {dm:.3f}, p={p_dm:.4g}", flush=True)
print(f"Bootstrap CI RMSE tanpa-COVID: [{ci[0]:,.0f}, {ci[1]:,.0f}]", flush=True)

r2s = [m["R2"] for m in per_seed]
out["without_covid"] = {
    "per_seed": per_seed,
    "R2_mean": float(np.mean(r2s)), "R2_std": float(np.std(r2s)),
    "ensemble": m_ens,
    "dm_vs_with_covid": {"DM": dm, "p": p_dm},
    "bootstrap_ci_rmse_ensemble": ci,
}
with_ref = ref["models"]["Hybrid"]
out["with_covid_reference"] = {
    "R2_per_seed": with_ref["R2"],
    "R2_mean": float(np.mean(with_ref["R2"])),
    "R2_std": float(np.std(with_ref["R2"])),
}

json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
print(f"\nSelesai. Hasil: {OUT}", flush=True)
