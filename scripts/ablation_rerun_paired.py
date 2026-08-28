# -*- coding: utf-8 -*-
"""
Rerun ablasi Tabel 6 dengan init terpasang (paired) dalam SATU proses.

Latar belakang audit (26 Agu 2026):
1. Baris "Correlation graph" lama DEGENERATE: tidak ada pasangan region
   dengan |korelasi| > 0.5 (maks 0.332), sehingga graf kosong = adjacency
   identitas -> hasil bit-identik dengan baris Identity. Menyesatkan.
2. Klaim COVID naskah membandingkan lintas-konfigurasi (Hybrid base dgn
   COVID 0.0523 vs HybridTuned tanpa COVID 0.0470) -> tidak valid.
3. Bobot awal model pada run asli diambil dari state RNG global yang
   berbeda-beda antar sel -> perbandingan antar sel tidak terpasang.

Solusi: semua sel dilatih ulang dalam satu proses dengan torch.manual_seed
dipanggil SEBELUM pembuatan model (init identik antar sel), konfigurasi
seragam HybridTuned (128/64/do0.2) seed 42, pipeline identik
experiment_zi_geocoded.py. Varian korelasi baru: k=3 tetangga terkorelasi,
korelasi dari periode TRAINING saja (bebas kebocoran).

Sel (10 fit):
  identity, random x5 (dirata-rata), distance k=3, correlation k=3,
  star Palembang, tanpa-COVID, dan distance dgn-COVID sebagai acuan.
Output: results/ablation_rerun_paired.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
from scipy import stats
import torch
import torch.nn as nn

torch.set_num_threads(4)

DATA = r"E:\Download\Jurnal\data\view_penjualan_detail data hingga oktober.xlsx"
GEO = r"E:\Download\Jurnal\data\customers_geocoded_final.csv"
OUT = r"E:\Download\Jurnal\results\ablation_rerun_paired.json"

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
REGIONS = ["Banyuasin", "Empat Lawang", "Lahat", "Lubuk Linggau", "Muara Enim",
           "Musi Banyuasin", "Musi Rawas", "Ogan Ilir", "Ogan Komering Ilir",
           "Ogan Komering Ulu", "Ogan Komering Ulu Selatan", "Ogan Komering Ulu Timur",
           "Pagar Alam", "Palembang", "Penukal Abab Lematang Ilir", "Prabumulih"]
N = len(REGIONS)

# ---------------------------------------------------------------------------
# Pipeline data (identik experiment_zi_geocoded.py)
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

mu = np.log1p(X_all[tr_mask][:, :, :, 0]).mean()
sd = np.log1p(X_all[tr_mask][:, :, :, 0]).std()
X_all_n = X_all.copy()
X_all_n[:, :, :, 0] = (np.log1p(X_all[:, :, :, 0]) - mu) / sd

X_1n = X_all[:, :, :, :1].copy()
mu1 = np.log1p(X_all[tr_mask][:, :, :, 0]).mean()
sd1 = np.log1p(X_all[tr_mask][:, :, :, 0]).std()
X_1n[:, :, :, 0] = (np.log1p(X_all[:, :, :, 0]) - mu1) / sd1

ytr, yva, yte = y_all[tr_mask], y_all[va_mask], y_all[te_mask]
ybin_tr = (ytr > 0).astype(np.float32)
ybin_va = (yva > 0).astype(np.float32)


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
adj_dist = np.zeros((N, N))
for i in range(N):
    order = np.argsort(D[i])
    for j in order[1:K + 1]:
        adj_dist[i, j] = 1
        adj_dist[j, i] = 1
np.fill_diagonal(adj_dist, 0)

# Graf korelasi k=3: korelasi |Pearson| dari periode TRAINING saja
train_end = pd.Timestamp("2023-12-31")
panel_train = panel[[m.start_time <= train_end for m in panel.index]]
corr_tr = np.corrcoef(panel_train.values.T)
C = np.abs(corr_tr)
np.fill_diagonal(C, 0.0)
third_nn = np.sort(C, axis=1)[:, -K]          # korelasi tangga ke-3 tiap node
CORR_THRESH = float(third_nn.min())            # mirip protokol jarak: min atas node
adj_corr = (C >= CORR_THRESH).astype(float)
np.fill_diagonal(adj_corr, 0)


def build_adj(adj_matrix):
    A_tilde = adj_matrix + np.eye(N)
    D_tilde = A_tilde.sum(axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D_tilde))
    return torch.tensor(D_inv_sqrt @ A_tilde @ D_inv_sqrt, dtype=torch.float32)


A_norm_t = build_adj(adj_dist)


class HybridModel(nn.Module):
    """HybridTuned: LSTM 128 / GCN 64 / dropout 0.2 (make_model else-branch asli)."""

    def __init__(self, hidden_lstm=128, hidden_gnn=64, dropout=0.2, in_features=2):
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
    return model


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


def metrics(y_true, y_pred):
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"RMSE": rmse, "MAE": mae, "R2": float(r2_score(y_true, y_pred))}


def dm_test(e1, e2):
    """DM sederhana pada galat kuadrat (konsisten metodologi naskah)."""
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    denom = np.sqrt(d.var(ddof=1) / T)
    if denom == 0 or not np.isfinite(denom):
        return float("nan"), float("nan")
    dm = float(d.mean() / denom)
    p = float(2 * (1 - stats.norm.cdf(abs(dm))))
    return dm, p


def fit_zi_paired(seed, adj_matrix=None, use_covid=True):
    """Init terpasang: manual_seed SEBELUM pembuatan model."""
    global A_norm_t
    if adj_matrix is not None:
        A_norm_t = build_adj(adj_matrix)
    if use_covid:
        Xtr_, Xva_, Xte_ = X_all_n[tr_mask], X_all_n[va_mask], X_all_n[te_mask]
        in_f = 2
    else:
        Xtr_, Xva_, Xte_ = X_1n[tr_mask], X_1n[va_mask], X_1n[te_mask]
        in_f = 1
    torch.manual_seed(seed)
    clf = HybridModel(in_features=in_f)
    train(clf, Xtr_, ybin_tr, True, 0.001, 64, 60, seed, Xva_, ybin_va)
    p_clf = 1 / (1 + np.exp(-predict(clf, Xte_)))
    torch.manual_seed(seed)
    amt = HybridModel(in_features=in_f)
    train(amt, Xtr_, np.log1p(ytr), False, 0.001, 64, 60, seed,
          Xva_, np.log1p(yva), mask=(ytr > 0), mask_va=(yva > 0))
    p_amt = np.clip(np.expm1(predict(amt, Xte_)), 0, None)
    pred = (p_clf > 0.5).astype(float) * p_amt
    return pred


def run_cell(label, adj_matrix=None, use_covid=True):
    pred = fit_zi_paired(42, adj_matrix, use_covid)
    m = metrics(yte, pred)
    print(f"{label}: RMSE {m['RMSE']:,.0f} MAE {m['MAE']:,.0f} R2 {m['R2']:.4f}", flush=True)
    return m, pred


def main():
    print(f"Graf korelasi k=3 (training-only): threshold {CORR_THRESH:.4f}, "
          f"edge {int(adj_corr.sum() // 2)}", flush=True)
    out = {"config": {"hidden_lstm": 128, "hidden_gnn": 64, "dropout": 0.2,
                      "lr": 0.001, "batch": 64, "seed": 42},
           "corr_graph": {"threshold": CORR_THRESH,
                          "n_edges": int(adj_corr.sum() // 2),
                          "computed_on": "training period only"}}

    preds = {}
    m_identity, p = run_cell("Identity", np.eye(N)); preds["identity"] = p
    rand_ms = []
    for ps in range(5):
        rng = np.random.default_rng(ps)
        perm = rng.permutation(N)
        mm, p = run_cell(f"Random {ps}", adj_dist[perm][:, perm])
        rand_ms.append(mm)
    out["random_mean_of_5"] = {
        "RMSE": float(np.mean([x["RMSE"] for x in rand_ms])),
        "MAE": float(np.mean([x["MAE"] for x in rand_ms])),
        "R2": float(np.mean([x["R2"] for x in rand_ms])),
    }
    m_distance, p = run_cell("Distance k=3 (acuan)", adj_dist); preds["distance"] = p
    m_corr, p = run_cell("Correlation k=3 (baru)", adj_corr); preds["correlation"] = p
    PALEMBANG_IDX = REGIONS.index("Palembang")
    star = np.zeros((N, N))
    star[PALEMBANG_IDX, :] = 1
    star[:, PALEMBANG_IDX] = 1
    np.fill_diagonal(star, 0)
    m_star, p = run_cell("Star Palembang", star); preds["star"] = p
    m_nocovid, p = run_cell("Tanpa COVID", adj_dist, use_covid=False); preds["nocovid"] = p

    e = {k: yte.ravel() - v.ravel() for k, v in preds.items()}
    e_dist = e["distance"]
    comparisons = {}
    for name in ("identity", "correlation", "star", "nocovid"):
        dm, pv = dm_test(e[name], e_dist)
        comparisons[f"{name}_vs_distance"] = {"DM": dm, "p": pv}
        print(f"DM {name} vs distance: {dm:.3f}, p={pv:.4g}", flush=True)

    out.update({
        "identity": m_identity,
        "random_per_perm": rand_ms,
        "distance_k3": m_distance,
        "correlation_k3": m_corr,
        "star_palembang": m_star,
        "without_covid": m_nocovid,
        "dm_comparisons": comparisons,
    })
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"\nSelesai. Disimpan ke {OUT}", flush=True)


if __name__ == "__main__":
    main()
