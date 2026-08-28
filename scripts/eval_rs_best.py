"""Evaluasi konfigurasi terbaik random search (trial 30) dengan 3 seeds.
Pipeline identik random_search_bb.py. Output: metrics per-seed, mean, ensemble,
DM vs Hybrid base, bootstrap CI. Disimpan ke skrip/rs_best_eval.json.
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from scipy import stats

DATA = r"E:\Download\Jurnal\data\view_penjualan_detail data hingga oktober.xlsx"
GEO = r"E:\Download\Jurnal\data\customers_geocoded_final.csv"
OUT_JSON = r"E:\Download\Jurnal\results\experiment_results_zi_geocoded.json"
OUT = r"E:\Download\Jurnal\results\rs_best_eval.json"

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

# Pipeline data (identik random_search_bb.py)
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


CFG = {"hidden_lstm": 219, "hidden_gnn": 107, "dropout": 0.17343490226741853,
       "lr": 0.001519955884094079, "batch": 32}
SEEDS = [42, 7, 123]


def fit_zi_seed(cfg, seed):
    clf = HybridModel(hidden_lstm=cfg["hidden_lstm"], hidden_gnn=cfg["hidden_gnn"],
                      dropout=cfg["dropout"], in_features=2)
    _, _ = train(clf, Xtr_n, ybin_tr, True, cfg["lr"], cfg["batch"], 60, seed, Xva_n, ybin_va)
    p_clf = 1 / (1 + np.exp(-predict(clf, Xte_n)))
    amt = HybridModel(hidden_lstm=cfg["hidden_lstm"], hidden_gnn=cfg["hidden_gnn"],
                      dropout=cfg["dropout"], in_features=2)
    _, _ = train(amt, Xtr_n, np.log1p(ytr), False, cfg["lr"], cfg["batch"], 60, seed,
                 Xva_n, np.log1p(yva), mask=(ytr > 0), mask_va=(yva > 0))
    p_amt = np.expm1(predict(amt, Xte_n))
    p_amt = np.clip(p_amt, 0, None)
    pred = (p_clf > 0.5).astype(float) * p_amt
    return pred, p_clf


def metrics(y_true, y_pred, p_clf=None):
    m = {"RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
         "MAE": float(np.mean(np.abs(y_true - y_pred))),
         "R2": float(r2_score(y_true, y_pred))}
    if p_clf is not None:
        ybin = (y_true > 0).astype(int)
        m["ACC"] = float(accuracy_score(ybin.ravel(), (p_clf > 0.5).astype(int).ravel()))
        m["AUC"] = float(roc_auc_score(ybin.ravel(), p_clf.ravel()))
    return m


def dm_test(e1, e2):
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    dm = d.mean() / np.sqrt(d.var(ddof=1) / T)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)


def bootstrap_ci(e1, e2, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    T = len(e1)
    diffs = e1 ** 2 - e2 ** 2
    boots = np.array([diffs[rng.integers(0, T, T)].mean() for _ in range(n_boot)])
    return boots


def main():
    preds, p_clfs = [], []
    per_seed = []
    for s in SEEDS:
        pred, p_clf = fit_zi_seed(CFG, s)
        preds.append(pred)
        p_clfs.append(p_clf)
        m = metrics(yte, pred, p_clf)
        per_seed.append({"seed": s, **m})
        print("Seed {}: R2={:.4f} RMSE={:,.0f} MAE={:,.0f} ACC={:.4f} AUC={:.4f}".format(
            s, m["R2"], m["RMSE"], m["MAE"], m["ACC"], m["AUC"]), flush=True)

    rmse = np.array([p["RMSE"] for p in per_seed])
    mae = np.array([p["MAE"] for p in per_seed])
    r2 = np.array([p["R2"] for p in per_seed])

    pred_ens = np.mean(preds, axis=0)
    p_clf_ens = np.mean(p_clfs, axis=0)
    m_ens = metrics(yte, pred_ens, p_clf_ens)
    print("Ensemble: R2={:.4f} RMSE={:,.0f} MAE={:,.0f} ACC={:.4f} AUC={:.4f}".format(
        m_ens["R2"], m_ens["RMSE"], m_ens["MAE"], m_ens["ACC"], m_ens["AUC"]), flush=True)

    # DM vs Hybrid base (preds_mean dari JSON eksperimen utama)
    res = json.load(open(OUT_JSON))
    pred_hybrid_base = np.array(res["models"]["Hybrid"]["preds_mean"])
    e1 = yte.ravel() - pred_ens.ravel()
    e2 = yte.ravel() - pred_hybrid_base.ravel()
    dm, p = dm_test(e1, e2)
    print("DM vs Hybrid base: DM={:.3f}, p={:.4f}".format(dm, p), flush=True)

    boots = bootstrap_ci(e1, e2)
    # Bootstrap CI RMSE untuk prediksi model ini (mean over seeds)
    rng = np.random.default_rng(42)
    T = len(e1)
    idxs = rng.integers(0, T, (1000, T))
    boot_mse = np.array([np.mean(e1[i] ** 2) for i in idxs])
    ci_this = np.sqrt(np.percentile(boot_mse, [2.5, 97.5]))

    print("Bootstrap CI RMSE (ensemble): [{:,.0f}, {:,.0f}]".format(ci_this[0], ci_this[1]), flush=True)

    out = {
        "config": CFG,
        "seeds": SEEDS,
        "per_seed": per_seed,
        "mean": {"RMSE": float(rmse.mean()), "RMSE_std": float(rmse.std(ddof=1)),
                 "MAE": float(mae.mean()), "MAE_std": float(mae.std(ddof=1)),
                 "R2": float(r2.mean()), "R2_std": float(r2.std(ddof=1))},
        "ensemble": m_ens,
        "dm_vs_hybrid_base": {"DM": dm, "p": p},
        "bootstrap_ci_rmse_ensemble": [float(ci_this[0]), float(ci_this[1])],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("Selesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()
