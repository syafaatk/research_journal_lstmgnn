# -*- coding: utf-8 -*-
"""Eksperimen V4: temuan baru dimasukkan ke model.

Temuan yang diuji:
  E11 (siklus fiskal): puncak pengadaan syringe Juli & November ->
      fitur month-of-year one-hot (12 fitur tambahan).
  E8/E9 (outlier produk mahal): Airvo 762,5 jt, Blanketrol 441,5 jt ->
      target clipping amount stage pada quantile tinggi (robust terhadap
      outlier yang menarik loss).

Baseline: V3 (random-search config 219/107/0.1734/1.52e-3/32 + V2 calendar),
          R2 0.0666, RMSE 7.609.122 (results/exp_features_v3.json).

Varian:
  V3_ref          : replikasi V3 (untuk DM test & perbandingan)
  V4a_month       : V3 + month-of-year one-hot
  V4b_clip        : V3 + target clipping amount (p99.5 train)
  V4c_month_clip  : V4a + V4b

Pipeline identik exp_features.py (16 region, anti-leakage, log1p train-only,
zero-inflated clf+amt, 3 seeds ensemble). Evaluasi tetap pada skala asli.

Output: results/exp_temuan.json + log.
"""
import json
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
torch.set_num_threads(4)

DATA = r"E:\Download\Jurnal\data\view_penjualan_detail data hingga oktober.xlsx"
GEO = r"E:\Download\Jurnal\data\customers_geocoded_final.csv"
OUT = r"E:\Download\Jurnal\results\exp_temuan.json"

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

HOLIDAYS = {
    "2020-01-01", "2020-01-25", "2020-03-22", "2020-03-25", "2020-05-01", "2020-05-07",
    "2020-05-21", "2020-05-24", "2020-05-25", "2020-07-31", "2020-08-17", "2020-10-29",
    "2020-12-24", "2020-12-25",
    "2021-01-01", "2021-02-12", "2021-03-11", "2021-03-14", "2021-05-01", "2021-05-13",
    "2021-05-14", "2021-05-26", "2021-07-20", "2021-08-17", "2021-10-19", "2021-12-24",
    "2021-12-25",
    "2022-01-01", "2022-02-01", "2022-02-28", "2022-03-03", "2022-05-01", "2022-05-02",
    "2022-05-03", "2022-05-16", "2022-05-26", "2022-07-10", "2022-08-17", "2022-10-08",
    "2022-12-24", "2022-12-25",
    "2023-01-01", "2023-01-22", "2023-02-18", "2023-03-22", "2023-04-22", "2023-04-23",
    "2023-05-01", "2023-05-18", "2023-06-04", "2023-06-29", "2023-08-17", "2023-09-28",
    "2023-12-24", "2023-12-25",
    "2024-01-01", "2024-02-08", "2024-02-10", "2024-03-11", "2024-04-10", "2024-04-11",
    "2024-05-01", "2024-05-09", "2024-05-23", "2024-06-17", "2024-08-17", "2024-09-16",
    "2024-12-24", "2024-12-25",
    "2025-01-01", "2025-01-27", "2025-01-29", "2025-03-29", "2025-03-31", "2025-04-01",
    "2025-05-01", "2025-05-12", "2025-05-29", "2025-06-06", "2025-08-17", "2025-09-05",
    "2025-12-24", "2025-12-25",
}
HOL = pd.to_datetime(list(HOLIDAYS))

# Pipeline data
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
    def __init__(self, hidden_lstm=128, hidden_gnn=128, dropout=0.0, in_features=2):
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


def bootstrap_ci_rmse(e, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    T = len(e)
    idxs = rng.integers(0, T, (n_boot, T))
    boot_mse = np.array([np.mean(e[i] ** 2) for i in idxs])
    return np.sqrt(np.percentile(boot_mse, [2.5, 97.5]))


# Fitur kalender hari target (konstan sepanjang window)
def build_calendar_features(t_all, variant, with_month=False):
    feats = []
    for m in t_all:
        t = m.start_time
        dow = t.dayofweek
        is_weekend = 1.0 if dow >= 5 else 0.0
        is_hol = 1.0 if t in HOL else 0.0
        if t <= pd.Timestamp("2020-12-31"):
            fase = [1.0, 0.0]
        elif t <= pd.Timestamp("2022-12-31"):
            fase = [0.0, 1.0]
        else:
            fase = [0.0, 0.0]
        if variant == "V1":
            base = [is_weekend, is_hol] + fase
        else:  # V2: dow one-hot
            onehot = [1.0 if d == dow else 0.0 for d in range(7)]
            base = onehot + [is_hol] + fase
        if with_month:
            month = [1.0 if mo == t.month else 0.0 for mo in range(1, 13)]
            base = base + month
        feats.append(base)
    return np.array(feats, dtype=np.float32)


def add_calendar(X_np, cal_feats):
    B, Nn, T, F = X_np.shape
    n_extra = cal_feats.shape[1]
    out = np.zeros((B, Nn, T, F + n_extra), dtype=np.float32)
    out[:, :, :, :F] = X_np
    for b in range(B):
        out[b, :, :, F:] = np.broadcast_to(cal_feats[b], (Nn, T, n_extra))
    return out


CFG = {"hidden_lstm": 219, "hidden_gnn": 107, "dropout": 0.1734, "lr": 0.00152, "batch": 32}
SEEDS = [42, 7, 123]


def fit_zi_seed(Xtr_f, Xva_f, Xte_f, cfg, seed, clip_q=None):
    clf = HybridModel(hidden_lstm=cfg["hidden_lstm"], hidden_gnn=cfg["hidden_gnn"],
                      dropout=cfg["dropout"], in_features=Xtr_f.shape[-1])
    _, _ = train(clf, Xtr_f, ybin_tr, True, cfg["lr"], cfg["batch"], 60, seed, Xva_f, ybin_va)
    p_clf = 1 / (1 + np.exp(-predict(clf, Xte_f)))
    amt = HybridModel(hidden_lstm=cfg["hidden_lstm"], hidden_gnn=cfg["hidden_gnn"],
                      dropout=cfg["dropout"], in_features=Xtr_f.shape[-1])
    ytr_log = np.log1p(ytr)
    yva_log = np.log1p(yva)
    if clip_q is not None:
        pos = ytr_log[ytr > 0]
        cap = float(np.quantile(pos, clip_q))
        ytr_log = np.minimum(ytr_log, cap)
        yva_log = np.minimum(yva_log, cap)
    _, _ = train(amt, Xtr_f, ytr_log, False, cfg["lr"], cfg["batch"], 60, seed,
                  Xva_f, yva_log, mask=(ytr > 0), mask_va=(yva > 0))
    p_amt = np.expm1(predict(amt, Xte_f))
    p_amt = np.clip(p_amt, 0, None)
    pred = (p_clf > 0.5).astype(float) * p_amt
    return pred, p_clf


def main():
    cal_v2 = build_calendar_features(t_all, "V2")
    cal_v2_month = build_calendar_features(t_all, "V2", with_month=True)

    variants = {
        "V3_ref": (cal_v2, None),
        "V4a_month": (cal_v2_month, None),
        "V4b_clip": (cal_v2, 0.995),
        "V4c_month_clip": (cal_v2_month, 0.995),
    }

    results = {}
    preds_store = {}
    preds_np = {}
    for vname, (cal, clip_q) in variants.items():
        print("=== {} (clip_q={}) ===".format(vname, clip_q), flush=True)
        Xtr_f = add_calendar(Xtr_n, cal[tr_mask])
        Xva_f = add_calendar(Xva_n, cal[va_mask])
        Xte_f = add_calendar(Xte_n, cal[te_mask])
        preds, p_clfs = [], []
        per_seed = []
        for s in SEEDS:
            pred, p_clf = fit_zi_seed(Xtr_f, Xva_f, Xte_f, CFG, s, clip_q=clip_q)
            preds.append(pred)
            p_clfs.append(p_clf)
            m = metrics(yte, pred, p_clf)
            per_seed.append({"seed": s, **m})
            print("  Seed {}: R2={:.4f} RMSE={:,.0f} MAE={:,.0f} ACC={:.4f} AUC={:.4f}".format(
                s, m["R2"], m["RMSE"], m["MAE"], m["ACC"], m["AUC"]), flush=True)

        pred_ens = np.mean(preds, axis=0)
        p_clf_ens = np.mean(p_clfs, axis=0)
        m_ens = metrics(yte, pred_ens, p_clf_ens)
        print("  Ensemble: R2={:.4f} RMSE={:,.0f} MAE={:,.0f} ACC={:.4f} AUC={:.4f}".format(
            m_ens["R2"], m_ens["RMSE"], m_ens["MAE"], m_ens["ACC"], m_ens["AUC"]), flush=True)

        e_this = yte.ravel() - pred_ens.ravel()
        ci = bootstrap_ci_rmse(e_this)
        print("  Bootstrap CI RMSE: [{:,.0f}, {:,.0f}]".format(ci[0], ci[1]), flush=True)

        plb_idx = REGIONS.index("Palembang")
        resid_plb = yte[:, plb_idx] - pred_ens[:, plb_idx]
        rmse_plb = np.sqrt(np.mean(resid_plb ** 2))
        print("  Palembang RMSE: {:,.0f}".format(rmse_plb), flush=True)

        entry = {
            "per_seed": per_seed,
            "ensemble": m_ens,
            "bootstrap_ci_rmse": [float(ci[0]), float(ci[1])],
            "palembang_rmse": float(rmse_plb),
        }
        if vname != "V3_ref":
            e_ref = yte.ravel() - preds_np["V3_ref"].ravel()
            dm, p = dm_test(e_this, e_ref)
            entry["dm_vs_v3ref"] = {"DM": dm, "p": p}
            print("  DM vs V3_ref: DM={:.3f}, p={:.4f}".format(dm, p), flush=True)
        results[vname] = entry
        preds_np[vname] = pred_ens
        preds_store[vname] = pred_ens.tolist()
        print("", flush=True)

    out = {
        "config": CFG,
        "seeds": SEEDS,
        "clip_q": 0.995,
        "note": "V4a: month-of-year one-hot (siklus fiskal E11); V4b: target clipping amount p99.5 (outlier E8/E9); V4c: kombinasi.",
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "pred_ens"} for k, v in results.items()},
        "preds": preds_store,
        "y_test": yte.tolist(),
        "regions": REGIONS,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("Selesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()