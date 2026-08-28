# -*- coding: utf-8 -*-
"""
Eksperimen atribut node GCN (jumlah RS) pada konfigurasi HybridTuned.

Latar belakang: exp_rolling_nodefeat.py (Hybrid base 128/128/do0) menunjukkan
jumlah RS sebagai atribut node GCN secara signifikan memperbaiki R2 (DM p=0.004).
Pertanyaan: apakah efek ini bertahan pada konfigurasi HybridTuned (128/64/do0.2)
yang dipakai di Tabel 7 naskah?

Protokol: pipeline identik experiment_zi_geocoded.py, distance graph k=3,
paired init (manual_seed sebelum pembuatan model), seeds 42/7/123.

Varian:
  V2_ref   : V2 replikasi (dow one-hot + holiday + fase pandemi), F=12, acuan DM.
  T1_hosp  : V2 + jumlah RS sebagai atribut node GCN (F=12, node_dim=1).
  T2_roll  : V2 + rolling mean/std 7&30 hari + jumlah RS node (F=16, node_dim=1).
  T3_full  : V2 + rolling + jumlah RS + log luas daerah (F=16, node_dim=2).

Output: results/exp_nodeattr_tuned.json
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from scipy import stats

torch.set_num_threads(4)

DATA = r"E:\Download\Jurnal\data\view_penjualan_detail data hingga oktober.xlsx"
GEO = r"E:\Download\Jurnal\data\customers_geocoded_final.csv"
OUT = r"E:\Download\Jurnal\results\exp_nodeattr_tuned.json"

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

RS_COUNT = {
    "Banyuasin": 6, "Empat Lawang": 2, "Lahat": 4,
    "Lubuk Linggau": 6, "Muara Enim": 7, "Musi Banyuasin": 3,
    "Musi Rawas": 4, "Ogan Ilir": 2, "Ogan Komering Ilir": 2,
    "Ogan Komering Ulu": 8, "Ogan Komering Ulu Selatan": 1,
    "Ogan Komering Ulu Timur": 4, "Pagar Alam": 1, "Palembang": 33,
    "Penukal Abab Lematang Ilir": 2, "Prabumulih": 4,
}
AREA_KM2 = {
    "Banyuasin": 12258.59, "Empat Lawang": 2234.10, "Lahat": 4333.07,
    "Lubuk Linggau": 367.73, "Muara Enim": 6763.91, "Musi Banyuasin": 14550.79,
    "Musi Rawas": 12060.39, "Ogan Ilir": 2302.86,
    "Ogan Komering Ilir": 17075.71, "Ogan Komering Ulu": 3774.50,
    "Ogan Komering Ulu Selatan": 4369.25, "Ogan Komering Ulu Timur": 3412.72,
    "Pagar Alam": 625.91, "Palembang": 352.52,
    "Penukal Abab Lematang Ilir": 1842.56, "Prabumulih": 447.31,
}

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


# ---------------------------------------------------------------------------
# Pipeline data
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

# Rolling features (kausal)
log_sales = np.log1p(panel)
roll_feats = []
for win in (7, 30):
    r_mean = log_sales.rolling(win, min_periods=1).mean()
    r_std = log_sales.rolling(win, min_periods=2).std().fillna(0.0)
    roll_feats += [r_mean.values, r_std.values]
ROLL = np.stack(roll_feats, axis=-1)  # (n_days, N, 4)

WINDOW = 30
X_base, y_all, t_all = [], [], []
for i in range(n_days - WINDOW):
    sales = panel.iloc[i:i + WINDOW].values
    flags = np.tile(covid_flag[i:i + WINDOW][:, None], (1, N))
    X_base.append(np.stack([sales, flags], axis=-1))  # (T, N, 2)
    y_all.append(panel.iloc[i + WINDOW].values)
    t_all.append(panel.index[i + WINDOW])
X_base = np.array(X_base).transpose(0, 2, 1, 3)  # (samples, N, T, 2)
y_all = np.array(y_all)

tr_mask = np.array([m.start_time <= pd.Timestamp("2023-12-31") for m in t_all])
va_mask = np.array([(m.start_time > pd.Timestamp("2023-12-31")) & (m.start_time <= pd.Timestamp("2024-12-31")) for m in t_all])
te_mask = np.array([m.start_time > pd.Timestamp("2024-12-31") for m in t_all])

# Normalisasi train-only (sales channel only; COVID = biner)
mu0 = np.log1p(X_base[tr_mask][:, :, :, 0]).mean()
sd0 = np.log1p(X_base[tr_mask][:, :, :, 0]).std()
X_base_n = X_base.copy()
X_base_n[:, :, :, 0] = (np.log1p(X_base[:, :, :, 0]) - mu0) / sd0

# Rolling: normalisasi train-only per channel (day-level mask, bukan sample-level)
day_tr_mask = np.array([m.start_time <= pd.Timestamp("2023-12-31") for m in panel.index])
ROLL_n = ROLL.copy()
for ch in range(4):
    mu_r = ROLL[day_tr_mask][:, :, ch].mean()
    sd_r = ROLL[day_tr_mask][:, :, ch].std()
    if sd_r == 0:
        sd_r = 1.0
    ROLL_n[:, :, ch] = (ROLL[:, :, ch] - mu_r) / sd_r

ytr, yva, yte = y_all[tr_mask], y_all[va_mask], y_all[te_mask]
ybin_tr = (ytr > 0).astype(np.float32)
ybin_va = (yva > 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Adjacency & node attributes
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

# Node attributes: z-score antar region
rs_vec = np.array([RS_COUNT[r] for r in REGIONS], dtype=np.float64)
area_vec = np.log(np.array([AREA_KM2[r] for r in REGIONS], dtype=np.float64))
NODE_RS = torch.tensor(((rs_vec - rs_vec.mean()) / rs_vec.std()).reshape(-1, 1),
                       dtype=torch.float32)
NODE_RS_AREA = torch.tensor(np.stack([
    (rs_vec - rs_vec.mean()) / rs_vec.std(),
    (area_vec - area_vec.mean()) / area_vec.std(),
], axis=-1), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model: HybridTuned dengan opsi node attributes
# ---------------------------------------------------------------------------
class HybridModel(nn.Module):
    """HybridTuned LSTM 128 / GCN 64 / dropout 0.2 dengan node attr opsional."""

    def __init__(self, in_features=2, node_attr=None):
        super().__init__()
        self.node_attr = node_attr
        gnn_in = 128 + (node_attr.shape[1] if node_attr is not None else 0)
        self.lstm = nn.LSTM(in_features, 128, batch_first=True)
        self.gcn = nn.Linear(gnn_in, 64)
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(128 + 64, 1)

    def forward(self, x):
        B, Nn, T, F = x.shape
        h, _ = self.lstm(x.reshape(B * Nn, T, F))
        h_lstm = h[:, -1, :].view(B, Nn, -1)
        if self.node_attr is not None:
            attr = self.node_attr.unsqueeze(0).expand(B, -1, -1)
            gcn_in = torch.cat([h_lstm, attr], dim=-1)
        else:
            gcn_in = h_lstm
        h_gcn = torch.relu(self.gcn(gcn_in))
        h_gcn = A_norm_t @ h_gcn
        fused = torch.cat([h_lstm, h_gcn], dim=-1)
        return self.fc(self.drop(fused)).view(B, Nn)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------
def train_model(model, X, y, binary, lr, batch, epochs, seed, Xva_, yva_,
                mask=None, mask_va=None, patience=8):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss() if binary else nn.MSELoss()
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    Xv = torch.tensor(Xva_, dtype=torch.float32)
    yv = torch.tensor(yva_, dtype=torch.float32)
    best_loss = float("inf")
    best_state = None
    bad = 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
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
    denom = np.sqrt(d.var(ddof=1) / T)
    if denom == 0 or not np.isfinite(denom):
        return float("nan"), float("nan")
    dm = float(d.mean() / denom)
    p = float(2 * (1 - stats.norm.cdf(abs(dm))))
    return dm, p


def build_calendar_features(t_list):
    """V2: dow one-hot (7) + holiday (1) + fase pandemi (2) -> (n_samples, 10)."""
    feats = []
    for m in t_list:
        t = m.start_time
        dow = t.dayofweek
        is_hol = 1.0 if t in HOL else 0.0
        if t <= pd.Timestamp("2020-12-31"):
            fase = [1.0, 0.0]
        elif t <= pd.Timestamp("2022-12-31"):
            fase = [0.0, 1.0]
        else:
            fase = [0.0, 0.0]
        onehot = [1.0 if d == dow else 0.0 for d in range(7)]
        feats.append(onehot + [is_hol] + fase)
    return np.array(feats, dtype=np.float32)


def add_calendar(X_np, cal):
    """Tile fitur kalender hari target ke semua region & timestep."""
    B, Nn, T, F = X_np.shape
    out = np.zeros((B, Nn, T, F + cal.shape[1]), dtype=np.float32)
    out[:, :, :, :F] = X_np
    for b in range(B):
        out[b, :, :, F:] = np.broadcast_to(cal[b], (Nn, T, cal.shape[1]))
    return out


SEEDS = [42, 7, 123]
cal = build_calendar_features(t_all)


def fit_zi_seed(seed, Xtr, Xva, Xte, in_features, node_attr=None):
    """Zero-inflated: classifier + amount model, paired init."""
    torch.manual_seed(seed)
    clf = HybridModel(in_features=in_features, node_attr=node_attr)
    train_model(clf, Xtr, ybin_tr, True, 0.001, 64, 60, seed, Xva, ybin_va)
    p_clf = 1 / (1 + np.exp(-predict(clf, Xte)))

    torch.manual_seed(seed)
    amt = HybridModel(in_features=in_features, node_attr=node_attr)
    train_model(amt, Xtr, np.log1p(ytr), False, 0.001, 64, 60, seed,
                Xva, np.log1p(yva), mask=(ytr > 0), mask_va=(yva > 0))
    p_amt = np.clip(np.expm1(predict(amt, Xte)), 0, None)

    pred = (p_clf > 0.5).astype(float) * p_amt
    return pred, p_clf


def run_variant(name, Xtr, Xva, Xte, in_features, node_attr=None):
    print(f"=== {name} ===", flush=True)
    preds, p_clfs, per_seed = [], [], []
    for s in SEEDS:
        pred, p_clf = fit_zi_seed(s, Xtr, Xva, Xte, in_features, node_attr)
        preds.append(pred)
        p_clfs.append(p_clf)
        m = metrics(yte, pred, p_clf)
        per_seed.append({"seed": s, **m})
        print(f"  Seed {s}: R2={m['R2']:.4f} RMSE={m['RMSE']:,.0f} "
              f"MAE={m['MAE']:,.0f} ACC={m['ACC']:.4f} AUC={m['AUC']:.4f}",
              flush=True)
    pred_ens = np.mean(preds, axis=0)
    p_clf_ens = np.mean(p_clfs, axis=0)
    m_ens = metrics(yte, pred_ens, p_clf_ens)
    plb = REGIONS.index("Palembang")
    rmse_plb = float(np.sqrt(np.mean((yte[:, plb] - pred_ens[:, plb]) ** 2)))
    print(f"  Ensemble: R2={m_ens['R2']:.4f} RMSE={m_ens['RMSE']:,.0f} "
          f"Palembang RMSE={rmse_plb:,.0f}", flush=True)
    return {
        "per_seed": per_seed,
        "ensemble": m_ens,
        "palembang_rmse": rmse_plb,
        "_pred_ens": pred_ens,
    }


def main():
    results = {}

    # --- V2_ref (acuan DM) ---
    # Input: sales+covid (F=2) + calendar (F=10) = F=12, no rolling, no node attr
    X_v2 = add_calendar(X_base_n, cal)
    Xtr, Xva, Xte = X_v2[tr_mask], X_v2[va_mask], X_v2[te_mask]
    res = run_variant("V2_ref", Xtr, Xva, Xte, in_features=12)
    e_ref = yte.ravel() - res["_pred_ens"].ravel()
    results["V2_ref"] = {k: v for k, v in res.items() if k != "_pred_ens"}

    # --- T1: V2 + hospital node attr (no rolling) ---
    # Input: sales+covid + calendar = F=12, node_dim=1
    res = run_variant("T1_hosp", Xtr, Xva, Xte, in_features=12, node_attr=NODE_RS)
    e_t1 = yte.ravel() - res["_pred_ens"].ravel()
    dm, p = dm_test(e_t1, e_ref)
    print(f"  DM vs V2_ref: DM={dm:.3f}, p={p:.4f}", flush=True)
    entry = {k: v for k, v in res.items() if k != "_pred_ens"}
    entry["dm_vs_V2"] = {"DM": dm, "p": p}
    results["T1_hosp"] = entry

    # --- T2: V2 + rolling + hospital node (F=16, node_dim=1) ---
    # ROLL_n is (n_days, N, 4) — build windowed version to match X_base_n
    ROLL_win = np.stack([ROLL_n[i:i + WINDOW] for i in range(n_days - WINDOW)])  # (samples, WINDOW, N, 4)
    ROLL_win = ROLL_win.transpose(0, 2, 1, 3)  # (samples, N, WINDOW, 4)
    X_roll = add_calendar(
        np.concatenate([X_base_n[:, :, :, :2], ROLL_win], axis=-1), cal)
    Xtr_r, Xva_r, Xte_r = X_roll[tr_mask], X_roll[va_mask], X_roll[te_mask]
    res = run_variant("T2_roll_hosp", Xtr_r, Xva_r, Xte_r, in_features=16, node_attr=NODE_RS)
    e_t2 = yte.ravel() - res["_pred_ens"].ravel()
    dm, p = dm_test(e_t2, e_ref)
    print(f"  DM vs V2_ref: DM={dm:.3f}, p={p:.4f}", flush=True)
    entry = {k: v for k, v in res.items() if k != "_pred_ens"}
    entry["dm_vs_V2"] = {"DM": dm, "p": p}
    results["T2_roll_hosp"] = entry

    # --- T3: V2 + rolling + hospital + area node (F=16, node_dim=2) ---
    res = run_variant("T3_full", Xtr_r, Xva_r, Xte_r, in_features=16, node_attr=NODE_RS_AREA)
    e_t3 = yte.ravel() - res["_pred_ens"].ravel()
    dm, p = dm_test(e_t3, e_ref)
    print(f"  DM vs V2_ref: DM={dm:.3f}, p={p:.4f}", flush=True)
    entry = {k: v for k, v in res.items() if k != "_pred_ens"}
    entry["dm_vs_V2"] = {"DM": dm, "p": p}
    results["T3_full"] = entry

    out = {
        "config": {"hidden_lstm": 128, "hidden_gnn": 64, "dropout": 0.2,
                    "lr": 0.001, "batch": 64},
        "seeds": SEEDS,
        "graph": "distance k=3",
        "node_attrs": {
            "rs_count": {r: int(RS_COUNT[r]) for r in REGIONS},
            "area_km2": {r: AREA_KM2[r] for r in REGIONS},
        },
        "results": results,
    }
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    print(f"\nSelesai. Disimpan ke {OUT}", flush=True)


if __name__ == "__main__":
    main()
