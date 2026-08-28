# -*- coding: utf-8 -*-
"""
Eksperimen fitur rolling + atribut node (jumlah RS, luas daerah).

Latar belakang (diskusi 26 Agu 2026): arsitektur baru bukan jalur tepat
menaikkan R2; informasi tambahan lebih menjanjikan. Dua kandidat murah:
  1. Fitur rolling (mean/std 7 & 30 hari) sebagai kanal input tambahan.
  2. Atribut node statis: jumlah RS (BPS 2025, korelasi silang r=0.669)
     dan luas daerah (BPS 2025; kontrol - ekspektasi rendah karena
     korelasi tidak signifikan r=-0.341 p=0.196 pada analisis BPS).

Protokol IDENTIK exp_features.py (pipeline experiment_zi_geocoded.py,
config Hybrid base 128/128/do0/lr1e-3/bs64, seeds 42/7/123).
Varian:
  W0_v2_ref      : V2 replikasi (dow one-hot + holiday + fase, F=12).
                   Sanity check vs hasil lama (R2 ens 0.0612) + acuan DM.
  W1_rolling     : V2 + rolling 4 kanal (F=16).
  W2_rs_node     : W1 + jumlah RS sebagai atribut node GCN.
  W3_rs_area_node: W1 + jumlah RS + log luas sebagai atribut node GCN.

Rolling dihitung kausal (hanya hari dalam window, semua < hari target),
normalisasi train-only. Atribut node distandardisasi antar 16 region.
Output: results/exp_rolling_nodefeat.json
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
MAIN_JSON = r"E:\Download\Jurnal\results\experiment_results_zi_geocoded.json"
OUT = r"E:\Download\Jurnal\results\exp_rolling_nodefeat.json"

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

# Jumlah RS (umum + khusus, BPS 2025; "-" = 0; Musi Rawas Utara digabung)
RS_COUNT = {
    "Banyuasin": 6 + 0, "Empat Lawang": 2 + 0, "Lahat": 3 + 1,
    "Lubuk Linggau": 4 + 2, "Muara Enim": 7 + 0, "Musi Banyuasin": 3 + 0,
    "Musi Rawas": 3 + 1, "Ogan Ilir": 2 + 0, "Ogan Komering Ilir": 2 + 0,
    "Ogan Komering Ulu": 3 + 5, "Ogan Komering Ulu Selatan": 1 + 0,
    "Ogan Komering Ulu Timur": 4 + 0, "Pagar Alam": 1 + 0, "Palembang": 24 + 9,
    "Penukal Abab Lematang Ilir": 2 + 0, "Prabumulih": 4 + 0,
}
# Luas wilayah km2 (BPS 2025; Musi Rawas Utara 5937.80 digabung ke Musi Rawas)
AREA_KM2 = {
    "Banyuasin": 12258.59, "Empat Lawang": 2234.10, "Lahat": 4333.07,
    "Lubuk Linggau": 367.73, "Muara Enim": 6763.91, "Musi Banyuasin": 14550.79,
    "Musi Rawas": 6122.59 + 5937.80, "Ogan Ilir": 2302.86,
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
# Pipeline data (identik exp_features.py)
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

# Rolling features: kausal (window hanya memuat hari < target), skala log1p
log_sales = np.log1p(panel)
roll_feats = []
for win in (7, 30):
    r_mean = log_sales.rolling(win, min_periods=1).mean()
    r_std = log_sales.rolling(win, min_periods=2).std().fillna(0.0)
    roll_feats += [r_mean.values, r_std.values]
ROLL = np.stack(roll_feats, axis=-1)   # (n_days, N, 4)

WINDOW = 30
X_all, y_all, t_all = [], [], []
for i in range(n_days - WINDOW):
    sales = panel.iloc[i:i + WINDOW].values
    flags = np.tile(covid_flag[i:i + WINDOW][:, None], (1, N))
    rolls = ROLL[i:i + WINDOW]                      # (T, N, 4)
    feat = np.concatenate([
        np.stack([sales, flags], axis=-1),          # (T, N, 2)
        rolls,                                      # (T, N, 4)
    ], axis=-1)                                     # (T, N, 6)
    X_all.append(feat.transpose(1, 0, 2))           # (N, T, 6)
    y_all.append(panel.iloc[i + WINDOW].values)
    t_all.append(panel.index[i + WINDOW])
X_all = np.array(X_all)                             # (samples, N, T, 6)
y_all = np.array(y_all)

tr_mask = np.array([m.start_time <= pd.Timestamp("2023-12-31") for m in t_all])
va_mask = np.array([(m.start_time > pd.Timestamp("2023-12-31")) & (m.start_time <= pd.Timestamp("2024-12-31")) for m in t_all])
te_mask = np.array([m.start_time > pd.Timestamp("2024-12-31") for m in t_all])

# Normalisasi train-only. Protokol asli: kanal sales di-log1p dan
# distandardisasi; kanal COVID TIDAK ditransformasi; kanal rolling
# (sudah skala log1p) distandardisasi.
X_all_n = X_all.copy()
norm_info = {}
ch0_mu = np.log1p(X_all[tr_mask][:, :, :, 0]).mean()
ch0_sd = np.log1p(X_all[tr_mask][:, :, :, 0]).std()
X_all_n[:, :, :, 0] = (np.log1p(X_all[:, :, :, 0]) - ch0_mu) / ch0_sd
norm_info["sales"] = [float(ch0_mu), float(ch0_sd)]
for ch in range(2, X_all.shape[-1]):
    mu = X_all[tr_mask][:, :, :, ch].mean()
    sd = X_all[tr_mask][:, :, :, ch].std()
    if sd == 0:
        sd = 1.0
    X_all_n[:, :, :, ch] = (X_all[:, :, :, ch] - mu) / sd
    norm_info[f"roll_{ch}"] = [float(mu), float(sd)]
# Kanal 1 (COVID) dibiarkan biner apa adanya.

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

# Atribut node: z-score antar region
rs_vec = np.array([RS_COUNT[r] for r in REGIONS], dtype=np.float64)
area_vec = np.log(np.array([AREA_KM2[r] for r in REGIONS], dtype=np.float64))
NODE_RS = torch.tensor(((rs_vec - rs_vec.mean()) / rs_vec.std()).reshape(-1, 1), dtype=torch.float32)
NODE_RS_AREA = torch.tensor(np.stack([
    (rs_vec - rs_vec.mean()) / rs_vec.std(),
    (area_vec - area_vec.mean()) / area_vec.std(),
], axis=-1), dtype=torch.float32)


class HybridModel(nn.Module):
    """Hybrid LSTM-GNN dengan opsi atribut node statis pada jalur GCN."""

    def __init__(self, hidden_lstm=128, hidden_gnn=128, dropout=0.0, in_features=2,
                 node_attr=None):
        super().__init__()
        self.node_attr = node_attr            # (N, A) atau None
        gnn_in = hidden_lstm + (node_attr.shape[1] if node_attr is not None else 0)
        self.lstm = nn.LSTM(in_features, hidden_lstm, batch_first=True)
        self.gcn = nn.Linear(gnn_in, hidden_gnn)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_lstm + hidden_gnn, 1)

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


CFG = {"hidden_lstm": 128, "hidden_gnn": 128, "dropout": 0.0, "lr": 1e-3, "batch": 64}
SEEDS = [42, 7, 123]
VARIANTS = [
    ("W0_v2_ref", None),
    ("W1_rolling", None),
    ("W2_rs_node", NODE_RS),
    ("W3_rs_area_node", NODE_RS_AREA),
]

cal2 = build_calendar_features(t_all)
def fit_zi_seed(seed, node_attr=None):
    clf = HybridModel(hidden_lstm=CFG["hidden_lstm"], hidden_gnn=CFG["hidden_gnn"],
                      dropout=CFG["dropout"], in_features=Xtr_f.shape[-1],
                      node_attr=node_attr)
    train(clf, Xtr_f, ybin_tr, True, CFG["lr"], CFG["batch"], 60, seed, Xva_f, ybin_va)
    p_clf = 1 / (1 + np.exp(-predict(clf, Xte_f)))
    amt = HybridModel(hidden_lstm=CFG["hidden_lstm"], hidden_gnn=CFG["hidden_gnn"],
                      dropout=CFG["dropout"], in_features=Xtr_f.shape[-1],
                      node_attr=node_attr)
    train(amt, Xtr_f, np.log1p(ytr), False, CFG["lr"], CFG["batch"], 60, seed,
          Xva_f, np.log1p(yva), mask=(ytr > 0), mask_va=(yva > 0))
    p_amt = np.expm1(predict(amt, Xte_f))
    p_amt = np.clip(p_amt, 0, None)
    pred = (p_clf > 0.5).astype(float) * p_amt
    return pred, p_clf


def run_variant(name, node_attr=None):
    print(f"=== {name} ===", flush=True)
    preds, p_clfs, per_seed = [], [], []
    for s in SEEDS:
        pred, p_clf = fit_zi_seed(s, node_attr)
        preds.append(pred)
        p_clfs.append(p_clf)
        m = metrics(yte, pred, p_clf)
        per_seed.append({"seed": s, **m})
        print(f"  Seed {s}: R2={m['R2']:.4f} RMSE={m['RMSE']:,.0f} MAE={m['MAE']:,.0f} "
              f"ACC={m['ACC']:.4f} AUC={m['AUC']:.4f}", flush=True)
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
    global Xtr_f, Xva_f, Xte_f
    results = {}

    # W0: V2 replikasi (sales+covid + kalender, F=12, tanpa rolling)
    X_v2_only = add_calendar(X_all_n[:, :, :, :2], cal2)
    Xtr_f, Xva_f, Xte_f = X_v2_only[tr_mask], X_v2_only[va_mask], X_v2_only[te_mask]
    res = run_variant("W0_v2_ref")
    e_ref = yte.ravel() - res["_pred_ens"].ravel()
    results["W0_v2_ref"] = {k: v for k, v in res.items() if k != "_pred_ens"}

    # W1-W3: aktifkan kanal rolling (F=16)
    X_roll = add_calendar(X_all_n, cal2)
    Xtr_f, Xva_f, Xte_f = X_roll[tr_mask], X_roll[va_mask], X_roll[te_mask]
    for name, attr in VARIANTS[1:]:
        res = run_variant(name, attr)
        e_this = yte.ravel() - res["_pred_ens"].ravel()
        dm, p = dm_test(e_this, e_ref)
        print(f"  DM vs W0(V2): DM={dm:.3f}, p={p:.4f}", flush=True)
        entry = {k: v for k, v in res.items() if k != "_pred_ens"}
        entry["dm_vs_W0"] = {"DM": dm, "p": p}
        results[name] = entry

    # DM semua varian vs baseline Hybrid base eksperimen utama
    main_res = json.load(open(MAIN_JSON, encoding="utf-8"))
    pred_base = np.array(main_res["models"]["Hybrid"]["preds_mean"])
    e_base = yte.ravel() - pred_base.ravel()

    out = {"config": CFG, "seeds": SEEDS,
           "node_attrs": {"rs_count": {r: int(RS_COUNT[r]) for r in REGIONS},
                          "area_km2": {r: AREA_KM2[r] for r in REGIONS}},
           "norm_info": norm_info,
           "results": results}
    json.dump(out, open(OUT, "w", encoding="utf-8"), indent=1, default=str)
    print(f"Selesai. Disimpan ke {OUT}", flush=True)


if __name__ == "__main__":
    main()
