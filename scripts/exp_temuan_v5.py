# -*- coding: utf-8 -*-
"""V5: kanal komposisi produk (temuan pemetaan produk -> pemicu, docs/temuan.md s6).

Temuan (grain terkoreksi, total nilai Rp99,97 M): APD/higienitas 18,3%,
bedah-OK 12,9%, DBD 12,4%, spuit injeksi 12,3%, respiratorium 10,2%,
operasional rutin 10,2%, COVID 6,3%, imunisasi murni 1,0%. Spike empiris:
Delta Jul-Agu 2021 (APD+tes+oksigen; ADS x5,1), BIAN Agu 2022 (handscoon
x17,4); karhutla NEGATIF; DBD terkontaminasi seasonality.

V5 BEDA dari V4 (exp_temuan.py: month-of-year + target clipping p99.5):
dekomposisi total penjualan per region-hari menjadi kanal produk (covid/imm/rutin).

Varian:
  V3_ref        : replikasi V3 (baseline, DM test)
  V5a_prodval   : V3 + 3 kanal nilai produk (covid, imunisasi, rutin)
  V5b_prodshare : V3 + 2 kanal share komposisi (event-driven, rutin)

Pipeline identik exp_temuan.py (16 region, anti-leakage, log1p train-only,
zero-inflated clf+amt, 3 seeds ensemble). Evaluasi skala asli.
Output: results/exp_temuan_v5.json
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
torch.set_num_threads(4)

from exp_temuan import (df as df_inv, panel, t_all, tr_mask, va_mask, te_mask,
                        Xtr_n, Xva_n, Xte_n, ytr, yva, yte, ybin_tr, ybin_va, ybin_te,
                        REGIONS, N, build_calendar_features, add_calendar,
                        HybridModel, train, predict, metrics, dm_test,
                        bootstrap_ci_rmse, CFG, SEEDS)
from product_driver_mapping import classify_row

OUT = r"E:\Download\Jurnal\results\exp_temuan_v5.json"
PRED_DIR = r"E:\Download\Jurnal\results"
WINDOW = 30
CHANNELS = ["covid", "imm", "routine"]
DATA_XLSX = r"E:\Download\Jurnal\data\view_penjualan_detail data hingga oktober.xlsx"


def channel_of(groups):
    """classify_row mengembalikan LIST grup; prioritas kanal: imm > covid > routine."""
    gs = set(groups)
    if "IMUNISASI" in gs:
        return "imm"
    if gs & {"PANDEMI_COVID", "KARHUTLA"}:
        return "covid"
    return "routine"


# --- Kanal komposisi produk: atribusi dominan per faktur ---
# jual_total_fak berulang di tiap baris item satu faktur, jadi kanal TIDAK boleh
# dipetakan dari nama item pertama faktur. Klasifikasi dilakukan level-item
# (nama+spesifikasi via classify_row), lalu tiap faktur masuk SATU kanal =
# kanal dengan nilai item dominan (kolom `jumlah`).
# covid+imm+routine == panel total persis (setiap faktur tepat satu kanal).
items = pd.read_excel(DATA_XLSX)
items = items.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
if "barang_spesifikasi" not in items.columns:
    items["barang_spesifikasi"] = ""
items["_ch"] = [channel_of(classify_row(a, b))
                for a, b in zip(items["barang_nama"], items["barang_spesifikasi"].fillna(""))]
if "jumlah" in items.columns:
    items["_val"] = pd.to_numeric(items["jumlah"], errors="coerce").fillna(0.0)
else:
    items["_val"] = 0.0

cv = items.groupby(["d_jual_nofak", "_ch"])["_val"].sum().unstack(fill_value=0.0)
for c in CHANNELS:
    if c not in cv.columns:
        cv[c] = 0.0
ch_dom = cv[CHANNELS].idxmax(axis=1).rename("ch")
ch_dom[cv[CHANNELS].sum(axis=1) <= 0] = "routine"

df_inv = df_inv.copy()
df_inv["ch"] = df_inv["d_jual_nofak"].map(ch_dom).fillna("routine")

chan_panels = {}
for ch in CHANNELS:
    sub = df_inv[df_inv["ch"] == ch]
    p = sub.pivot_table(index="period", columns="region", values="jual_total", aggfunc="sum")
    chan_panels[ch] = p.reindex(index=panel.index, columns=REGIONS).fillna(0.0).sort_index()

# Window kanal -> (B, N, T)
def window_panel(p):
    arr = np.array([p.iloc[i:i + WINDOW].values for i in range(len(p) - WINDOW)])
    return arr.transpose(0, 2, 1)

chan_w = {ch: window_panel(p) for ch, p in chan_panels.items()}

# Normalisasi log1p + z-score per kanal (train only)
chan_n = {}
for ch in CHANNELS:
    w = chan_w[ch]
    mu = np.log1p(w[tr_mask]).mean()
    sd = np.log1p(w[tr_mask]).std()
    chan_n[ch] = (np.log1p(w) - mu) / sd

# Share komposisi (V5b): event = covid+imm, rutin; 0 pada hari tanpa penjualan
total_w = window_panel(panel)
safe = np.maximum(total_w, 1e-12)
event_share = np.where(total_w > 0, (chan_w["covid"] + chan_w["imm"]) / safe, 0.0)
routine_share = np.where(total_w > 0, chan_w["routine"] / safe, 0.0)

cal_v2 = build_calendar_features(t_all, "V2")


def build_X(extra_tr, extra_va, extra_te):
    Xtr = np.concatenate([Xtr_n, extra_tr], axis=-1)
    Xva = np.concatenate([Xva_n, extra_va], axis=-1)
    Xte = np.concatenate([Xte_n, extra_te], axis=-1)
    return (add_calendar(Xtr, cal_v2[tr_mask]),
            add_calendar(Xva, cal_v2[va_mask]),
            add_calendar(Xte, cal_v2[te_mask]))


def fit_zi_seed(Xtr_f, Xva_f, Xte_f, cfg, seed):
    clf = HybridModel(hidden_lstm=cfg["hidden_lstm"], hidden_gnn=cfg["hidden_gnn"],
                      dropout=cfg["dropout"], in_features=Xtr_f.shape[-1])
    _, _ = train(clf, Xtr_f, ybin_tr, True, cfg["lr"], cfg["batch"], 60, seed, Xva_f, ybin_va)
    p_clf = 1 / (1 + np.exp(-predict(clf, Xte_f)))
    amt = HybridModel(hidden_lstm=cfg["hidden_lstm"], hidden_gnn=cfg["hidden_gnn"],
                      dropout=cfg["dropout"], in_features=Xtr_f.shape[-1])
    _, _ = train(amt, Xtr_f, np.log1p(ytr), False, cfg["lr"], cfg["batch"], 60, seed,
                  Xva_f, np.log1p(yva), mask=(ytr > 0), mask_va=(yva > 0))
    p_amt = np.clip(np.expm1(predict(amt, Xte_f)), 0, None)
    return (p_clf > 0.5).astype(float) * p_amt, p_clf


def save_out(results, preds_store):
    out = {
        "config": CFG, "seeds": SEEDS,
        "note": "V5a: +3 kanal nilai produk (covid/imm/rutin); V5b: +2 kanal share komposisi. Atribusi kanal dominan per faktur (classify_row nama+spesifikasi, bobot nilai item jumlah).",
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "pred_ens"} for k, v in results.items()},
        "preds": preds_store, "y_test": yte.tolist(), "regions": REGIONS,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)


def main():
    prod_val = np.stack([chan_n["covid"], chan_n["imm"], chan_n["routine"]], axis=-1)
    prod_share = np.stack([event_share, routine_share], axis=-1)

    variants = {
        "V3_ref": build_X(np.zeros((int(tr_mask.sum()), N, WINDOW, 0), dtype=np.float32),
                          np.zeros((int(va_mask.sum()), N, WINDOW, 0), dtype=np.float32),
                          np.zeros((int(te_mask.sum()), N, WINDOW, 0), dtype=np.float32)),
        "V5a_prodval": build_X(prod_val[tr_mask], prod_val[va_mask], prod_val[te_mask]),
        "V5b_prodshare": build_X(prod_share[tr_mask], prod_share[va_mask], prod_share[te_mask]),
    }

    results, preds_store, preds_np = {}, {}, {}
    for vname, (Xtr_f, Xva_f, Xte_f) in variants.items():
        print("=== {} ({} kanal) ===".format(vname, Xtr_f.shape[-1]), flush=True)
        preds, p_clfs, per_seed = [], [], []
        for s in SEEDS:
            pred_path = os.path.join(PRED_DIR, "v5_{}_pred_seed{}.npy".format(vname, s))
            pclf_path = os.path.join(PRED_DIR, "v5_{}_pclf_seed{}.npy".format(vname, s))
            if os.path.exists(pred_path) and os.path.exists(pclf_path):
                pred = np.load(pred_path)
                p_clf = np.load(pclf_path)
                print("  Seed {}: dimuat dari cache".format(s), flush=True)
            else:
                pred, p_clf = fit_zi_seed(Xtr_f, Xva_f, Xte_f, CFG, s)
                np.save(pred_path, pred)
                np.save(pclf_path, p_clf)
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

        plb = REGIONS.index("Palembang")
        rmse_plb = np.sqrt(np.mean((yte[:, plb] - pred_ens[:, plb]) ** 2))
        print("  Palembang RMSE: {:,.0f}".format(rmse_plb), flush=True)

        entry = {"per_seed": per_seed, "ensemble": m_ens,
                 "bootstrap_ci_rmse": [float(ci[0]), float(ci[1])],
                 "palembang_rmse": float(rmse_plb)}
        if vname != "V3_ref":
            e_ref = yte.ravel() - preds_np["V3_ref"].ravel()
            dm, p = dm_test(e_this, e_ref)
            entry["dm_vs_v3ref"] = {"DM": dm, "p": p}
            print("  DM vs V3_ref: DM={:.3f}, p={:.4f}".format(dm, p), flush=True)
        results[vname] = entry
        preds_store[vname] = pred_ens.tolist()
        preds_np[vname] = pred_ens
        # Simpan JSON parsial tiap selesai satu varian (anti kehilangan saat kill)
        save_out(results, preds_store)
        print("", flush=True)

    save_out(results, preds_store)
    print("Selesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()