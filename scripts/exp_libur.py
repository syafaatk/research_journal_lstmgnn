# -*- coding: utf-8 -*-
"""Uji dampak hari kerja kosong (54 tanggal tanpa transaksi, Sen-Jum non-libur)
ke model ZI+amount. Latar: docs/temuan.md s3.5 + results/tanggal_kosong_kerja.csv.

Varian:
  V3_ref     : baseline V3 (cache v5 dipakai ulang - input identik)
  V_eidnatal : +2 fitur kalender eks-ante: jendela Idul Fitri (D-7..D+7),
               jendela Natal-Tahun Baru (23 Des-2 Jan). SAH (bisa diketahui
               sebelumnya dari kalender).
  V_hard54   : +1 flag hardcoded 54 tanggal kosong. KEBOCORAN (tanggal
               didefinisikan dari penjualan nol itu sendiri) - hanya untuk
               menunjukkan batas atas informasi, BUKAN kandidat fitur.

Pipeline identik exp_temuan.py (16 region, anti-leakage, log1p train-only,
zero-inflated clf+amt, 3 seeds ensemble). Evaluasi skala asli.
Output: results/exp_libur.json
"""
import json
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
torch.set_num_threads(4)

from exp_temuan import (t_all, tr_mask, va_mask, te_mask,
                        Xtr_n, Xva_n, Xte_n,
                        ytr, yva, yte, ybin_tr, ybin_va, ybin_te,
                        REGIONS, N, build_calendar_features, add_calendar,
                        HybridModel, train, predict, metrics, dm_test,
                        bootstrap_ci_rmse, CFG, SEEDS)

OUT = r"E:\Download\Jurnal\results\exp_libur.json"
PRED_DIR = r"E:\Download\Jurnal\results"
WINDOW = 30

# --- Fitur kalender dasar V2 (dow one-hot + is_hol + fase) ---
cal_v2 = build_calendar_features(t_all, "V2")

# --- Flag eks-ante ---
EID_ANCHORS = pd.to_datetime(["2020-05-24", "2021-05-13", "2022-05-02",
                              "2023-04-22", "2024-04-10", "2025-03-31"])
tgt = pd.DatetimeIndex([m.start_time for m in t_all])

def eid_flag(t):
    return float(any(abs((t - a).days) <= 7 for a in EID_ANCHORS))

def natal_flag(t):
    if t.month == 12 and t.day >= 23:
        return 1.0
    if t.month == 1 and t.day <= 2:
        return 1.0
    return 0.0

f_eid = np.array([eid_flag(t) for t in tgt], dtype=np.float32).reshape(-1, 1)
f_nat = np.array([natal_flag(t) for t in tgt], dtype=np.float32).reshape(-1, 1)

# --- Flag hardcoded 54 tanggal (KEBOCORAN - jangan dipakai sebagai fitur) ---
kosong = pd.read_csv(r"E:\Download\Jurnal\results\tanggal_kosong_kerja.csv")
hard_set = set(pd.to_datetime(kosong["tanggal"]))
f_hard = np.array([1.0 if t in hard_set else 0.0 for t in tgt],
                  dtype=np.float32).reshape(-1, 1)

# Konteks: sebaran flag di tiap split
for name, f, m in [("eid", f_eid.ravel(), te_mask), ("natal", f_nat.ravel(), te_mask),
                   ("hard54", f_hard.ravel(), te_mask)]:
    print("Flag {} : train={} val={} test={}".format(
        name, int(f[tr_mask].sum()), int(f[va_mask].sum()), int(f[te_mask].sum())))


def make_inputs(cal_full):
    return (add_calendar(Xtr_n, cal_full[tr_mask]),
            add_calendar(Xva_n, cal_full[va_mask]),
            add_calendar(Xte_n, cal_full[te_mask]))


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
        "note": ("Uji hari kerja kosong. V_eidnatal: +jendela Idul Fitri D-7..D+7 & "
                 "Natal-Tahun Baru 23 Des-2 Jan (eks-ante, sah). V_hard54: +flag "
                 "hardcoded 54 tanggal kosong (KEBOCORAN, bukan kandidat fitur)."),
        "results": {k: {kk: vv for kk, vv in v.items() if kk != "pred_ens"} for k, v in results.items()},
        "preds": preds_store, "y_test": yte.tolist(), "regions": REGIONS,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)


def main():
    variants = {
        "V3_ref": make_inputs(cal_v2),
        "V_eidnatal": make_inputs(np.concatenate([cal_v2, f_eid, f_nat], axis=1)),
        "V_hard54": make_inputs(np.concatenate([cal_v2, f_hard], axis=1)),
    }

    results, preds_store, preds_np = {}, {}, {}
    for vname, (Xtr_f, Xva_f, Xte_f) in variants.items():
        print("=== {} ({} feat kalender) ===".format(vname, Xtr_f.shape[-1]), flush=True)
        preds, p_clfs, per_seed = [], [], []
        for s in SEEDS:
            if vname == "V3_ref":
                # input identik dengan v5 -> pakai cache v5 langsung
                pred_path = os.path.join(PRED_DIR, "v5_V3_ref_pred_seed{}.npy".format(s))
                pclf_path = os.path.join(PRED_DIR, "v5_V3_ref_pclf_seed{}.npy".format(s))
            else:
                pred_path = os.path.join(PRED_DIR, "libur_{}_pred_seed{}.npy".format(vname, s))
                pclf_path = os.path.join(PRED_DIR, "libur_{}_pclf_seed{}.npy".format(vname, s))
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
        entry = {"per_seed": per_seed, "ensemble": m_ens,
                 "bootstrap_ci_rmse": [float(ci[0]), float(ci[1])]}
        if vname != "V3_ref":
            e_ref = yte.ravel() - preds_np["V3_ref"].ravel()
            dm, p = dm_test(e_this, e_ref)
            entry["dm_vs_v3ref"] = {"DM": dm, "p": p}
            print("  DM vs V3_ref: DM={:.3f}, p={:.4f}".format(dm, p), flush=True)
        results[vname] = entry
        preds_store[vname] = pred_ens.tolist()
        preds_np[vname] = pred_ens
        save_out(results, preds_store)
        print("", flush=True)

    save_out(results, preds_store)
    print("Selesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()
