"""V3: random search config terbaik (hl=219 hg=107 do=0.17 lr=1.52e-3 bs=32)
+ fitur kalender V2 (dow one-hot + holiday + fase). 3 seeds (42, 7, 123).
Bandingkan dengan V2 base (R2 0.0612) dan random search tanpa fitur (R2 0.0582).
Resume: prediksi per seed disimpan ke results/v3_preds_seed{n}.npy, seed yang
sudah ada dilewati. Output: results/exp_features_v3.json
"""
import json
import os
import sys
import numpy as np
import torch
from exp_features import (Xtr_n, Xva_n, Xte_n, ytr, yva, yte, ybin_tr, ybin_va, ybin_te,
                          tr_mask, va_mask, te_mask, t_all, REGIONS,
                          build_calendar_features, add_calendar, HybridModel,
                          train, predict, metrics, dm_test, bootstrap_ci_rmse, OUT_JSON)

BASE = r"E:\Download\Jurnal"
OUT = os.path.join(BASE, "results", "exp_features_v3.json")
PRED_DIR = os.path.join(BASE, "results")
SEEDS = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [42, 7, 123]
CFG_RS = {"hidden_lstm": 219, "hidden_gnn": 107, "dropout": 0.1734,
          "lr": 1.52e-3, "batch": 32}

res = json.load(open(OUT_JSON))
pred_base = np.array(res["models"]["Hybrid"]["preds_mean"])
e_base = yte.ravel() - pred_base.ravel()

cal2 = build_calendar_features(t_all, "V2")
Xtr_f = add_calendar(Xtr_n, cal2[tr_mask])
Xva_f = add_calendar(Xva_n, cal2[va_mask])
Xte_f = add_calendar(Xte_n, cal2[te_mask])

print("V3: rs config + fitur kalender V2 (resume-aware)", flush=True)
preds, p_clfs = [], []
per_seed = []
for s in SEEDS:
    pred_path = os.path.join(PRED_DIR, "v3_preds_seed{}.npy".format(s))
    pclf_path = os.path.join(PRED_DIR, "v3_pclf_seed{}.npy".format(s))
    if os.path.exists(pred_path) and os.path.exists(pclf_path):
        pred = np.load(pred_path)
        p_clf = np.load(pclf_path)
        print("  Seed {}: dimuat dari cache".format(s), flush=True)
    else:
        clf = HybridModel(hidden_lstm=CFG_RS["hidden_lstm"], hidden_gnn=CFG_RS["hidden_gnn"],
                          dropout=CFG_RS["dropout"], in_features=Xtr_f.shape[-1])
        _, _ = train(clf, Xtr_f, ybin_tr, True, CFG_RS["lr"], CFG_RS["batch"], 60, s, Xva_f, ybin_va)
        p_clf = 1 / (1 + np.exp(-predict(clf, Xte_f)))
        amt = HybridModel(hidden_lstm=CFG_RS["hidden_lstm"], hidden_gnn=CFG_RS["hidden_gnn"],
                          dropout=CFG_RS["dropout"], in_features=Xtr_f.shape[-1])
        _, _ = train(amt, Xtr_f, np.log1p(ytr), False, CFG_RS["lr"], CFG_RS["batch"], 60, s,
                      Xva_f, np.log1p(yva), mask=(ytr > 0), mask_va=(yva > 0))
        p_amt = np.clip(np.expm1(predict(amt, Xte_f)), 0, None)
        pred = (p_clf > 0.5).astype(float) * p_amt
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
dm, p = dm_test(e_this, e_base)
print("  DM vs baseline: DM={:.3f}, p={:.4f}".format(dm, p), flush=True)
ci = bootstrap_ci_rmse(e_this)
print("  Bootstrap CI RMSE: [{:,.0f}, {:,.0f}]".format(ci[0], ci[1]), flush=True)

plb_idx = REGIONS.index("Palembang")
resid_plb = yte[:, plb_idx] - pred_ens[:, plb_idx]
rmse_plb = np.sqrt(np.mean(resid_plb ** 2))
print("  Palembang RMSE: {:,.0f} (baseline {:,.0f})".format(
    rmse_plb, np.sqrt(np.mean((yte[:, plb_idx] - pred_base[:, plb_idx]) ** 2))), flush=True)

out = {
    "config": CFG_RS,
    "seeds": SEEDS,
    "per_seed": per_seed,
    "ensemble": m_ens,
    "dm_vs_baseline": {"DM": dm, "p": p},
    "bootstrap_ci_rmse": [float(ci[0]), float(ci[1])],
    "palembang_rmse": float(rmse_plb),
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, default=str)
print("Selesai. Disimpan ke " + OUT)