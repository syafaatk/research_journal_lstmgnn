# -*- coding: utf-8 -*-
"""
Generator angka final untuk revisi.md dan balasan_reviewer.md.
Membaca experiment_results_zi_geocoded.json (16 region, anti-leakage)
dan mencetak semua baris tabel + angka narasi dalam format markdown
siap tempel. Jalankan SETELAH eksperimen selesai dan dm_hybrid.py
ditambahkan (dm_vs_hybrid, bootstrap_ci_ensemble, per_region_hybrid,
residual_hybrid).

Usage: python generate_table_numbers.py [path_json]
"""
import json
import sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else r"E:\Download\Jurnal\results\experiment_results_zi_geocoded.json"
RS_EVAL = r"E:\Download\Jurnal\results\rs_best_eval.json"
d = json.load(open(OUT, encoding="utf-8"))
try:
    rs = json.load(open(RS_EVAL, encoding="utf-8"))
    HAS_RS = True
except FileNotFoundError:
    HAS_RS = False

def mean(x):
    return float(np.mean(x)) if isinstance(x, list) else float(x)

def f_int(x):
    return f"{x:,.0f}"

def f_r2(x):
    return f"{x:.4f}"

def f_acc(x):
    return f"{x:.4f}"

def f_auc(x):
    return f"{x:.3f}"

def f_dm(x):
    return f"{x:.3f}"

def f_p(x):
    return "< 0.001" if x < 0.001 else f"{x:.3f}"

def f_ci(lo, hi):
    return f"[{f_int(lo)}, {f_int(hi)}]"

MODELS = ["LSTM", "GNN", "Hybrid", "HybridTuned"]
LABELS = {
    "LSTM": "Standalone LSTM",
    "GNN": "Standalone GNN",
    "Hybrid": "Hybrid LSTM-GNN",
    "HybridTuned": "Hybrid LSTM-GNN Tuned",
}

print("=" * 70)
print("TABEL 3 - Performance Comparison (mean over seeds)")
print("=" * 70)
print("| Model | RMSE (IDR) | MAE (IDR) | R-squared | Classification Accuracy | AUC |")
print("|---|---|---|---|---|---|")
for m in MODELS:
    v = d["models"][m]
    print(f"| {LABELS[m]} | {f_int(mean(v['RMSE']))} | {f_int(mean(v['MAE']))} | "
          f"{f_r2(mean(v['R2']))} | {f_acc(mean(v['ACC']))} | {f_auc(mean(v['AUC']))} |")
if HAS_RS:
    rs_m = rs["mean"]
    rs_e = rs["ensemble"]
    print(f"| Hybrid LSTM-GNN (random search) | {f_int(rs_m['RMSE'])} | {f_int(rs_m['MAE'])} | "
          f"{f_r2(rs_m['R2'])} | {f_acc(rs_e['ACC'])} | {f_auc(rs_e['AUC'])} |")

print()
print("=" * 70)
print("TABEL 5 - Mean +/- Std, DM vs Hybrid (ensemble), Bootstrap CI (ensemble)")
print("=" * 70)
print("| Model | RMSE (mean +/- std, IDR) | MAE (mean +/- std, IDR) | DM vs Hybrid | p-value | Bootstrap 95% CI (RMSE, IDR) |")
print("|---|---|---|---|---|---|")
for m in MODELS:
    v = d["models"][m]
    rmses = v["RMSE"] if isinstance(v["RMSE"], list) else [v["RMSE"]]
    maes = v["MAE"] if isinstance(v["MAE"], list) else [v["MAE"]]
    rmse_m, rmse_s = np.mean(rmses), np.std(rmses)
    mae_m, mae_s = np.mean(maes), np.std(maes)
    if m == "Hybrid":
        dm_cell, p_cell = "-", "-"
    else:
        dmv = d["dm_vs_hybrid"][m]
        dm_cell, p_cell = f_dm(dmv["DM"]), f_p(dmv["p"])
    lo, hi = d["bootstrap_ci_ensemble"][m]
    print(f"| {LABELS[m]} | {f_int(rmse_m)} +/- {f_int(rmse_s)} | {f_int(mae_m)} +/- {f_int(mae_s)} | "
          f"{dm_cell} | {p_cell} | {f_ci(lo, hi)} |")
if HAS_RS:
    rs_m = rs["mean"]
    rs_dm = rs["dm_vs_hybrid_base"]
    rs_ci = rs["bootstrap_ci_rmse_ensemble"]
    print(f"| Hybrid LSTM-GNN (random search) | {f_int(rs_m['RMSE'])} +/- {f_int(rs_m['RMSE_std'])} | "
          f"{f_int(rs_m['MAE'])} +/- {f_int(rs_m['MAE_std'])} | {f_dm(rs_dm['DM'])} | "
          f"{f_p(rs_dm['p'])} | {f_ci(rs_ci[0], rs_ci[1])} |")

print()
print("=" * 70)
print("TABEL 6 - Ablation (Hybrid architecture, seed 42)")
print("=" * 70)
print("| Variant | RMSE (IDR) | MAE (IDR) | R-squared |")
print("|---|---|---|---|")
ABL_ORDER = [
    "Identity adjacency",
    "Random graph (mean of 5)",
    "Distance graph k=3",
    "Correlation graph",
    "Distribution network (star from Palembang)",
    "Tanpa fitur COVID",
]
for k in ABL_ORDER:
    if k not in d["ablation"]:
        continue
    a = d["ablation"][k]
    print(f"| {k} | {f_int(a['RMSE'])} | {f_int(a['MAE'])} | {f_r2(a['R2'])} |")

print()
print("=" * 70)
print("TABEL 7 - Per-Region (Hybrid ensemble, original scale)")
print("=" * 70)
print("| Region | RMSE (IDR) | MAE (IDR) | R-squared |")
print("|---|---|---|---|")
for r in d["per_region_hybrid"]:
    m = d["per_region_hybrid"][r]
    print(f"| {r} | {f_int(m['RMSE'])} | {f_int(m['MAE'])} | {f_r2(m['R2'])} |")

print()
print("=" * 70)
print("ANGKA NARASI / ABSTRAK")
print("=" * 70)
h = d["models"]["Hybrid"]
ht = d["models"]["HybridTuned"]
lstm = d["models"]["LSTM"]
gnn = d["models"]["GNN"]
print(f"Hybrid: R2 {f_r2(mean(h['R2']))} RMSE {f_int(mean(h['RMSE']))} MAE {f_int(mean(h['MAE']))} ACC {f_acc(mean(h['ACC']))} AUC {f_auc(mean(h['AUC']))}")
print(f"LSTM:   R2 {f_r2(mean(lstm['R2']))} RMSE {f_int(mean(lstm['RMSE']))} MAE {f_int(mean(lstm['MAE']))} ACC {f_acc(mean(lstm['ACC']))} AUC {f_auc(mean(lstm['AUC']))}")
print(f"GNN:    R2 {f_r2(mean(gnn['R2']))} RMSE {f_int(mean(gnn['RMSE']))} MAE {f_int(mean(gnn['MAE']))} ACC {f_acc(mean(gnn['ACC']))} AUC {f_auc(mean(gnn['AUC']))}")
print(f"HybridTuned: R2 {f_r2(mean(ht['R2']))} RMSE {f_int(mean(ht['RMSE']))} MAE {f_int(mean(ht['MAE']))} ACC {f_acc(mean(ht['ACC']))} AUC {f_auc(mean(ht['AUC']))}")
if HAS_RS:
    rs_m = rs["mean"]
    rs_e = rs["ensemble"]
    print(f"RandomSearch: R2 {f_r2(rs_m['R2'])} +/- {f_r2(rs_m['R2_std'])} RMSE {f_int(rs_m['RMSE'])} MAE {f_int(rs_m['MAE'])} ACC {f_acc(rs_e['ACC'])} AUC {f_auc(rs_e['AUC'])}")

abl = d["ablation"]
star = abl.get("Distribution network (star from Palembang)")
star_txt = f"Star {f_r2(star['R2'])} | " if star else ""
print(f"Ablasi: Identity {f_r2(abl['Identity adjacency']['R2'])} | Random {f_r2(abl['Random graph (mean of 5)']['R2'])} | "
      f"Distance {f_r2(abl['Distance graph k=3']['R2'])} | Correlation {f_r2(abl['Correlation graph']['R2'])} | "
      f"{star_txt}NoCOVID {f_r2(abl['Tanpa fitur COVID']['R2'])}")

res = d["residual_hybrid"]
print(f"Residual: mean {f_int(res['mean'])} Q={res['ljung_box_q']:.3f} p={res['ljung_box_p']:.4f}")

pr = d["per_region_hybrid"]
best = min(pr, key=lambda k: pr[k]["RMSE"])
worst = max(pr, key=lambda k: pr[k]["RMSE"])
print(f"Per region terbaik: {best} R2 {f_r2(pr[best]['R2'])} RMSE {f_int(pr[best]['RMSE'])}")
print(f"Per region terburuk: {worst} R2 {f_r2(pr[worst]['R2'])} RMSE {f_int(pr[worst]['RMSE'])} MAE {f_int(pr[worst]['MAE'])}")

print()
print("=" * 70)
print("KONTEKS PANEL")
print("=" * 70)
print(f"Panel: {d['panel']['n_days']} hari, {d['panel']['n_regions']} region")
print(f"Split: train {d['split']['train']}, val {d['split']['val']}, test {d['split']['test']}")
print(f"Zero actuals test: {d['zero_actuals_test']} dari {d['split']['test'] * d['panel']['n_regions']}")
print(f"Graf: {d['graph']['n_edges']} edge, threshold {d['graph']['threshold']:.2f} km")
print(f"COVID: {d['covid']['flag_active_days']} hari aktif")