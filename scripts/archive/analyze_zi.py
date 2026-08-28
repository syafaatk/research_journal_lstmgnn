# -*- coding: utf-8 -*-
"""Rangkum hasil eksperimen final zero-inflated + COVID."""
import json
import numpy as np

d = json.load(open(r"E:\Download\Jurnal\skrip\experiment_results_zi.json", encoding="utf-8"))

print("=== PANEL ===")
print(f"hari: {d['panel']['n_days']} | region: {d['panel']['n_regions']} | faktur: {d['panel']['n_records_mapped']}")
print(f"COVID aktif: {d['covid']['flag_active_days']} hari, s.d. {d['covid']['end']}")
print(f"split: train {d['split']['train']}, val {d['split']['val']}, test {d['split']['test']}")
print(f"zero test: {d['zero_actuals_test']} dari {d['split']['test'] * 17}")

print("\n=== MODEL ===")
for k, v in d["models"].items():
    r2 = np.mean(v["R2"]); r2s = np.std(v["R2"])
    print(f"{k}: R2 {r2:.4f} +- {r2s:.4f} | RMSE {np.mean(v['RMSE']):,.0f} | "
          f"MAE {np.mean(v['MAE']):,.0f} | ACC {np.mean(v['ACC']):.4f} | AUC {np.mean(v['AUC']):.4f}")

print("\n=== BASELINE ===")
for k, v in d["baselines"].items():
    print(f"{k}: R2 {v['R2']:.4f} RMSE {v['RMSE']:,.0f} MAE {v['MAE']:,.0f}")

print("\n=== DM (vs HybridTuned) ===")
for k, v in d["dm"].items():
    print(f"{k}: DM={v['DM']:.3f} p={v['p']:.4f}")

print("\n=== BOOTSTRAP CI RMSE ===")
for k, v in d["bootstrap_ci"].items():
    print(f"{k}: [{v[0]:,.0f}, {v[1]:,.0f}]")

print("\n=== ABLASI ===")
for k, v in d["ablation"].items():
    print(f"{k}: R2 {v['R2']:.4f} RMSE {v['RMSE']:,.0f} MAE {v['MAE']:,.0f}")

print("\n=== PER WILAYAH (HybridTuned seed 42) ===")
for k, v in d["per_region"].items():
    print(f"{k}: R2 {v['R2']:.4f} RMSE {v['RMSE']:,.0f} MAE {v['MAE']:,.0f}")

print("\n=== RESIDUAL ===")
print(d["residual"])

print("\n=== GRAF ===")
print(f"edges: {d['graph']['n_edges']}, degrees: {d['graph']['degrees']}")
print(f"knn: min {np.min(d['graph']['knn_dists']):.2f}, median {np.median(d['graph']['knn_dists']):.2f}, "
      f"max {np.max(d['graph']['knn_dists']):.2f}")

print("\n=== ZERO FRACTION PER REGION ===")
for k, v in d["panel"]["zero_frac"].items():
    print(f"{k}: {v:.1%}")