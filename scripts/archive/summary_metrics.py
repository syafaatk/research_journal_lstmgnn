import json
import numpy as np

with open(r"C:\Users\ADMLAY~1\AppData\Local\Temp\opencode\experiment_results.json", encoding="utf-8") as f:
    r = json.load(f)

print("=== METRIK SKALA TRANSFORMASI (per model, 5 seeds) ===")
for name, m in r["models"].items():
    t = m["transformed"]
    print(f"{name}: RMSE {np.mean(t['RMSE']):.4f} +- {np.std(t['RMSE']):.4f} | "
          f"MAE {np.mean(t['MAE']):.4f} | MAPE {np.mean(t['MAPE']):.2f} | R2 {np.mean(t['R2']):.4f}")

print("\n=== METRIK SKALA ASLI (per model, 5 seeds) ===")
for name, m in r["models"].items():
    o = m["original"]
    print(f"{name}: RMSE {np.mean(o['RMSE']):,.0f} +- {np.std(o['RMSE']):,.0f} | "
          f"MAE {np.mean(o['MAE']):,.0f} | MAPE {np.mean(o['MAPE']):.2f} | R2 {np.mean(o['R2']):.4f}")

print("\n=== DM TEST (skala asli, seed 42) ===")
for k, v in r["dm"].items():
    print(f"{k} vs HybridTuned: DM={v['DM']:.3f}, p={v['p']:.4f}")

print("\n=== BOOTSTRAP RMSE CI (skala asli, seed 42) ===")
for k, v in r["bootstrap_ci"].items():
    print(f"{k}: [{v[0]:,.0f}, {v[1]:,.0f}]")

print("\n=== RESIDUAL ===")
print(r["residual"])

print("\n=== ABLASI (skala asli) ===")
for k, v in r["ablation"].items():
    print(f"{k}: RMSE={v['RMSE']:,.0f} MAE={v['MAE']:,.0f} R2={v['R2']:.3f}")

print("\n=== ZERO FRACTION per region (panel) ===")
for k, v in r["panel"]["zero_frac"].items():
    print(f"  {k}: {v:.2%}")