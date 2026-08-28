# -*- coding: utf-8 -*-
"""Cek std RMSE/MAE aktual per model dari JSON."""
import json
import numpy as np

d = json.load(open(r"E:\Download\Jurnal\skrip\experiment_results_zi.json", encoding="utf-8"))
for k, v in d["models"].items():
    print(f"{k}: seeds={len(v['R2'])}")
    print(f"  R2   mean {np.mean(v['R2']):.4f} std {np.std(v['R2']):.4f}")
    print(f"  RMSE mean {np.mean(v['RMSE']):,.0f} std {np.std(v['RMSE']):,.0f}")
    print(f"  MAE  mean {np.mean(v['MAE']):,.0f} std {np.std(v['MAE']):,.0f}")
    print(f"  ACC  mean {np.mean(v['ACC']):.4f} std {np.std(v['ACC']):.4f}")
    print(f"  AUC  mean {np.mean(v['AUC']):.4f} std {np.std(v['AUC']):.4f}")