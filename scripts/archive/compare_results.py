# -*- coding: utf-8 -*-
"""Bandingkan hasil eksperimen lama (17 region, experiment_results_zi.json)
dengan baru (16 region geocoded, experiment_results_zi_geocoded.json).

Usage: python compare_results.py
Jika file baru belum ada, cetak status dan keluar (eksperimen masih jalan).
"""
import json
import os
import sys

NEW = r"E:\Download\Jurnal\skrip\experiment_results_zi_geocoded.json"
OLD = r"E:\Download\Jurnal\skrip\experiment_results_zi.json"

if len(sys.argv) >= 2:
    NEW = sys.argv[1]
if len(sys.argv) >= 3:
    OLD = sys.argv[2]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def mean(xs):
    return sum(xs) / len(xs)


def fmt(x, nd=4):
    if x is None:
        return "-"
    try:
        if x != x:  # NaN
            return "NaN"
        return f"{x:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def graph_line(tag, d):
    g = d["graph"]
    deg = g["degrees"]
    knn = [k for row in g["knn_dists"] for k in row]
    knn_sorted = sorted(knn)
    med = knn_sorted[len(knn_sorted) // 2]
    return (f"{tag}: {g['n_edges']} edge, threshold {g['threshold']:.2f} km, "
            f"derajat min {min(deg):.0f} max {max(deg):.0f} rata {mean(deg):.2f}, "
            f"3-NN min {min(knn):.2f} median {med:.2f} max {max(knn):.2f}")


def model_line(tag, d):
    out = [f"{tag}:"]
    for name in ["LSTM", "GNN", "Hybrid", "HybridTuned"]:
        m = d["models"].get(name)
        if not m:
            continue
        out.append(f"  {name}: R2 {fmt(mean(m['R2']))}, RMSE {fmt(mean(m['RMSE']), 0)}, "
                   f"MAE {fmt(mean(m['MAE']), 0)}, ACC {fmt(mean(m['ACC']))}")
    return "\n".join(out)


def main():
    new = load(NEW)
    old = load(OLD)
    if new is None:
        print("File baru belum ada:", NEW)
        print("Eksperimen masih berjalan. Cek skrip/zi_geocoded.log")
        return

    print("=" * 72)
    print("PANEL")
    print(f"  baru: {new['panel']['n_regions']} region, {new['panel']['n_days']} hari")
    if old:
        print(f"  lama: {old['panel']['n_regions']} region, {old['panel']['n_days']} hari")

    print("=" * 72)
    print("GRAF")
    print(" ", graph_line("baru", new))
    if old:
        print(" ", graph_line("lama", old))

    print("=" * 72)
    print("MODEL UTAMA (mean seeds)")
    print(model_line("baru", new))
    if old:
        print(model_line("lama", old))

    print("=" * 72)
    print("BASELINES")
    for tag, d in (("baru", new), ("lama", old)):
        if not d:
            continue
        print(f"  {tag}:")
        for name, m in d["baselines"].items():
            print(f"    {name}: R2 {fmt(m.get('R2'))}, RMSE {fmt(m.get('RMSE'), 0)}, MAE {fmt(m.get('MAE'), 0)}")

    print("=" * 72)
    print("ABLASI")
    for tag, d in (("baru", new), ("lama", old)):
        if not d:
            continue
        print(f"  {tag}:")
        for name, m in d["ablation"].items():
            print(f"    {name}: R2 {fmt(m.get('R2'))}, RMSE {fmt(m.get('RMSE'), 0)}, MAE {fmt(m.get('MAE'), 0)}")

    print("=" * 72)
    print("PER REGION (R2 / RMSE / MAE)")
    for tag, d in (("baru", new), ("lama", old)):
        if not d:
            continue
        print(f"  {tag}:")
        for r, m in d["per_region"].items():
            print(f"    {r}: R2 {fmt(m.get('R2'))}, RMSE {fmt(m.get('RMSE'), 0)}, MAE {fmt(m.get('MAE'), 0)}")

    print("=" * 72)
    print("ZERO ACTUALS TEST")
    print(f"  baru: {new.get('zero_actuals_test')}")
    if old:
        print(f"  lama: {old.get('zero_actuals_test')}")

    print("=" * 72)
    print("RESIDUAL")
    print(f"  baru: {new.get('residual')}")
    if old:
        print(f"  lama: {old.get('residual')}")


if __name__ == "__main__":
    main()