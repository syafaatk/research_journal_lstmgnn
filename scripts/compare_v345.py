# -*- coding: utf-8 -*-
"""Bandingkan hasil V3 (resmi), V4 (exp_temuan.json), V5 (exp_temuan_v5.json).

Output: tabel metrik ensemble + DM test antar varian terbaik.
DM antar eksperimen dihitung dari preds ensemble yang disimpan di masing-masing JSON.
"""
import json
import os
import numpy as np
from scipy import stats

BASE = r"E:\Download\Jurnal"
P_V3 = os.path.join(BASE, "results", "exp_features_v3.json")
P_V4 = os.path.join(BASE, "results", "exp_temuan.json")
P_V5 = os.path.join(BASE, "results", "exp_temuan_v5.json")
P_ZI = os.path.join(BASE, "results", "experiment_results_zi_geocoded.json")


def dm_test(e1, e2):
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    dm = d.mean() / np.sqrt(d.var(ddof=1) / T)
    p = 2 * (1 - stats.norm.cdf(abs(dm)))
    return float(dm), float(p)


def main():
    rows = []  # (label, R2, RMSE, MAE, pred_ens, yte)

    if os.path.exists(P_V3):
        d = json.load(open(P_V3))
        e = d["ensemble"]
        # Muat preds ensemble V3 resmi dari cache per-seed (untuk DM test)
        p_v3 = None
        seed_files = [os.path.join(BASE, "results", "v3_preds_seed{}.npy".format(s))
                      for s in (42, 7, 123)]
        if all(os.path.exists(f) for f in seed_files):
            p_v3 = np.mean([np.load(f) for f in seed_files], axis=0)
            # y_test diambil dari eksperimen lain saat tersedia (split identik);
            # diverifikasi lewat sanity RMSE di bawah.
        rows.append(("V3 resmi (rs+kalender)", e["R2"], e["RMSE"], e["MAE"], p_v3, None))
        print("V3 dimuat:", {k: round(v, 4) for k, v in e.items()})
    if os.path.exists(P_V4):
        d = json.load(open(P_V4))
        yte = np.array(d["y_test"])
        for k, v in d["results"].items():
            rows.append(("V4:" + k, v["ensemble"]["R2"], v["ensemble"]["RMSE"],
                         v["ensemble"]["MAE"], np.array(d["preds"][k]), yte))
        print("V4 dimuat:", list(d["results"].keys()))
    if os.path.exists(P_V5):
        d = json.load(open(P_V5))
        yte = np.array(d["y_test"])
        for k, v in d["results"].items():
            rows.append(("V5:" + k, v["ensemble"]["R2"], v["ensemble"]["RMSE"],
                         v["ensemble"]["MAE"], np.array(d["preds"][k]), yte))
        print("V5 dimuat:", list(d["results"].keys()))

    # Isi y_test untuk baris V3 resmi dari eksperiman mana pun, lalu sanity-check
    v3_row = next((r for r in rows if r[0].startswith("V3 resmi")), None)
    any_yte = next((r[5] for r in rows if r[5] is not None), None)
    if v3_row is not None and v3_row[4] is not None and any_yte is not None:
        p_v3, yte = v3_row[4], any_yte
        if p_v3.shape == yte.shape:
            rmse_chk = float(np.sqrt(np.mean((yte - p_v3) ** 2)))
            print("Sanity V3 resmi vs y_test eksp: RMSE={:,.0f} (harus dekat {:,.0f})".format(
                rmse_chk, v3_row[2]))
            if abs(rmse_chk - v3_row[2]) / v3_row[2] < 0.05:
                rows = [r if not r[0].startswith("V3 resmi")
                        else ("V3 resmi (rs+kalender)", r[1], r[2], r[3], p_v3, yte)
                        for r in rows]
            else:
                print("WARNING: window/alignmen beda - V3 resmi DIKELUARKAN dari DM test")
                rows = [r for r in rows if not r[0].startswith("V3 resmi")]
        else:
            print("WARNING: shape {} != {} - V3 resmi DIKELUARKAN dari DM test".format(
                p_v3.shape, yte.shape))
            rows = [r for r in rows if not r[0].startswith("V3 resmi")]

    print("\n=== Tabel gabungan (ensemble, skala asli) ===")
    print("{:28s} {:>8s} {:>14s} {:>14s}".format("Varian", "R2", "RMSE", "MAE"))
    for label, r2, rmse, mae, _, _ in rows:
        print("{:28s} {:8.4f} {:>14,.0f} {:>14,.0f}".format(label, r2, rmse, mae))

    # DM test antar semua pasangan yang punya preds
    have = [(l, p, y) for l, _, _, _, p, y in rows if p is not None]
    if len(have) >= 2:
        print("\n=== DM test antar varian (positif = baris lebih buruk) ===")
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                l1, p1, y1 = have[i]
                l2, p2, y2 = have[j]
                if y1.shape != y2.shape:
                    continue
                dm, p = dm_test((y1 - p1).ravel(), (y2 - p2).ravel())
                print("{:22s} vs {:22s} DM={:>7.3f} p={:.4f}".format(l1, l2, dm, p))


if __name__ == "__main__":
    main()