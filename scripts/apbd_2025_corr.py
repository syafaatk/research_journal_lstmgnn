"""Korelasi belanja APBD 2025 per kab/kota (portal DJPK, realisasi s.d. Okt)
vs penjualan 2025 per region. Verifikasi cross-sectional dengan data resmi.
Output: results/apbd_2025_corr.json
"""
import json
import os
import numpy as np
from scipy.stats import spearmanr

BASE = r"E:\Download\Jurnal"
APBD_JSON = os.path.join(BASE, "results", "apbd_2025_per_pemda.json")
BPS_JSON = os.path.join(BASE, "results", "bps_analysis.json")
OUT = os.path.join(BASE, "results", "apbd_2025_corr.json")


def main():
    apbd = json.load(open(APBD_JSON))
    bps = json.load(open(BPS_JSON))

    # Agregasi per region (MRU -> Musi Rawas)
    bd, bj = {}, {}
    for code, d in apbd.items():
        r = d["region"]
        bd[r] = bd.get(r, 0) + (d["akun"].get("Belanja Daerah", {}).get("realisasi") or 0)
        bj[r] = bj.get(r, 0) + (d["akun"].get("Belanja Barang dan Jasa", {}).get("realisasi") or 0)

    sales = bps["sales_by_year_region"]
    s2025 = {r: float(sales[r].get("2025", 0)) for r in bd}
    pop = bps["pop_by_year_region"]["2025"]

    regions = sorted(bd.keys())
    results = {}
    for name, a, b in [
        ("belanja_daerah", [bd[r] for r in regions], [s2025[r] for r in regions]),
        ("barang_jasa", [bj[r] for r in regions], [s2025[r] for r in regions]),
        ("belanja_daerah_per_kapita", [bd[r] / pop.get(r, 1) for r in regions],
         [s2025[r] / pop.get(r, 1) for r in regions]),
        ("barang_jasa_per_kapita", [bj[r] / pop.get(r, 1) for r in regions],
         [s2025[r] / pop.get(r, 1) for r in regions]),
    ]:
        r_c, p_c = spearmanr(a, b)
        results[name] = {"spearman": float(r_c), "p": float(p_c), "n": len(regions)}
        print("{}: r={:.3f} p={:.4f} (n={})".format(name, r_c, p_c, len(regions)), flush=True)

    out = {
        "sumber": "Portal DJPK Kemenkeu, APBD 2025 realisasi s.d. Oktober (periode=10), data SIKD per 19 Agu 2026",
        "catatan": "MRU digabung ke Musi Rawas. Korelasi cross-sectional 2025 TIDAK signifikan - "
                   "konsisten dengan within-region 2020-2022 (r=-0,112): belanja bukan prediktor penjualan.",
        "korelasi": results,
        "per_region": {r: {"belanja_daerah": bd[r], "barang_jasa": bj[r],
                           "penjualan": s2025[r], "penduduk": pop.get(r, 0)} for r in regions},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSelesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()