"""Analisis Belanja Barang dan Jasa (BJ) per region 2020-2025 vs penjualan.
Sumber: portal DJPK Kemenkeu untuk SEMUA tahun (konsisten), realisasi:
2020-2024 = akhir tahun (periode=12), 2025 = s.d. Oktober (periode=10).
Fokus BJ karena paling relevan untuk pengadaan alat kesehatan.
Analisis: pooled, within-region (demean), cross-sectional per tahun.
Output: results/apbd_bj_analysis.json
"""
import json
import os
import numpy as np
from scipy.stats import spearmanr

BASE = r"E:\Download\Jurnal"
OUT = os.path.join(BASE, "results", "apbd_bj_analysis.json")

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def load_bj(year):
    """DJPK json -> {region: belanja_barang_jasa_rupiah} (MRU digabung ke Musi Rawas)."""
    path = os.path.join(BASE, "results", "apbd_{}_per_pemda.json".format(year))
    data = json.load(open(path))
    out = {}
    for code, d in data.items():
        r = d["region"]
        bj = d["akun"].get("Belanja Barang dan Jasa", {}).get("realisasi") or 0
        out[r] = out.get(r, 0) + bj
    return out


def main():
    bps = json.load(open(os.path.join(BASE, "results", "bps_analysis.json")))
    sales = bps["sales_by_year_region"]

    bj = {y: load_bj(y) for y in YEARS}
    penjualan = {}
    for r, yd in sales.items():
        for y, v in yd.items():
            penjualan.setdefault(r, {})[int(y)] = float(v)

    regions = sorted(bj[2020].keys())
    print("=== Belanja Barang & Jasa per region per tahun (miliar rupiah) ===")
    hdr = "{:>28}" + " {:>10}" * len(YEARS)
    print(hdr.format("region", *YEARS))
    for r in regions:
        print(hdr.format(r, *[round(bj[y].get(r, 0) / 1e9) for y in YEARS]))

    rows = []
    for r in regions:
        for y in YEARS:
            if r in bj[y] and y in penjualan[r]:
                rows.append((r, y, bj[y][r], penjualan[r][y]))
    b = [x[2] for x in rows]
    s = [x[3] for x in rows]
    r_pool, p_pool = spearmanr(b, s)
    print("\nPooled BJ 2020-2025 (n={}): r={:.3f} p={:.4f}".format(len(rows), r_pool, p_pool), flush=True)

    d_b, d_s = [], []
    for r in regions:
        sub = [x for x in rows if x[0] == r]
        mb = np.mean([x[2] for x in sub])
        ms = np.mean([x[3] for x in sub])
        for _, _, bb, ss in sub:
            d_b.append(bb - mb)
            d_s.append(ss - ms)
    r_within, p_within = spearmanr(d_b, d_s)
    print("Within-region BJ 2020-2025 (n={}): r={:.3f} p={:.4f}".format(len(d_b), r_within, p_within), flush=True)

    print("\nCross-sectional per tahun:")
    cross = {}
    for y in YEARS:
        sub = [x for x in rows if x[1] == y]
        if len(sub) >= 6:
            rr, pp = spearmanr([x[2] for x in sub], [x[3] for x in sub])
            cross[y] = {"r": float(rr), "p": float(pp), "n": len(sub)}
            print("  {}: r={:.3f} p={:.4f} (n={})".format(y, rr, pp, len(sub)), flush=True)

    print("\nPer region (n={} tahun):".format(len(YEARS)))
    per_region = {}
    for r in regions:
        sub = [x for x in rows if x[0] == r]
        b_r = [x[2] for x in sub]
        s_r = [x[3] for x in sub]
        if len(set(b_r)) > 1 and len(set(s_r)) > 1:
            rr, pp = spearmanr(b_r, s_r)
            per_region[r] = {"r": float(rr), "p": float(pp)}
            print("  {:>28}: r={:+.3f} p={:.3f}".format(r, rr, pp), flush=True)
        else:
            print("  {:>28}: konstan".format(r), flush=True)

    out = {
        "sumber": "Portal DJPK Kemenkeu, Belanja Barang dan Jasa realisasi per kab/kota, "
                  "2020-2024 periode=12 (akhir tahun), 2025 periode=10 (s.d. Oktober). Rupiah.",
        "pooled": {"n": len(rows), "r": float(r_pool), "p": float(p_pool)},
        "within_region": {"n": len(d_b), "r": float(r_within), "p": float(p_within)},
        "cross_sectional": cross,
        "per_region": per_region,
        "bj_miliar": {str(y): {r: round(bj[y].get(r, 0) / 1e9) for r in regions} for y in YEARS},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSelesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()