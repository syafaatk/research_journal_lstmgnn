"""Analisis gabungan belanja APBD per region 2020-2025 vs penjualan.
Sumber: 2020-2023 BPS (ribu rupiah, 2021 dikoreksi /1000), 2024-2025 DJPK
(portal Kemenkeu, rupiah, realisasi 2024 = akhir tahun, 2025 = s.d. Oktober).
Semua dikonversi ke rupiah. Analisis: pooled, within-region (demean),
cross-sectional per tahun.
Output: results/apbd_combined_analysis.json
"""
import json
import os
import numpy as np
from scipy.stats import spearmanr

BASE = r"E:\Download\Jurnal"
BPS_JSON = os.path.join(BASE, "results", "bps_analysis.json")
APBD24 = os.path.join(BASE, "results", "apbd_2024_per_pemda.json")
APBD25 = os.path.join(BASE, "results", "apbd_2025_per_pemda.json")
OUT = os.path.join(BASE, "results", "apbd_combined_analysis.json")

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def load_djpk(path):
    """DJPK json -> {region: belanja_daerah_rupiah} (MRU digabung ke Musi Rawas)."""
    data = json.load(open(path))
    out = {}
    for code, d in data.items():
        r = d["region"]
        bd = d["akun"].get("Belanja Daerah", {}).get("realisasi") or 0
        out[r] = out.get(r, 0) + bd
    return out


def main():
    bps = json.load(open(BPS_JSON))
    sales = bps["sales_by_year_region"]

    # Belanja per region per tahun (rupiah)
    belanja = {}
    for y in [2020, 2021, 2022, 2023]:
        for r, v in bps["belanja_by_year_region"].get(str(y), {}).items():
            belanja.setdefault(r, {})[y] = float(v) * 1000.0  # ribu -> rupiah
    for y, path in [(2024, APBD24), (2025, APBD25)]:
        for r, v in load_djpk(path).items():
            belanja.setdefault(r, {})[y] = v

    # Penjualan per region per tahun
    penjualan = {}
    for r, yd in sales.items():
        for y, v in yd.items():
            penjualan.setdefault(r, {})[int(y)] = float(v)

    regions = sorted(belanja.keys())
    print("=== Belanja APBD per region per tahun (miliar rupiah) ===")
    hdr = "{:>28}" + " {:>10}" * len(YEARS)
    print(hdr.format("region", *YEARS))
    for r in regions:
        print(hdr.format(r, *[round(belanja[r].get(y, 0) / 1e9) for y in YEARS]))

    # --- Pooled 2020-2025 ---
    rows = []
    for r in regions:
        for y in YEARS:
            if y in belanja[r] and y in penjualan[r]:
                rows.append((r, y, belanja[r][y], penjualan[r][y]))
    b = [x[2] for x in rows]
    s = [x[3] for x in rows]
    r_pool, p_pool = spearmanr(b, s)
    print("\nPooled 2020-2025 (n={}): r={:.3f} p={:.4f}".format(len(rows), r_pool, p_pool), flush=True)

    # --- Within-region (demean) ---
    d_b, d_s = [], []
    for r in regions:
        sub = [x for x in rows if x[0] == r]
        mb = np.mean([x[2] for x in sub])
        ms = np.mean([x[3] for x in sub])
        for _, _, bb, ss in sub:
            d_b.append(bb - mb)
            d_s.append(ss - ms)
    r_within, p_within = spearmanr(d_b, d_s)
    print("Within-region 2020-2025 (n={}): r={:.3f} p={:.4f}".format(len(d_b), r_within, p_within), flush=True)

    # --- Cross-sectional per tahun ---
    print("\nCross-sectional per tahun:")
    cross = {}
    for y in YEARS:
        sub = [x for x in rows if x[1] == y]
        if len(sub) >= 6:
            rr, pp = spearmanr([x[2] for x in sub], [x[3] for x in sub])
            cross[y] = {"r": float(rr), "p": float(pp), "n": len(sub)}
            print("  {}: r={:.3f} p={:.4f} (n={})".format(y, rr, pp, len(sub)), flush=True)

    # --- Per region (korelasi 6 titik) ---
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
        "sumber": "2020-2023 BPS (ribu rupiah, 2021 dikoreksi), 2024 DJPK periode=12 (akhir tahun), "
                  "2025 DJPK periode=10 (s.d. Oktober). Semua dalam rupiah.",
        "pooled": {"n": len(rows), "r": float(r_pool), "p": float(p_pool)},
        "within_region": {"n": len(d_b), "r": float(r_within), "p": float(p_within)},
        "cross_sectional": cross,
        "per_region": per_region,
        "belanja_miliar": {r: {str(y): round(belanja[r].get(y, 0) / 1e9) for y in YEARS} for r in regions},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSelesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()