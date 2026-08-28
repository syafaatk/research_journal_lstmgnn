"""Uji hubungan pengadaan syringe dengan wabah (COVID, DBD) - grain benar.
Rekonstruksi dari syringe_covid.py + syringe_dbd.py (skrip lama terhapus saat
rapikan folder; log lamanya memakai qty hasil dedup per faktur -> basi).
Grain analisis = baris item (qty level-item, TANPA dedup faktur).
Output: results/syringe_wabah.json
"""
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
DATA_BPS = os.path.join(BASE, "data", "bps")
OUT = os.path.join(BASE, "results", "syringe_wabah.json")

REGIONS = ["Banyuasin", "Empat Lawang", "Lahat", "Lubuk Linggau", "Muara Enim",
           "Musi Banyuasin", "Musi Rawas", "Ogan Ilir", "Ogan Komering Ilir",
           "Ogan Komering Ulu", "Ogan Komering Ulu Selatan", "Ogan Komering Ulu Timur",
           "Pagar Alam", "Palembang", "Penukal Abab Lematang Ilir", "Prabumulih"]

# wilayah penjualan -> region panel (sama dengan bps_analysis.py)
CITY_MAP = {
    "Palembang": "Palembang", "Lahat": "Lahat", "Baturaja": "Ogan Komering Ulu",
    "Oku (Ogan Komering Ulu)": "Ogan Komering Ulu Timur", "Musi Banyuasin": "Musi Banyuasin",
    "Muara Enim": "Muara Enim", "Tanjung Enim": "Muara Enim", "Muara Lawai": "Muara Enim",
    "Lubuklinggau": "Lubuk Linggau", "Ogan Ilir": "Ogan Ilir", "Oku Selatan": "Ogan Komering Ulu Selatan",
    "Prabumulih": "Prabumulih", "Prabumulih Timur": "Prabumulih",
    "Pali (Penukal Abab Lematang Ilir)": "Penukal Abab Lematang Ilir", "Talang Ubi": "Penukal Abab Lematang Ilir",
    "Musi Rawas": "Musi Rawas", "Oku Timur": "Ogan Komering Ulu Timur", "Martapura": "Ogan Komering Ulu Timur",
    "Empat Lawang": "Empat Lawang", "Banyuasin": "Banyuasin", "Pagaralam": "Pagar Alam",
    "Muara Rupit": "Musi Rawas", "Rupit": "Musi Rawas",
    "Oki (Ogan Komering Ilir)": "Ogan Komering Ilir",
}

# nama kab/kota BPS -> region panel (sama dengan bps_analysis.py)
BPS_MAP = {
    "Ogan Komering Ulu": "Ogan Komering Ulu",
    "Ogan Komering Ilir": "Ogan Komering Ilir",
    "Muara Enim": "Muara Enim",
    "Lahat": "Lahat",
    "Musi Rawas": "Musi Rawas",
    "Musi Banyuasin": "Musi Banyuasin",
    "Ogan Komering Ulu Selatan": "Ogan Komering Ulu Selatan",
    "Ogan Komering Ulu Timur": "Ogan Komering Ulu Timur",
    "Ogan Ilir": "Ogan Ilir",
    "Empat Lawang": "Empat Lawang",
    "Musi Rawas Utara": "Musi Rawas",
    "Banyu Asin": "Banyuasin",
    "Penukal Abab Lematang Ilir": "Penukal Abab Lematang Ilir",
    "Kota Palembang": "Palembang",
    "Kota Prabumulih": "Prabumulih",
    "Kota Pagar Alam": "Pagar Alam",
    "Kota Lubuklinggau": "Lubuk Linggau",
}

YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
DBD_COL = "Jumlah Kasus Penyakit - Angka Kesakitan DBD per 100.000 Penduduk"


def load_dbd():
    """(region -> {year: dbd per 100k}, prov -> {year: dbd per 100k})"""
    out = {}
    prov = {}
    for y in YEARS:
        p = os.path.join(DATA_BPS, "dbd_{}.csv".format(y))
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p, encoding="utf-8-sig")
        df = df[df.iloc[:, 0].isin(BPS_MAP)].copy()
        df["region"] = df.iloc[:, 0].map(BPS_MAP)
        df["value"] = pd.to_numeric(df[DBD_COL], errors="coerce")
        for r, v in df.groupby("region")["value"].sum().items():
            out.setdefault(r, {})[y] = float(v) if pd.notna(v) else None
        # baris tingkat provinsi (rate resmi BPS, bukan jumlah antar kab/kota)
        df_full = pd.read_csv(p, encoding="utf-8-sig")
        prow = df_full[df_full.iloc[:, 0].astype(str).str.strip() == "Sumatera Selatan"]
        if len(prow) == 1:
            v = pd.to_numeric(prow[DBD_COL], errors="coerce").iloc[0]
            if pd.notna(v):
                prov[y] = float(v)
    return out, prov


def main():
    # --- Syringe qty level-item (TANPA dedup faktur) ---
    df = pd.read_excel(DATA)
    df = df.dropna(subset=["d_jual_nofak", "tanggal", "barang_nama", "d_jual_qty"])
    mask = df["barang_nama"].astype(str).str.lower().str.contains("syringe", na=False)
    sy = df[mask].copy()
    sy["tanggal"] = pd.to_datetime(sy["tanggal"])
    sy["ym"] = sy["tanggal"].dt.to_period("M").astype(str)
    sy["year"] = sy["tanggal"].dt.year
    print("Total syringe: {} faktur, {:,} unit".format(
        sy["d_jual_nofak"].nunique(), int(sy["d_jual_qty"].sum())))

    # --- Uji COVID: bulanan pandemi vs endemi ---
    mon = sy.groupby("ym")["d_jual_qty"].sum().sort_index()
    pand = [ym for ym in mon.index if "2020-07" <= ym <= "2022-12"]
    ende = [ym for ym in mon.index if ym >= "2023-01"]
    u, p_mw = stats.mannwhitneyu(mon[pand], mon[ende], alternative="two-sided")
    flag = np.array([1 if ym in pand else 0 for ym in mon.index])
    r_sp, p_sp = stats.spearmanr(mon.values, flag)
    print("\n=== COVID: pandemi (Jul20-Des22, n={}) vs endemi (2023+, n={}) ===".format(
        len(pand), len(ende)))
    print("  Pandemi : mean={:,.0f}/bulan median={:,.0f}".format(mon[pand].mean(), mon[pand].median()))
    print("  Endemi  : mean={:,.0f}/bulan median={:,.0f}".format(mon[ende].mean(), mon[ende].median()))
    print("  Mann-Whitney U={:.0f} p={:.4f}".format(u, p_mw))
    print("  Spearman qty vs pandemi: r={:+.3f} p={:.4f}".format(r_sp, p_sp))

    peaks = {}
    for y, g in sy.groupby("year"):
        bym = g.groupby("ym")["d_jual_qty"].sum()
        peaks[str(int(y))] = {"puncak_ym": bym.idxmax(), "puncak_qty": int(bym.max()),
                              "total_tahun": int(g["d_jual_qty"].sum())}
        print("  {}: puncak {} ({:,} unit) | total {:,}".format(
            int(y), bym.idxmax(), int(bym.max()), int(g["d_jual_qty"].sum())))

    events = {"delta_jul21": "2021-07", "delta_agu21": "2021-08",
              "omicron_jan22": "2022-01", "omicron_feb22": "2022-02",
              "endemi_nov23": "2023-11"}
    ev_out = {k: int(mon.get(v, 0)) for k, v in events.items()}
    print("\n  Event: " + ", ".join("{}={:,}".format(k, v) for k, v in ev_out.items()))

    # --- Uji DBD: panel region-tahun ---
    dbd, dbd_prov = load_dbd()
    rows = []
    sub = sy[sy["wilayah"].isin(CITY_MAP)].copy()
    sub["region"] = sub["wilayah"].map(CITY_MAP)
    pan = sub.groupby(["year", "region"])["d_jual_qty"].sum()
    for (y, r), q in pan.items():
        d = dbd.get(r, {}).get(y)
        if d is not None:
            rows.append({"year": int(y), "region": r, "qty": int(q), "dbd_per_100k": d})
    panel = pd.DataFrame(rows)
    print("\n=== DBD: panel region-tahun n={} ===".format(len(panel)))
    res_panel = {}
    if len(panel) >= 5:
        r1, p1 = stats.spearmanr(panel["qty"], panel["dbd_per_100k"])
        print("  pooled: r={:+.3f} p={:.4f}".format(r1, p1))
        # within-region: residu terhadap mean region
        dm = panel["qty"] - panel.groupby("region")["qty"].transform("mean")
        dd = panel["dbd_per_100k"] - panel.groupby("region")["dbd_per_100k"].transform("mean")
        r2, p2 = stats.spearmanr(dm, dd)
        print("  within-region: r={:+.3f} p={:.4f}".format(r2, p2))
        agg = panel.groupby("year")["qty"].sum()
        for y, q in agg.items():
            pv = dbd_prov.get(int(y))
            print("  {}: syringe={:>9,} dbd_prov_bps={} /100k".format(
                y, int(q), "{:,.1f}".format(pv) if pv is not None else "n/a"))
        res_panel = {"n": int(len(panel)), "pooled_r": float(r1), "pooled_p": float(p1),
                     "within_r": float(r2), "within_p": float(p2),
                     "dbd_prov_bps_per_100k": {str(k): v for k, v in sorted(dbd_prov.items())}}

    out = {
        "catatan": "grain = baris item (tanpa dedup faktur); rekonstruksi syringe_covid+dbd",
        "total_syringe": {"faktur": int(sy["d_jual_nofak"].nunique()),
                          "qty": int(sy["d_jual_qty"].sum())},
        "covid": {"pandemi_n": len(pand), "endemi_n": len(ende),
                  "pandemi_mean": float(mon[pand].mean()), "endemi_mean": float(mon[ende].mean()),
                  "mannwhitney_p": float(p_mw), "spearman_r": float(r_sp), "spearman_p": float(p_sp)},
        "puncak_per_tahun": peaks,
        "event_bulanan": ev_out,
        "dbd": res_panel,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\nSelesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()
