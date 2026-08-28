"""Analisis cakupan imunisasi Td WUS 2024 vs penjualan per region.
Variabel: jumlah WUS, jumlah imunisasi per dosis (td1-td5), cakupan persen,
total imunisasi, cakupan agregat.
Korelasi: (1) cross-sectional 2024 vs penjualan 2024;
(2) vs residual model V3 (test 2025); (3) sensitivitas tanpa Palembang.
Output: results/imunisasi_2024.json
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
IMU_JSON = os.path.join(BASE, "data",
    "PERSENTASE CAKUPAN IMUNISASI Td PADA WANITA USIA SUBUR (HAMIL DAN TIDAK HAMIL) "
    "MENURUT KABUPATEN/KOTA PROVINSI SUMATERA SELATAN TAHUN 2024.json")
OUT = os.path.join(BASE, "results", "imunisasi_2024.json")

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
REGIONS = ["Banyuasin", "Empat Lawang", "Lahat", "Lubuk Linggau", "Muara Enim",
           "Musi Banyuasin", "Musi Rawas", "Ogan Ilir", "Ogan Komering Ilir",
           "Ogan Komering Ulu", "Ogan Komering Ulu Selatan", "Ogan Komering Ulu Timur",
           "Pagar Alam", "Palembang", "Penukal Abab Lematang Ilir", "Prabumulih"]
IMU_MAP = {
    "Ogan Komering Ulu": "Ogan Komering Ulu", "Ogan Komering Ilir": "Ogan Komering Ilir",
    "Muara Enim": "Muara Enim", "Lahat": "Lahat", "Musi Rawas": "Musi Rawas",
    "Musi Banyuasin": "Musi Banyuasin", "Banyuasin": "Banyuasin",
    "OKU Selatan": "Ogan Komering Ulu Selatan", "OKU Timur": "Ogan Komering Ulu Timur",
    "Ogan Ilir": "Ogan Ilir", "Empat Lawang": "Empat Lawang",
    "PALI": "Penukal Abab Lematang Ilir", "Muratara": "Musi Rawas",
    "Palembang": "Palembang", "Prabumulih": "Prabumulih",
    "Pagar Alam": "Pagar Alam", "Lubuk Linggau": "Lubuk Linggau",
}


def to_num(s):
    if s is None:
        return np.nan
    s = str(s).replace(".", "").replace(",", ".").strip()
    if s in ("", "-"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 6:
        return None
    r, p = spearmanr(a[m], b[m])
    return float(r), float(p), int(m.sum())


def main():
    with open(IMU_JSON, encoding="utf-8") as f:
        rows = json.load(f)

    agg = {r: {"wus": 0.0, "td1": 0.0, "td2": 0.0, "td3": 0.0, "td4": 0.0, "td5": 0.0}
           for r in REGIONS}
    for row in rows:
        name = row["kabupaten_kota"]
        if name.startswith("TOTAL"):
            continue
        region = IMU_MAP.get(name)
        if region is None:
            print("TIDAK TERPETAKAN:", name)
            continue
        a = agg[region]
        a["wus"] += to_num(row["jumlah_wus"])
        for d in ["td1", "td2", "td3", "td4", "td5"]:
            a[d] += to_num(row[d + "_jumlah"])

    # Variabel turunan
    for r in REGIONS:
        a = agg[r]
        a["total_imu"] = sum(a[d] for d in ["td1", "td2", "td3", "td4", "td5"])
        a["cakupan_total"] = a["total_imu"] / a["wus"] * 100 if a["wus"] > 0 else np.nan
        a["cakupan_td5"] = a["td5"] / a["wus"] * 100 if a["wus"] > 0 else np.nan

    # Penjualan 2024 per region
    df = pd.read_excel(DATA)
    df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
    df = df.drop_duplicates("d_jual_nofak")
    df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                            "jual_total_fak": "jual_total"})
    df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
    df["region"] = df["pelanggan_kota"].map(CITY_MAP)
    df["year"] = df["jual_tanggal"].dt.year
    sales_2024 = df[df["year"] == 2024].groupby("region")["jual_total"].sum().reindex(REGIONS).fillna(0.0)

    # Residual V3 test 2025
    preds = [np.load(os.path.join(BASE, "results", "v3_preds_seed{}.npy".format(s))) for s in [42, 7, 123]]
    pred_ens = np.mean(preds, axis=0)
    df2 = pd.read_excel(DATA)
    df2 = df2.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"]).drop_duplicates("d_jual_nofak")
    df2 = df2.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota", "jual_total_fak": "jual_total"})
    df2 = df2[df2["pelanggan_kota"].isin(CITY_MAP)].copy()
    df2["region"] = df2["pelanggan_kota"].map(CITY_MAP)
    df2["period"] = df2["jual_tanggal"].dt.to_period("D")
    panel = df2.pivot_table(index="period", columns="region", values="jual_total", aggfunc="sum")
    panel = panel.reindex(columns=REGIONS).fillna(0.0).sort_index()
    n_days = len(panel)
    WINDOW = 30
    y_all, t_all = [], []
    for i in range(n_days - WINDOW):
        y_all.append(panel.iloc[i + WINDOW].values)
        t_all.append(panel.index[i + WINDOW])
    y_all = np.array(y_all)
    te_mask = np.array([m.start_time > pd.Timestamp("2024-12-31") for m in t_all])
    resid = (y_all[te_mask] - pred_ens).mean(axis=0)

    print("=== Agregasi imunisasi 2024 per region ===")
    print("{:<28} {:>9} {:>8} {:>8} {:>8} {:>8}".format(
        "Region", "WUS", "td1", "td5", "total", "cakupan%"))
    for r in REGIONS:
        a = agg[r]
        print("{:<28} {:>9.0f} {:>8.0f} {:>8.0f} {:>8.0f} {:>8.1f}".format(
            r, a["wus"], a["td1"], a["td5"], a["total_imu"], a["cakupan_total"]))

    print("\n=== Korelasi vs penjualan 2024 dan residual V3 (n=16) ===")
    results = {}
    keys = ["wus", "td1", "td2", "td3", "td4", "td5", "total_imu", "cakupan_total", "cakupan_td5"]
    for key in keys:
        vals = np.array([agg[r][key] for r in REGIONS])
        c1 = spearman(vals, sales_2024.values)
        c2 = spearman(vals, resid)
        if c1:
            results[key] = {"sales2024": {"r": c1[0], "p": c1[1], "n": c1[2]}}
            print("  {:>14}: sales2024 r={:+.3f} p={:.4f} | resid2025 r={:+.3f} p={:.4f}".format(
                key, c1[0], c1[1], c2[0] if c2 else float("nan"), c2[1] if c2 else float("nan")), flush=True)
        if c2:
            results[key]["resid2025"] = {"r": c2[0], "p": c2[1], "n": c2[2]}

    print("\n=== Sensitivitas TANPA Palembang (n=15) ===")
    idx = [r for r in REGIONS if r != "Palembang"]
    for key in keys:
        vals = np.array([agg[r][key] for r in idx])
        sv = np.array([sales_2024[r] for r in idx])
        rv = np.array([resid[REGIONS.index(r)] for r in idx])
        c1 = spearman(vals, sv)
        c2 = spearman(vals, rv)
        print("  {:>14}: sales2024 r={:+.3f} p={:.4f} | resid2025 r={:+.3f} p={:.4f}".format(
            key, c1[0] if c1 else float("nan"), c1[1] if c1 else float("nan"),
            c2[0] if c2 else float("nan"), c2[1] if c2 else float("nan")), flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"agg": {r: {k: (None if isinstance(v, float) and np.isnan(v) else v)
                                for k, v in agg[r].items()} for r in REGIONS},
                   "sales2024": sales_2024.to_dict(),
                   "resid2025": resid.tolist(),
                   "correlations": results}, f, indent=2, default=str)
    print("\nSelesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()