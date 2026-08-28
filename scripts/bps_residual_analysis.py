"""Analisis residual model V3 vs faktor BPS (test 2025).
Menjawab: apakah faktor X BPS menjelaskan kesalahan prediksi model?
Jika residual tidak berkorelasi dengan faktor BPS, menambahkannya sebagai
fitur model tidak akan mengubah hasil eksperimen.
Output: results/bps_residual_analysis.json
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
GEO = os.path.join(BASE, "data", "customers_geocoded_final.csv")
OUT = os.path.join(BASE, "results", "bps_residual_analysis.json")

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
N = len(REGIONS)

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
    "Banyuasin": "Banyuasin",
    "Pali": "Penukal Abab Lematang Ilir",
    "Palembang": "Palembang",
    "Prabumulih": "Prabumulih",
    "Pagar Alam": "Pagar Alam",
    "Lubuk Linggau": "Lubuk Linggau",
    "OKU Selatan": "Ogan Komering Ulu Selatan",
    "OKU Timur": "Ogan Komering Ulu Timur",
}


def load_bps(prefix, year, value_col=None):
    path = os.path.join(BASE, "data", "bps", "{}_{}.csv".format(prefix, year))
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df.iloc[:, 0].isin(BPS_MAP)].copy()
    df["region"] = df.iloc[:, 0].map(BPS_MAP)
    if value_col and value_col in df.columns:
        df["value"] = pd.to_numeric(df[value_col], errors="coerce")
    else:
        df["value"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    return df.groupby("region")["value"].sum()


def load_static(relpath, value_col=None, skiprows=0):
    path = os.path.join(BASE, "data", "Data BPS", relpath)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig", skiprows=skiprows)
    df = df[df.iloc[:, 0].isin(BPS_MAP)].copy()
    df["region"] = df.iloc[:, 0].map(BPS_MAP)
    if value_col:
        df["value"] = pd.to_numeric(df[value_col], errors="coerce")
    else:
        df["value"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    return df.groupby("region")["value"].sum()


def main():
    # --- Pipeline data (identik exp_features.py) ---
    df = pd.read_excel(DATA)
    df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
    df = df.drop_duplicates("d_jual_nofak")
    df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                            "jual_total_fak": "jual_total"})
    df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
    df["region"] = df["pelanggan_kota"].map(CITY_MAP)
    df["period"] = df["jual_tanggal"].dt.to_period("D")
    panel = df.pivot_table(index="period", columns="region", values="jual_total", aggfunc="sum")
    panel = panel.reindex(columns=REGIONS).fillna(0.0).sort_index()
    n_days = len(panel)

    WINDOW = 30
    y_all, t_all = [], []
    for i in range(n_days - WINDOW):
        y_all.append(panel.iloc[i + WINDOW].values)
        t_all.append(panel.index[i + WINDOW])
    y_all = np.array(y_all)
    te_mask = np.array([m.start_time > pd.Timestamp("2024-12-31") for m in t_all])
    yte = y_all[te_mask]
    tte = [m.start_time for m in t_all if m.start_time > pd.Timestamp("2024-12-31")]

    # --- Load ensemble V3 preds ---
    preds = []
    for s in [42, 7, 123]:
        p = np.load(os.path.join(BASE, "results", "v3_preds_seed{}.npy".format(s)))
        preds.append(p)
    pred_ens = np.mean(preds, axis=0)
    resid = yte - pred_ens
    abs_resid = np.abs(resid)

    # Residual per region (rata-rata test 2025)
    resid_mean = resid.mean(axis=0)
    absresid_mean = abs_resid.mean(axis=0)
    rmse_region = np.sqrt((resid ** 2).mean(axis=0))

    # --- Faktor BPS 2025 ---
    pop = load_bps("penduduk", 2025)
    dbd = load_bps("dbd", 2025, "Jumlah Kasus Penyakit - Angka Kesakitan DBD per 100.000 Penduduk")
    rs = load_bps("rs", 2025)
    sarana = load_bps("sarana", 2025)
    jarak = load_static("Jarak Ibukota Berdasarkan Tipe Jarak, 2025.csv", skiprows=2)
    luas = load_static("Luas Daerah dan Jumlah Pulau Menurut Kabupaten_Kota di Provinsi Sumatera Selatan, 2025.csv",
                       value_col="Luas Wilayah (Km2)")

    factors = {"penduduk": pop, "dbd_per_100k": dbd, "rs_puskesmas": rs,
               "sarana_kesehatan": sarana, "jarak_km": jarak, "luas_km2": luas}

    print("=== Korelasi residual model V3 (test 2025) vs faktor BPS ===")
    print("n = 16 region. Residual = y_true - y_pred (rata-rata per region).\n")
    out = {}
    for name, s in factors.items():
        if s is None:
            continue
        s = s.reindex(REGIONS)
        mask = s.notna().values
        if mask.sum() < 6:
            print("  {}: data kurang (n={})".format(name, mask.sum()))
            continue
        r_res, p_res = spearmanr(resid_mean[mask], s.values[mask])
        r_abs, p_abs = spearmanr(absresid_mean[mask], s.values[mask])
        r_rmse, p_rmse = spearmanr(rmse_region[mask], s.values[mask])
        out[name] = {
            "residual": {"r": float(r_res), "p": float(p_res)},
            "abs_residual": {"r": float(r_abs), "p": float(p_abs)},
            "rmse_region": {"r": float(r_rmse), "p": float(p_rmse)},
            "n": int(mask.sum()),
        }
        print("  {:>16}: resid r={:+.3f} p={:.3f} | |resid| r={:+.3f} p={:.3f} | RMSE r={:+.3f} p={:.3f}".format(
            name, r_res, p_res, r_abs, p_abs, r_rmse, p_rmse), flush=True)

    # Tabel residual per region
    tbl = pd.DataFrame({
        "RMSE": rmse_region, "resid_mean": resid_mean,
        "penduduk": pop.reindex(REGIONS), "dbd": dbd.reindex(REGIONS),
        "jarak": jarak.reindex(REGIONS), "luas": luas.reindex(REGIONS),
    }, index=REGIONS)
    print("\n=== Residual per region (test 2025) ===")
    print(tbl.round(0).to_string(), flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"corr": out, "region_table": tbl.round(0).to_dict()}, f, indent=2, default=str)
    print("\nSelesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()