"""Uji SEMUA kolom Data BPS yang tersedia sebagai kandidat faktor X.
Kolom belum diuji: TBC, HIV, kusta, malaria (file dbd); RSK, puskesmas,
klinik, posyandu (file rs); apotek, balai kesehatan (file sarana).
Analisis: (1) korelasi pooled per region-tahun vs penjualan;
(2) korelasi residual model V3 (test 2025) vs faktor 2025.
Output: results/bps_all_factors.json
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(BASE, "results", "bps_all_factors.json")

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
BPS_MAP = {
    "Ogan Komering Ulu": "Ogan Komering Ulu", "Ogan Komering Ilir": "Ogan Komering Ilir",
    "Muara Enim": "Muara Enim", "Lahat": "Lahat", "Musi Rawas": "Musi Rawas",
    "Musi Banyuasin": "Musi Banyuasin", "Ogan Komering Ulu Selatan": "Ogan Komering Ulu Selatan",
    "Ogan Komering Ulu Timur": "Ogan Komering Ulu Timur", "Ogan Ilir": "Ogan Ilir",
    "Empat Lawang": "Empat Lawang", "Musi Rawas Utara": "Musi Rawas",
    "Banyu Asin": "Banyuasin", "Penukal Abab Lematang Ilir": "Penukal Abab Lematang Ilir",
    "Kota Palembang": "Palembang", "Kota Prabumulih": "Prabumulih",
    "Kota Pagar Alam": "Pagar Alam", "Kota Lubuklinggau": "Lubuk Linggau",
    "Banyuasin": "Banyuasin", "Pali": "Penukal Abab Lematang Ilir",
    "Palembang": "Palembang", "Prabumulih": "Prabumulih", "Pagar Alam": "Pagar Alam",
    "Lubuk Linggau": "Lubuk Linggau", "OKU Selatan": "Ogan Komering Ulu Selatan",
    "OKU Timur": "Ogan Komering Ulu Timur",
}
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]

# Kolom per file (nama pendek -> nama kolom)
FACTOR_COLS = {
    "dbd": {
        "TBC_penemuan": "Jumlah Kasus Penyakit - Angka Penemuan TBC",
        "TBC_keberhasilan": "Jumlah Kasus Penyakit - Angka Keberhasilan Pengobatan TBC",
        "HIV_baru": "Jumlah Kasus Penyakit - HIV/AIDS Kasus Baru",
        "kusta": "Jumlah Kasus Penyakit - Penemuan Kasus Baru Kusta per 100.000 Penduduk",
        "malaria": "Jumlah Kasus Penyakit - Angka Kesakitan Malaria per 1.000 Penduduk",
        "DBD": "Jumlah Kasus Penyakit - Angka Kesakitan DBD per 100.000 Penduduk",
    },
    "rs": {
        "RS_umum": "Jumlah Rumah Sakit Umum",
        "RS_khusus": "Jumlah Rumah Sakit Khusus",
        "puskesmas_rawat_inap": "Jumlah Puskesmas Rawat Inap",
        "puskesmas_non_rawat": "Jumlah Puskesmas Non Rawat Inap",
        "klinik_pratama": "Jumlah Klinik Pratama",
        "posyandu": "Jumlah Posyandu",
    },
    "sarana": {
        "desa_RS": "Desa/Kelurahan Yang Memiliki Sarana Kesehatan - Rumah Sakit",
        "desa_puskesmas": "Desa/Kelurahan Yang Memiliki Sarana Kesehatan - Puskesmas",
        "desa_pustu": "Desa/Kelurahan Yang Memiliki Sarana Kesehatan - Puskesmas Pembantu",
        "desa_apotek": "Desa/Kelurahan Yang Memiliki Sarana Kesehatan - Apotek",
        "desa_klinik_utama": "Desa/Kelurahan Yang Memiliki Sarana Kesehatan - Klinik Utama (Unit)",
        "desa_balai": "Desa/Kelurahan Yang Memiliki Sarana Kesehatan - Balai Kesehatan",
        "desa_klinik_pratama": "Desa/Kelurahan Yang Memiliki Sarana Kesehatan - Klinik Pratama",
    },
}


def load_factor(prefix, col, year):
    path = os.path.join(BASE, "data", "bps", "{}_{}.csv".format(prefix, year))
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    if col not in df.columns:
        return None
    df = df[df.iloc[:, 0].isin(BPS_MAP)].copy()
    df["region"] = df.iloc[:, 0].map(BPS_MAP)
    df["value"] = pd.to_numeric(df[col], errors="coerce")
    return df.groupby("region")["value"].sum()


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 6:
        return None
    r, p = spearmanr(a[m], b[m])
    return float(r), float(p), int(m.sum())


def main():
    # Penjualan per region-tahun
    df = pd.read_excel(DATA)
    df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
    df = df.drop_duplicates("d_jual_nofak")
    df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                            "jual_total_fak": "jual_total"})
    df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
    df["region"] = df["pelanggan_kota"].map(CITY_MAP)
    df["year"] = df["jual_tanggal"].dt.year
    sales = df.groupby(["year", "region"])["jual_total"].sum().unstack(fill_value=0).reindex(columns=REGIONS)

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
    yte = y_all[te_mask]
    resid = (yte - pred_ens).mean(axis=0)  # rata-rata residual per region

    results = {}
    print("=== Korelasi pooled (region-tahun) vs penjualan ===")
    for prefix, cols in FACTOR_COLS.items():
        for short, col in cols.items():
            vals = {}
            for y in YEARS:
                v = load_factor(prefix, col, y)
                if v is not None:
                    vals[y] = v
            rows = []
            for y, v in vals.items():
                for r in v.index:
                    if r in sales.columns and y in sales.index:
                        rows.append((r, y, sales.loc[y, r], v[r]))
            if len(rows) < 6:
                continue
            c = spearman([x[2] for x in rows], [x[3] for x in rows])
            if not c:
                continue
            results[short] = {"pooled": {"r": c[0], "p": c[1], "n": c[2]}}
            # Within-region (demean) untuk memisahkan efek skala
            regions_ = sorted(set(x[0] for x in rows))
            d_b, d_s = [], []
            for r in regions_:
                sub = [x for x in rows if x[0] == r]
                mb = np.mean([x[3] for x in sub])
                ms = np.mean([x[2] for x in sub])
                for _, _, ss, bb in sub:
                    d_b.append(bb - mb)
                    d_s.append(ss - ms)
            cw = spearman(d_b, d_s)
            if cw:
                results[short]["within_region"] = {"r": cw[0], "p": cw[1], "n": cw[2]}
            tag = "SIG" if c[1] < 0.05 else "   "
            tagw = "SIG" if (cw and cw[1] < 0.05) else "   "
            print("  {:>24}: pooled r={:+.3f} p={:.4f} (n={}) [{}] | within r={:+.3f} p={:.4f} [{}]".format(
                short, c[0], c[1], c[2], tag,
                cw[0] if cw else float("nan"), cw[1] if cw else float("nan"), tagw), flush=True)

    print("\n=== Korelasi residual V3 (test 2025) vs faktor 2025 ===")
    for prefix, cols in FACTOR_COLS.items():
        for short, col in cols.items():
            v = load_factor(prefix, col, 2025)
            if v is None:
                continue
            v = v.reindex(REGIONS)
            m = v.notna().values
            if m.sum() < 6:
                continue
            c = spearman(resid[m], v.values[m])
            if c:
                results[short]["residual_2025"] = {"r": c[0], "p": c[1], "n": c[2]}
                print("  {:>24}: resid r={:+.3f} p={:.4f} (n={})".format(short, c[0], c[1], c[2]), flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSelesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()