"""Analisis indikator kinerja RS 2024 (88 RS) vs penjualan per region.
Agregasi per region: jumlah RS, total tempat tidur, total pasien keluar,
BOR rata-rata (weighted), BTO, TOI, ALOS.
Korelasi: (1) cross-sectional 2024 vs penjualan 2024;
(2) vs residual model V3 (test 2025).
Output: results/rs_kinerja_2024.json
"""
import json
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
RS_JSON = os.path.join(BASE, "data", "Indikator kinerja pelayanan di RS Provinsi Sumatera Selatan.json")
OUT = os.path.join(BASE, "results", "rs_kinerja_2024.json")

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

# Mapping nama RS -> region (diverifikasi via web untuk yang ambigu)
RS_MAP = {
    "RS Umum Dr. Noesmir Baturaja": "Ogan Komering Ulu",
    "RS Umum Daerah Dr. Ibnu Sutowo Baturaja": "Ogan Komering Ulu",
    "RS Umum Santo Antonio": "Ogan Komering Ulu",
    "RS Khusus Bedah Dokter Maulana AK": "Ogan Komering Ulu",
    "RS Ibu dan Anak Amanna": "Ogan Komering Ulu",
    "RS Ibu dan Anak Graha Kurnia": "Ogan Komering Ulu",
    "RS Ibu dan Anak Prima Qonita": "Ogan Komering Ulu",
    "RS Ibu dan Anak AGDA": "Ogan Komering Ulu",
    "RS Umum Daerah Kayuagung": "Ogan Komering Ilir",
    "RS Umum Daerah Tugu Jaya": "Ogan Komering Ilir",
    "RS Umum Bukit Asam Medika": "Muara Enim",
    "RS Umum Daerah dr. H. M. Rabain Muara Enim": "Muara Enim",
    "RS Umum Daerah Semende Darat Laut": "Muara Enim",
    "RS Umum Daerah Lubai Ulu": "Muara Enim",
    "RS Umum Daerah Gelumbang": "Muara Enim",
    "RS Karunia Indah Medika": "Muara Enim",
    "RS Umum Trijaya Medical Center": "Muara Enim",
    "RS Umum Daerah Lahat": "Lahat",
    "RS Tk. IV Lahat": "Lahat",
    "RS Umum Daerah Tipe D Tanjung Tebat": "Lahat",
    "RS Ibu dan Anak Adellia Graha Medika": "Lahat",
    "RS Umum Daerah Muara Beliti": "Musi Rawas",
    "RS Umum Daerah Dr. Sobirin Kabupaten Musi Rawas": "Musi Rawas",
    "RS Umum Daerah Sekayu": "Musi Banyuasin",
    "RS Umum Daerah Sungai Lilin": "Musi Banyuasin",
    "RS Umum Daerah Bayung Lincir": "Musi Banyuasin",
    "RS Umum Daerah Banyuasin": "Banyuasin",
    "RS Hermina OPI Jakabaring": "Palembang",  # operasional Jakabaring; alamat adm. Banyuasin
    "RS Umum Daerah Pratama Makarti Jaya": "Banyuasin",
    "RS Umum Daerah Sukajadi": "Banyuasin",
    "RS Bunda Medika Jakabaring": "Palembang",  # operasional Jakabaring; alamat adm. Banyuasin
    "RS Umum Pusat Dr. Rivai Abdullah": "Palembang",
    "RS Umum Daerah Muara Dua": "Ogan Komering Ulu Selatan",
    "RS Umum Daerah Martapura": "Ogan Komering Ulu Timur",
    "RS Islam At-Taqwa Gumawang": "Ogan Komering Ulu Timur",
    "Charitas Hospital Belitang": "Ogan Komering Ulu Timur",
    "RS Umum Daerah Ogan Komering Ulu Timur": "Ogan Komering Ulu Timur",
    "RS Umum Daerah Kabupaten Ogan Ilir": "Ogan Ilir",
    "RS Umum Mahyuzahra": "Ogan Ilir",
    "RS Ar-Royyan": "Ogan Ilir",
    "RS Pratama Pendopo": "Empat Lawang",
    "RS Umum Daerah Kabupaten Empat Lawang": "Empat Lawang",
    "RS Umum Daerah Talang Ubi": "Penukal Abab Lematang Ilir",
    "RS Pratama Tanah Abang": "Penukal Abab Lematang Ilir",
    "RS Umum Daerah Rupit Kabupaten Musi Rawas Utara": "Musi Rawas",  # MRU digabung ke Musi Rawas
    "RS Umum Pusat Dr. Mohammad Hoesin Palembang": "Palembang",
    "RS Umum Pertamina Palembang": "Palembang",
    "RS Umum dr. AK. Gani Kota Palembang": "Palembang",
    "RS Umum Pusri Palembang": "Palembang",
    "RS Umum Charitas Hospital Palembang": "Palembang",
    "RS Jiwa Ernaldi Bahar Provinsi Sumatera Selatan": "Palembang",
    "RS Islam Siti Khadijah": "Palembang",
    "RS Umum Sriwijaya": "Palembang",
    "RS Umum Bunda Palembang": "Palembang",
    "RS Umum Daerah Palembang Bari": "Palembang",
    "RS Umum Myria Palembang": "Palembang",
    "RS Muhammadiyah Palembang": "Palembang",
    "RS Khusus Mata Masyarakat Provinsi Sumatera Selatan": "Palembang",
    "RS Ibu dan Anak Rika Amelia": "Palembang",
    "RS Hermina Palembang": "Palembang",
    "Charitas Hospital Kenten": "Palembang",
    "RS Khusus Gigi dan Mulut Palembang Provinsi Sumate": "Palembang",
    "RS Bhayangkara M. Hasan Palembang": "Palembang",
    "RS Umum YK Madira Palembang": "Palembang",
    "RS Pelabuhan Palembang": "Palembang",
    "RS Siloam Sriwijaya Palembang": "Palembang",
    "RS Umum Graha Mandiri": "Palembang",
    "RS Ibu dan Anak Trinanda Palembang": "Palembang",
    "RS Ibu dan Anak Bunda Noni": "Palembang",
    "RS Ibu dan Anak Az-Zahra Palembang": "Palembang",
    "RS Ibu dan Anak Marissa Palembang": "Palembang",
    "RS Musi Medika Cendikia": "Palembang",
    "RS Umum Ar-Rasyid Palembang": "Palembang",
    "RS Umum Daerah Siti Fatimah Provinsi Sumatera Sela": "Palembang",
    "RS Ibu dan Anak Mama": "Palembang",
    "RS Umum Daerah Gandus Palembang": "Palembang",
    "RS Permata Palembang": "Palembang",
    "RS Umum Fadhilah Kota Prabumulih": "Prabumulih",
    "RS Umum Daerah Kota Prabumulih": "Prabumulih",
    "RS Pertamina Kota Prabumulih": "Prabumulih",
    "RS AR Bunda Kota Prabumulih": "Prabumulih",
    "RS Umum Daerah Basemah Kota Pagar Alam": "Pagar Alam",
    "RS AR Bunda Kota Lubuk Linggau": "Lubuk Linggau",
    "RS Siloam Silampari": "Lubuk Linggau",
    "RS Ibu dan Anak Dwi Sari": "Lubuk Linggau",
    "RS Ibu dan Anak Ananda Lubuk Linggau": "Lubuk Linggau",
    "RS Umum Daerah Petanang": "Lubuk Linggau",
    "RS Umum Daerah Siti Aisyah Kota Lubuk Linggau": "Lubuk Linggau",
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
    with open(RS_JSON, encoding="utf-8") as f:
        rs = json.load(f)

    # Agregasi per region
    agg = {r: {"n_rs": 0, "tt": 0.0, "pasien_keluar": 0.0, "hari_perawatan": 0.0,
               "lama_dirawat": 0.0, "bor_w": 0.0, "bto_w": 0.0, "toi_w": 0.0, "alos_w": 0.0}
           for r in REGIONS}
    unmapped = []
    for row in rs:
        name = row["nama_rumah_sakit"]
        region = RS_MAP.get(name)
        if region is None:
            unmapped.append(name)
            continue
        tt = to_num(row["jumlah_tempat_tidur"])
        pk = to_num(row["pasien_keluar"])
        hp = to_num(row["jumlah_hari_perawatan"])
        ld = to_num(row["jumlah_lama_dirawat"])
        bor = to_num(row["bor_persen"])
        bto = to_num(row["bto_kali"])
        toi = to_num(row["toi_hari"])
        alos = to_num(row["alos_hari"])
        a = agg[region]
        a["n_rs"] += 1
        if not np.isnan(tt):
            a["tt"] += tt
        if not np.isnan(pk):
            a["pasien_keluar"] += pk
        if not np.isnan(hp):
            a["hari_perawatan"] += hp
        if not np.isnan(ld):
            a["lama_dirawat"] += ld
        if not np.isnan(bor) and not np.isnan(tt):
            a["bor_w"] += bor * tt
        if not np.isnan(bto) and not np.isnan(tt):
            a["bto_w"] += bto * tt
        if not np.isnan(toi) and not np.isnan(tt):
            a["toi_w"] += toi * tt
        if not np.isnan(alos) and not np.isnan(tt):
            a["alos_w"] += alos * tt

    if unmapped:
        print("TIDAK TERPETAKAN:", unmapped)

    # BOR/BTO/TOI/ALOS rata-rata tertimbang tempat tidur
    for r in REGIONS:
        a = agg[r]
        if a["tt"] > 0:
            a["bor"] = a["bor_w"] / a["tt"]
            a["bto"] = a["bto_w"] / a["tt"]
            a["toi"] = a["toi_w"] / a["tt"]
            a["alos"] = a["alos_w"] / a["tt"]
        else:
            a["bor"] = a["bto"] = a["toi"] = a["alos"] = np.nan
        a.pop("bor_w"); a.pop("bto_w"); a.pop("toi_w"); a.pop("alos_w")

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

    # Tabel hasil
    print("\n=== Agregasi RS 2024 per region ===")
    print("{:<28} {:>4} {:>7} {:>9} {:>7} {:>7} {:>7}".format(
        "Region", "RS", "TT", "PasKel", "BOR", "BTO", "ALOS"))
    for r in REGIONS:
        a = agg[r]
        print("{:<28} {:>4} {:>7.0f} {:>9.0f} {:>7.1f} {:>7.1f} {:>7.1f}".format(
            r, a["n_rs"], a["tt"], a["pasien_keluar"], a["bor"], a["bto"], a["alos"]))

    print("\n=== Korelasi vs penjualan 2024 (cross-sectional, n=16) ===")
    results = {}
    for key in ["n_rs", "tt", "pasien_keluar", "hari_perawatan", "bor", "bto", "toi", "alos"]:
        vals = np.array([agg[r][key] for r in REGIONS])
        c1 = spearman(vals, sales_2024.values)
        c2 = spearman(vals, resid)
        if c1:
            results[key] = {"sales2024": {"r": c1[0], "p": c1[1], "n": c1[2]}}
            print("  {:>16}: sales2024 r={:+.3f} p={:.4f} | resid2025 r={:+.3f} p={:.4f}".format(
                key, c1[0], c1[1], c2[0] if c2 else float("nan"), c2[1] if c2 else float("nan")), flush=True)
        if c2:
            results[key]["resid2025"] = {"r": c2[0], "p": c2[1], "n": c2[2]}

    # Validasi: jumlah RS dari JSON vs BPS rs_2024 (RS_umum + RS_khusus)
    print("\n=== Validasi jumlah RS: JSON vs BPS 2024 ===")
    bps_path = os.path.join(BASE, "data", "bps", "rs_2024.csv")
    if os.path.exists(bps_path):
        bps = pd.read_csv(bps_path, encoding="utf-8-sig")
        bps_map = {
            "Ogan Komering Ulu": "Ogan Komering Ulu", "Ogan Komering Ilir": "Ogan Komering Ilir",
            "Muara Enim": "Muara Enim", "Lahat": "Lahat", "Musi Rawas": "Musi Rawas",
            "Musi Banyuasin": "Musi Banyuasin", "Ogan Komering Ulu Selatan": "Ogan Komering Ulu Selatan",
            "Ogan Komering Ulu Timur": "Ogan Komering Ulu Timur", "Ogan Ilir": "Ogan Ilir",
            "Empat Lawang": "Empat Lawang", "Musi Rawas Utara": "Musi Rawas",
            "Banyu Asin": "Banyuasin", "Penukal Abab Lematang Ilir": "Penukal Abab Lematang Ilir",
            "Kota Palembang": "Palembang", "Kota Prabumulih": "Prabumulih",
            "Kota Pagar Alam": "Pagar Alam", "Kota Lubuklinggau": "Lubuk Linggau",
        }
        bps = bps[bps.iloc[:, 0].isin(bps_map)].copy()
        bps["region"] = bps.iloc[:, 0].map(bps_map)
        bps["rs_total"] = pd.to_numeric(bps["Jumlah Rumah Sakit Umum"], errors="coerce") + \
                          pd.to_numeric(bps["Jumlah Rumah Sakit Khusus"], errors="coerce")
        bps_agg = bps.groupby("region")["rs_total"].sum()
        for r in REGIONS:
            b = bps_agg.get(r, np.nan)
            j = agg[r]["n_rs"]
            flag = "" if (np.isnan(b) or abs(b - j) <= 1) else "  <-- BEDA"
            print("  {:<28} JSON={:>2} BPS={:>2}{}".format(r, j, b, flag))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"agg": {r: {k: (None if isinstance(v, float) and np.isnan(v) else v)
                                for k, v in agg[r].items()} for r in REGIONS},
                   "sales2024": sales_2024.to_dict(),
                   "resid2025": resid.tolist(),
                   "correlations": results}, f, indent=2, default=str)
    print("\nSelesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()