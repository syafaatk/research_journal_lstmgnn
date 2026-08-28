"""Analisis Data BPS sebagai kandidat faktor X.
Data: penduduk, kasus penyakit (DBD), RS/puskesmas/posyandu, sarana kesehatan desa
per kab/kota per tahun (2020-2025). Korelasi dengan penjualan per region per tahun.
Output: results/bps_analysis.json + tabel.
"""
import json
import os
import numpy as np
import pandas as pd

BASE = r"E:\Download\Jurnal"
DATA_BPS = os.path.join(BASE, "data", "bps")
SALES_XLSX = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(BASE, "results", "bps_analysis.json")

REGIONS = ["Banyuasin", "Empat Lawang", "Lahat", "Lubuk Linggau", "Muara Enim",
           "Musi Banyuasin", "Musi Rawas", "Ogan Ilir", "Ogan Komering Ilir",
           "Ogan Komering Ulu", "Ogan Komering Ulu Selatan", "Ogan Komering Ulu Timur",
           "Pagar Alam", "Palembang", "Penukal Abab Lematang Ilir", "Prabumulih"]

# Nama kab/kota di BPS -> region kita (MRU digabung ke Musi Rawas).
# Dua konvensi nama: file penduduk (tanpa "Kota", "Pali", "Banyuasin")
# vs file DBD/RS ("Kota Palembang", "Banyu Asin", "Penukal Abab Lematang Ilir").
BPS_MAP = {
    # sama di kedua konvensi
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
    # konvensi DBD/RS
    "Banyu Asin": "Banyuasin",
    "Penukal Abab Lematang Ilir": "Penukal Abab Lematang Ilir",
    "Kota Palembang": "Palembang",
    "Kota Prabumulih": "Prabumulih",
    "Kota Pagar Alam": "Pagar Alam",
    "Kota Lubuklinggau": "Lubuk Linggau",
    # konvensi penduduk
    "Banyuasin": "Banyuasin",
    "Pali": "Penukal Abab Lematang Ilir",
    "Palembang": "Palembang",
    "Prabumulih": "Prabumulih",
    "Pagar Alam": "Pagar Alam",
    "Lubuk Linggau": "Lubuk Linggau",
    # konvensi belanja/jarak (singkatan OKU)
    "OKU Selatan": "Ogan Komering Ulu Selatan",
    "OKU Timur": "Ogan Komering Ulu Timur",
}
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]


def find_csv(prefix, year):
    p = os.path.join(DATA_BPS, "{}_{}.csv".format(prefix, year))
    return p if os.path.exists(p) else None


def load_bps(prefix, year, value_col=None):
    path = find_csv(prefix, year)
    if path is None:
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df.iloc[:, 0].isin(BPS_MAP)].copy()
    df["region"] = df.iloc[:, 0].map(BPS_MAP)
    if value_col and value_col in df.columns:
        df["value"] = pd.to_numeric(df[value_col], errors="coerce")
    else:
        # Fallback: kolom kedua (file penduduk header-nya aneh, tahun di baris 3)
        df["value"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    return df.groupby("region")["value"].sum()


def load_sales_by_year():
    df = pd.read_excel(SALES_XLSX)
    df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
    df = df.drop_duplicates("d_jual_nofak")
    df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                            "jual_total_fak": "jual_total"})
    city_map = {
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
    df = df[df["pelanggan_kota"].isin(city_map)].copy()
    df["region"] = df["pelanggan_kota"].map(city_map)
    df["year"] = df["jual_tanggal"].dt.year
    return df.groupby(["year", "region"])["jual_total"].sum().unstack(fill_value=0).reindex(columns=REGIONS)


def spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return None
    from scipy.stats import spearmanr
    r, p = spearmanr(a[mask], b[mask])
    return float(r), float(p)


def load_belanja_kabkota():
    """Belanja pemerintah per kab/kota (ribu rupiah), 2020-2023. Format lebar: tahun di baris 2."""
    path = os.path.join(DATA_BPS, "..", "Data BPS", "Belanja Pemerintah menurut kabupaten kota.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig", skiprows=1)
    df = df[df.iloc[:, 0].isin(BPS_MAP)].copy()
    df["region"] = df.iloc[:, 0].map(BPS_MAP)
    out = {}
    for i, col in enumerate(df.columns[1:5]):
        year = 2020 + i
        vals = pd.to_numeric(df[col], errors="coerce")
        if year == 2021:
            # Kolom 2021 dalam rupiah (bukan ribu rupiah) - inkonsistensi file BPS
            vals = vals / 1000.0
        out[year] = vals.groupby(df["region"]).sum()
    return out


def load_realisasi_belanja():
    """Realisasi belanja pemerintah provinsi (ribu rupiah) per tahun 2020-2025."""
    path = os.path.join(DATA_BPS, "..", "Data BPS", "Realisasi Belanja Pemerintah.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig", skiprows=1)
    df = df[df.iloc[:, 0].astype(str).str.contains("Belanja Barang dan Jasa", na=False)]
    if df.empty:
        return None
    row = df.iloc[0]
    out = {}
    for i, col in enumerate(row.index[1:]):
        year = 2020 + i
        out[year] = float(pd.to_numeric(row[col], errors="coerce"))
    return out


def load_jarak_ibukota():
    """Jarak ibukota kab/kota dari Palembang (km), 2025."""
    path = os.path.join(DATA_BPS, "..", "Data BPS", "Jarak Ibukota Berdasarkan Tipe Jarak, 2025.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig", skiprows=2)
    df = df[df.iloc[:, 0].isin(BPS_MAP)].copy()
    df["region"] = df.iloc[:, 0].map(BPS_MAP)
    df["jarak"] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    # MRU digabung ke Musi Rawas: pakai jarak terjauh (max), bukan sum
    return df.groupby("region")["jarak"].max()


def load_luas_daerah():
    """Luas wilayah per kab/kota (km2), 2025."""
    path = os.path.join(DATA_BPS, "..", "Data BPS",
                        "Luas Daerah dan Jumlah Pulau Menurut Kabupaten_Kota di Provinsi Sumatera Selatan, 2025.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df.iloc[:, 0].isin(BPS_MAP)].copy()
    df["region"] = df.iloc[:, 0].map(BPS_MAP)
    df["luas"] = pd.to_numeric(df["Luas Wilayah (Km2)"], errors="coerce")
    return df.groupby("region")["luas"].sum()


def main():
    sales = load_sales_by_year()
    print("Penjualan per region per tahun (juta):")
    print((sales / 1e6).round(0).to_string(), flush=True)

    pop = {y: load_bps("penduduk", y) for y in YEARS}
    dbd = {y: load_bps("dbd", y, "Jumlah Kasus Penyakit - Angka Kesakitan DBD per 100.000 Penduduk") for y in YEARS}

    # Cek kolom yang tersedia di file RS
    path = find_csv("rs", 2025)
    if path:
        cols = pd.read_csv(path, encoding="utf-8-sig", nrows=2).columns.tolist()
        print("\nKolom RS 2025:", cols, flush=True)
    rs = {y: load_bps("rs", y, cols[1]) for y in YEARS}

    # Tabel ringkas 2025
    print("\n=== Ringkasan 2025 per region ===")
    tbl = pd.DataFrame({
        "Penduduk": pop[2025], "DBD/100k": dbd[2025],
        "Penjualan (jt)": sales.loc[2025] / 1e6,
    })
    tbl["Penjualan/kapita (rb)"] = (sales.loc[2025] / pop[2025] / 1e3).round(1)
    print(tbl.round(1).to_string(), flush=True)

    # Korelasi pooled (semua tahun x region)
    rows = []
    for y in YEARS:
        if y not in sales.index:
            continue
        for r in REGIONS:
            if r not in sales.columns:
                continue
            if r in pop[y].index:
                rows.append({
                    "year": y, "region": r,
                    "sales": sales.loc[y, r],
                    "pop": pop[y].get(r, np.nan),
                    "dbd": dbd[y].get(r, np.nan) if y in dbd and dbd[y] is not None else np.nan,
                })
    df = pd.DataFrame(rows)

    corr = {}
    for name, col in [("pop", "pop"), ("dbd", "dbd")]:
        c = spearman(df["sales"], df[col])
        corr[name] = {"spearman": c[0], "p": c[1], "n": int(df[col].notna().sum())} if c else None
        print("\nSpearman {} vs sales: r={:.3f}, p={:.4f}, n={}".format(
            name, c[0], c[1], int(df[col].notna().sum())) if c else "{}: tidak cukup data".format(name), flush=True)

    # Korelasi per tahun
    corr_by_year = {}
    for y in YEARS:
        sub = df[df["year"] == y]
        c_pop = spearman(sub["sales"], sub["pop"])
        c_dbd = spearman(sub["sales"], sub["dbd"])
        corr_by_year[y] = {
            "pop": c_pop, "dbd": c_dbd,
            "n": int(sub["dbd"].notna().sum()),
        }
        print("  {}: pop r={:.3f} (p={:.3f}) | dbd r={:.3f} (p={:.3f})".format(
            y, c_pop[0], c_pop[1], c_dbd[0], c_dbd[1]) if c_pop and c_dbd else "  {}: data kurang".format(y), flush=True)

    # DBD vs penjualan: region dengan DBD tinggi vs rendah (2025)
    hi = df[(df["year"] == 2025) & (df["dbd"] > df["dbd"].median())]["sales"].mean()
    lo = df[(df["year"] == 2025) & (df["dbd"] <= df["dbd"].median())]["sales"].mean()
    print("\n2025: penjualan rata-rata region DBD tinggi = {:,.0f} vs DBD rendah = {:,.0f} (rasio {:.2f})".format(
        hi, lo, hi / lo if lo else float("nan")), flush=True)

    # --- Belanja pemerintah per kab/kota (2020-2023) ---
    belanja = load_belanja_kabkota()
    if belanja:
        print("\n=== Belanja pemerintah per kab/kota (miliar) ===")
        bdf = pd.DataFrame({y: belanja[y] / 1e6 for y in belanja})
        print(bdf.round(0).to_string(), flush=True)
        rows_b = []
        for y in belanja:
            for r in REGIONS:
                if r in sales.columns and r in belanja[y].index:
                    rows_b.append({"year": y, "region": r,
                                   "sales": sales.loc[y, r], "belanja": belanja[y].get(r, np.nan)})
        bdf2 = pd.DataFrame(rows_b)
        c_b = spearman(bdf2["sales"], bdf2["belanja"])
        if c_b:
            corr["belanja"] = {"spearman": c_b[0], "p": c_b[1], "n": int(bdf2["belanja"].notna().sum())}
            print("Spearman belanja vs sales: r={:.3f}, p={:.4f}, n={}".format(
                c_b[0], c_b[1], int(bdf2["belanja"].notna().sum())), flush=True)

    # --- Realisasi belanja provinsi (2020-2025) vs total penjualan ---
    realisasi = load_realisasi_belanja()
    if realisasi:
        print("\n=== Realisasi belanja barang & jasa provinsi vs total penjualan ===")
        total_sales = sales.sum(axis=1)
        for y in sorted(realisasi):
            if y in total_sales.index:
                print("  {}: belanja barang&jasa {:,.0f} M vs penjualan {:,.0f} jt (rasio {:.2f})".format(
                    y, realisasi[y] / 1e6, total_sales.loc[y] / 1e6,
                    realisasi[y] / max(total_sales.loc[y], 1)), flush=True)
        yrs = [y for y in realisasi if y in total_sales.index]
        if len(yrs) >= 4:
            c_r = spearman([realisasi[y] for y in yrs], [total_sales.loc[y] for y in yrs])
            if c_r:
                corr["realisasi_belanja"] = {"spearman": c_r[0], "p": c_r[1], "n": len(yrs)}
                print("Spearman realisasi belanja vs total penjualan: r={:.3f}, p={:.4f}, n={}".format(
                    c_r[0], c_r[1], len(yrs)), flush=True)

    # --- Jarak ibukota dari Palembang (2025) ---
    jarak = load_jarak_ibukota()
    if jarak is not None:
        print("\n=== Jarak ibukota dari Palembang (2025) ===")
        jdf = pd.DataFrame({"jarak": jarak, "sales": sales.loc[2025]})
        print(jdf.round(1).to_string(), flush=True)
        c_j = spearman(jdf["sales"], jdf["jarak"])
        if c_j:
            corr["jarak"] = {"spearman": c_j[0], "p": c_j[1], "n": int(jdf["jarak"].notna().sum())}
            print("Spearman jarak vs sales: r={:.3f}, p={:.4f}".format(c_j[0], c_j[1]), flush=True)

    # --- Luas daerah (2025) ---
    luas = load_luas_daerah()
    if luas is not None:
        ldf = pd.DataFrame({"luas": luas, "sales": sales.loc[2025]})
        c_l = spearman(ldf["sales"], ldf["luas"])
        if c_l:
            corr["luas"] = {"spearman": c_l[0], "p": c_l[1], "n": int(ldf["luas"].notna().sum())}
            print("Spearman luas vs sales: r={:.3f}, p={:.4f}".format(c_l[0], c_l[1]), flush=True)

    out = {
        "sales_by_year_region": sales.to_dict(),
        "pop_by_year_region": {str(y): pop[y].to_dict() for y in YEARS},
        "dbd_by_year_region": {str(y): (dbd[y].to_dict() if dbd[y] is not None else {}) for y in YEARS},
        "corr_pooled": corr,
        "corr_by_year": {str(y): v for y, v in corr_by_year.items()},
        "summary_2025": tbl.round(2).to_dict(),
        "belanja_by_year_region": {str(y): belanja[y].to_dict() for y in belanja} if belanja else {},
        "realisasi_belanja": realisasi,
        "jarak_ibukota": jarak.to_dict() if jarak is not None else {},
        "luas_daerah": luas.to_dict() if luas is not None else {},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSelesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()