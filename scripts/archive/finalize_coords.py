# -*- coding: utf-8 -*-
"""
Finalisasi koordinat pelanggan:
1. Gabungkan hasil final2 + varian tambahan yang valid.
2. Validasi jarak dengan pusat region KOREKSI (koordinat kota benar, hardcode).
3. Output: customers_geocoded_final.csv (lengkap) + not_found_list.csv
"""
import math
import pandas as pd

IN = r"E:\Download\Jurnal\skrip\customers_geocoded_final2.csv"
OUT = r"E:\Download\Jurnal\skrip\customers_geocoded_final.csv"
OUT_NF = r"E:\Download\Jurnal\skrip\not_found_list.csv"

# Pusat region koreksi (koordinat kota/kabupaten, dari pengetahuan umum)
REGION_CENTER = {
    "Palembang": (-2.9909, 104.7569),
    "Banyuasin": (-2.9800, 104.3800),
    "Musi Banyuasin": (-2.8800, 103.8300),
    "Musi Rawas": (-3.2400, 103.0100),
    "Musi Rawas Utara": (-2.7300, 102.9000),
    "Lahat": (-3.7900, 103.5400),
    "Empat Lawang": (-3.7100, 102.9300),
    "Muara Enim": (-3.6500, 103.7700),
    "Penukal Abab Lematang Ilir": (-3.2900, 103.9100),
    "Prabumulih": (-3.4300, 104.2300),
    "Pagar Alam": (-4.0300, 103.2500),
    "Lubuk Linggau": (-3.3000, 102.8600),
    "Ogan Komering Ulu": (-4.1300, 104.1700),
    "Ogan Komering Ulu Timur": (-4.3000, 104.3400),
    "Ogan Komering Ulu Selatan": (-4.5400, 104.0700),
    "Ogan Ilir": (-3.2400, 104.6600),
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

res = pd.read_csv(IN)

# Koordinat tambahan valid (hasil manual, sudah diverifikasi). Format: (lat, lon, sumber)
EXTRA = {
    "RSUD. SEKAYU": (-2.8914091, 103.8567303, "PHOTON"),
    "RSUD MUARADUA": (-4.5430033, 104.0740295, "PHOTON"),
    "PT. MUTIARA CAHAYA LESARI MEDIKA (RS. Tiara Fatrin)": (-2.9690122, 104.7636364, "PHOTON"),
    "RUMAH SAKIT DR ERNALDI PROVINSI SUMSEL": (-2.9222280, 104.6846093, "PHOTON"),
    "RS Dr. Sobirin Kabupaten Musirawas": (-3.2861345, 102.8823737, "PHOTON"),  # Jl. Yos Sudarso, Lubuk Linggau (sama dgn Siloam)
    "Rumah Sakit TK.IV/DKT Lahat": (-3.7929104, 103.5342464, "PHOTON"),  # bersebelahan RSUD Lahat (Jl. Letjen Harun Sohar)
    "PT. LINTANG LAKSANA UTAMA (RS.SILOAM LINGGAU)": (-3.2861345, 102.8823737, "PHOTON"),
    "RS. PB. Charitas Belitang": (-4.1522567, 104.6354688, "PHOTON"),
    "RSUD KAB. EMPAT LAWANG": (-3.6533147, 103.0515708, "PHOTON"),
    "RS. AT TAQWA GUMAWANG": (-4.1522567, 104.6354688, "PHOTON"),  # area Belitang Madang Raya
    "PT. MEDIKALOKA PALEMBANG": (-2.9558533, 104.7484485, "PHOTON"),   # operator RS Hermina Palembang
    "PT. MEDIKALOKA JAKABARING": (-3.0341709, 104.7905648, "PHOTON"),   # operator RS Hermina OPI Jakabaring
    "PT. GRAHA PUSRI MEDIKA (RS PUSRI)": (-2.9469931, 104.7692778, "PHOTON"),  # RS Pusri Palembang
    "RSUD SITI AISYAH": (-3.2697178, 102.9153505, "PHOTON"),            # Lubuk Linggau Timur I
    "YAYASAN AR-ROYAN NUR IMAN": (-3.2360087, 104.6640174, "PHOTON"),   # RS Ar-Royyan, Indralaya
    "PT. BUNDA MEDIKA JAKABARING": (-3.0345645, 104.7933477, "PHOTON"), # RS Bunda Medika Jakabaring
    "RSIA KIM ( PT. Karunia Indah Medika )": (-3.6297378, 103.7508829, "PHOTON"),  # Muara Enim
    "PT KARUNIA INDAH MEDIKA": (-3.6297378, 103.7508829, "PHOTON"),     # Muara Enim
    "RSIA ANANDA / NURUSYIFA": (-3.2778012, 102.8970802, "PHOTON"),     # Jl. Yos Sudarso, Lubuk Linggau
    "PT. Mega Dwi Sari ( RSIA Dwi Sari )": (-3.2861345, 102.8823737, "PHOTON"),  # Jl. Yos Sudarso No.2, Lubuk Linggau
    # Batch 2 (19 Agu 2026): target realistis dari websearch/lewatmana/medicastore
    "PT. PERTAMINA BINA MEDIKA IHC": (-3.4349410, 104.2262160, "LEWATMANA"),  # RS Pertamina Prabumulih, Jl. Kesehatan No.100
    "PT.RIKA AMELIA (RSIA RIKA AMELIA)": (-2.9394118, 104.7000131, "PHOTON"),  # kel. Alang-Alang Lebar, Jl. SMB II KM 11.5
    "PT. FADHILAH MEDICA CENTRA": (-3.4439950, 104.2090060, "LEWATMANA"),  # RS Fadhilah, Jl. Jend. Sudirman, Patih Galung
    "RS AR Bunda Prabumulih": (-3.4095158, 104.2545864, "GOOGLE_MAPS"),  # Jl. Angkatan 45 No.222, Prabumulih Timur
    "RSUD SUNGAI LILIN": (-2.5411579, 104.1121393, "MEDICASTORE"),  # Jl. Palembang-Jambi KM 117, Musi Banyuasin
    "RSK GIGI DAN MULUT PROV. SUMSEL": (-2.9494370, 104.7344650, "PHOTON"),  # Jl. Kol. H. Burlian KM 6 (sama dgn RS Siti Fatimah)
    "RS AR Bunda Lubuklinggau / AR. Muhamad": (-3.2989670, 102.8624272, "PHOTON"),  # kel. Bandung Kiri, Jl. Garuda No.245
    # Batch 3 (19 Agu 2026): idalamat JSON-LD (GeoCoordinates)
    "PT. PINTU ILMU (RS. AB AZ ZAHRA)": (-2.9422889, 104.7843368, "IDALAMAT"),  # Jl. Brigjen Hasan Kasim No.1-2, Bukit Sangkal, Kalidoni
    "DINAS KESEHATAN KAB.MUSI BANYUASIN": (-2.8865755, 103.8387367, "IDALAMAT"),  # Jl. Kol. Wahid Udin No.230, Serasan Jaya, Sekayu
    "DINAS KESEHATAN KAB. LAHAT": (-3.7755329, 103.5467937, "IDALAMAT"),  # Jl. Bhayangkara, Bandar Jaya, Lahat
}
for name, (lat, lon, src) in EXTRA.items():
    m = res["pelanggan_nama"] == name
    if m.any() and res.loc[m, "status"].iloc[0] in ("CITY_LEVEL", "FOUND_GEO", "FOUND_SIRS"):
        res.loc[m, ["lat", "lon", "sumber", "status"]] = [lat, lon, src, "FOUND_GEO"]
        print(f"EXTRA OK: {name} -> {lat}, {lon} ({src})")

# Koreksi region yang salah di data (bukti kualitas data utk paper)
REGION_FIX = {
    "RSUD SUNGAI LILIN": "Musi Banyuasin",  # terlabel "Palembang" di data, padahal Kec. Sungai Lilin, Musi Banyuasin
}
for name, reg in REGION_FIX.items():
    m = res["pelanggan_nama"] == name
    if m.any():
        print(f"REGION FIX: {name}: {res.loc[m, 'region'].iloc[0]} -> {reg}")
        res.loc[m, "region"] = reg

# Validasi ulang semua FOUND_GEO dengan pusat region koreksi (<= 60 km)
geo_mask = res["status"] == "FOUND_GEO"
for i, row in res[geo_mask].iterrows():
    rc = REGION_CENTER.get(row["region"])
    if rc is None:
        continue
    d = haversine(row["lat"], row["lon"], rc[0], rc[1])
    if d > 60:
        print(f"  JAUH ({d:.0f} km): {row['pelanggan_nama']} [{row['region']}] -> CITY_LEVEL")
        res.loc[i, "status"] = "CITY_LEVEL"
        res.loc[i, "sumber"] = "CITY_LEVEL"
        res.loc[i, "lat"] = rc[0]
        res.loc[i, "lon"] = rc[1]

# FOUND_SIRS dengan lat kosong -> CITY_LEVEL
sirs_nolat = (res["status"] == "FOUND_SIRS") & (res["lat"].isna())
for i, row in res[sirs_nolat].iterrows():
    rc = REGION_CENTER.get(row["region"])
    if rc:
        res.loc[i, ["lat", "lon", "sumber", "status"]] = [rc[0], rc[1], "CITY_LEVEL", "CITY_LEVEL"]

# Jarak vs data asli
res["dist_data_km"] = res.apply(
    lambda r: haversine(r["lat"], r["lon"], r["lat_data"], r["lon_data"]) if pd.notna(r["lat"]) else None,
    axis=1)

res.to_csv(OUT, index=False)

# Not found list
nf = res[res["status"] == "CITY_LEVEL"].copy()
nf = nf.sort_values("n_inv", ascending=False)
nf[["pelanggan_nama", "region", "n_inv", "lat", "lon", "sumber", "status"]].to_csv(OUT_NF, index=False)

print("\n=== Status final ===")
print(res["status"].value_counts().to_string())
print(f"\nTotal dengan koordinat aktual: {(res['status'] != 'CITY_LEVEL').sum()}/{len(res)}")
print(f"Faktur tercakup koordinat aktual: {res[res['status'] != 'CITY_LEVEL']['n_inv'].sum()}/{res['n_inv'].sum()} ({100*res[res['status'] != 'CITY_LEVEL']['n_inv'].sum()/res['n_inv'].sum():.1f}%)")
d = res["dist_data_km"].dropna()
print(f"\nJarak koordinat aktual vs data (km): n={len(d)} median={d.median():.1f} mean={d.mean():.1f} p90={d.quantile(0.9):.1f} max={d.max():.1f}")
print("\n=== CITY_LEVEL tersisa (n_inv >= 10) ===")
print(res[(res["status"] == "CITY_LEVEL") & (res["n_inv"] >= 10)].sort_values("n_inv", ascending=False)[["pelanggan_nama", "region", "n_inv"]].to_string())