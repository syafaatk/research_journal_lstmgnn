# -*- coding: utf-8 -*-
"""
1. Fix false positive: TK.IV/DKT (militer) dan instansi non-RS (BPBD, DINAS, DINKES, BADAN)
   tidak boleh match ke RS.
2. Coba varian query Photon untuk RS penting yang masih CITY_LEVEL.
Output: skrip/customers_geocoded_final2.csv
"""
import re
import time
import json
import requests
import pandas as pd

IN = r"E:\Download\Jurnal\skrip\customers_geocoded_final.csv"
OUT = r"E:\Download\Jurnal\skrip\customers_geocoded_final2.csv"
CACHE = r"E:\Download\Jurnal\skrip\photon_cache.json"
HEADERS = {"User-Agent": "zara-research/1.0 (research contact)"}
BBOX = {"lat_min": -5.5, "lat_max": -1.5, "lon_min": 102.0, "lon_max": 106.0}

def in_bbox(lat, lon):
    return lat is not None and lon is not None and BBOX["lat_min"] <= lat <= BBOX["lat_max"] and BBOX["lon_min"] <= lon <= BBOX["lon_max"]

def photon(q, cache):
    if q in cache:
        v = cache[q]
        return (v[0], v[1], v[2], v[3]) if len(v) >= 4 else (None, None, "", "")
    for attempt in range(3):
        try:
            r = requests.get("https://photon.komoot.io/api/", params={"q": q, "limit": 5},
                             headers=HEADERS, timeout=30)
            if r.status_code == 200:
                for f in r.json().get("features", []):
                    lon, lat = f["geometry"]["coordinates"]
                    if in_bbox(lat, lon):
                        p = f["properties"]
                        cache[q] = (lat, lon, p.get("name", ""), p.get("city", ""))
                        return (lat, lon, p.get("name", ""), p.get("city", ""))
                cache[q] = (None, None, "", "")
                return (None, None, "", "")
            time.sleep(2)
        except Exception:
            time.sleep(2)
    cache[q] = (None, None, "", "")
    return (None, None, "", "")

res = pd.read_csv(IN)

# 1. Fix false positive
non_rs = res["pelanggan_nama"].str.upper().str.contains("TK\.IV|DKT|BPBD|BADAN|DINAS|DINKES|PEMKAB|PEMPROV", na=False, regex=True)
fix = non_rs & (res["status"] == "FOUND_SIRS")
print(f"Fix non-RS -> RS: {fix.sum()}")
for i, row in res[fix].iterrows():
    rc = res[res["region"] == row["region"]][["lat", "lon"]].mean()
    res.loc[i, "status"] = "CITY_LEVEL"
    res.loc[i, "sumber"] = "CITY_LEVEL"
    res.loc[i, "nama_sirs"] = None
    res.loc[i, "lat"] = rc["lat"]
    res.loc[i, "lon"] = rc["lon"]

# 2. Coba varian query untuk RS penting yang CITY_LEVEL
try:
    with open(CACHE, "r", encoding="utf-8") as f:
        cache = json.load(f)
except Exception:
    cache = {}

# daftar RS penting (n_inv >= 10) yang masih CITY_LEVEL
targets = res[(res["status"] == "CITY_LEVEL") & (res["n_inv"] >= 10)].copy()
variants = {
    "RSUD. SEKAYU": ["Rumah Sakit Sekayu", "RSUD Sekayu Musi Banyuasin", "RS Sekayu"],
    "RS. AT TAQWA GUMAWANG": ["RS At Taqwa Gumawang", "Rumah Sakit At Taqwa OKU Timur", "At Taqwa Gumawang"],
    "RSUD dr. H. M.Rabain": ["RSUD Rabain Muara Enim", "Rumah Sakit Rabain Muara Enim", "RS Rabain"],
    "RSUD Dr. Ibnu Sutowo": ["RSUD Baturaja", "Rumah Sakit Ibnu Sutowo Baturaja", "RS Ibnu Sutowo"],
    "RS Dr. Sobirin Kabupaten Musirawas": ["RS Sobirin Lubuk Linggau", "Rumah Sakit Sobirin Musi Rawas", "RS Dr Sobirin"],
    "RS. PB. Charitas Belitang": ["Charitas Belitang", "RS Charitas Belitang OKU Timur"],
    "PT. LINTANG LAKSANA UTAMA (RS.SILOAM LINGGAU)": ["RS Siloam Silampari Lubuk Linggau", "Siloam Hospitals Lubuk Linggau"],
    "RS TK.IV 02.07.05 Dr. Noesmir": ["RS Noesmir Baturaja", "Rumah Sakit Noesmir Baturaja"],
    "RSUD. Rupit": ["RSUD Rupit Musi Rawas Utara", "Rumah Sakit Rupit"],
    "RSUD SITI AISYAH": ["RSUD Siti Aisyah Lubuk Linggau", "Rumah Sakit Siti Aisyah Lubuk Linggau"],
    "RSUD KAB. EMPAT LAWANG": ["RSUD Empat Lawang", "Rumah Sakit Empat Lawang Tebing Tinggi"],
    "RS AR Bunda Prabumulih": ["RS AR Bunda Prabumulih", "Rumah Sakit AR Bunda Prabumulih"],
    "RSUD SUNGAI LILIN": ["RSUD Sungai Lilin", "Rumah Sakit Sungai Lilin Musi Banyuasin"],
    "RSUD GANDUS": ["RSUD Gandus Palembang", "Rumah Sakit Gandus Palembang"],
    "RSIA KIM ( PT. Karunia Indah Medika )": ["RSIA Karunia Indah Medika Muara Enim", "RS Karunia Indah Medika"],
    "RSIA ANANDA / NURUSYIFA": ["RSIA Ananda Lubuk Linggau", "Rumah Sakit Ananda Lubuk Linggau"],
    "PT. MUTIARA CAHAYA LESARI MEDIKA (RS. Tiara Fatrin)": ["RS Tiara Fatrin Palembang", "Rumah Sakit Tiara Fatrin"],
    "RS AR Bunda Lubuklinggau / AR. Muhamad": ["RS AR Bunda Lubuk Linggau", "Rumah Sakit AR Bunda Lubuklinggau"],
    "RUMAH SAKIT DR ERNALDI PROVINSI SUMSEL": ["RS Ernaldi Bahar Palembang", "Rumah Sakit Jiwa Ernaldi Bahar"],
    "RSK GIGI DAN MULUT PROV. SUMSEL": ["RSGM Palembang", "Rumah Sakit Gigi Mulut Palembang"],
}

found_new = 0
for _, row in targets.iterrows():
    name = row["pelanggan_nama"]
    if name not in variants:
        continue
    for v in variants[name]:
        lat, lon, gn, gc = photon(v, cache)
        if in_bbox(lat, lon):
            res.loc[res["pelanggan_nama"] == name, ["lat", "lon", "sumber", "status"]] = [lat, lon, "PHOTON", "FOUND_GEO"]
            print(f"  OK: {name} -> {lat}, {lon} (via '{v}')")
            found_new += 1
            break
        time.sleep(0.6)

with open(CACHE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False)

# 3. Validasi jarak FOUND_GEO baru ke pusat region (<= 60 km)
import math
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

region_center = res.groupby("region")[["lat", "lon"]].mean()
geo_mask = res["status"] == "FOUND_GEO"
for i, row in res[geo_mask].iterrows():
    rc = region_center.loc[row["region"]]
    d = haversine(row["lat"], row["lon"], rc["lat"], rc["lon"])
    if d > 60:
        print(f"  JAUH ({d:.0f} km): {row['pelanggan_nama']} [{row['region']}] -> CITY_LEVEL")
        res.loc[i, "status"] = "CITY_LEVEL"
        res.loc[i, "sumber"] = "CITY_LEVEL"
        res.loc[i, "lat"] = rc["lat"]
        res.loc[i, "lon"] = rc["lon"]

res.to_csv(OUT, index=False)
print(f"\nKoordinat baru ditemukan: {found_new}")
print("\n=== Status final ===")
print(res["status"].value_counts().to_string())
print("\n=== CITY_LEVEL tersisa (n_inv >= 10) ===")
print(res[(res["status"] == "CITY_LEVEL") & (res["n_inv"] >= 10)].sort_values("n_inv", ascending=False)[["pelanggan_nama", "region", "n_inv"]].to_string())