# -*- coding: utf-8 -*-
"""
Perbaikan kualitas koordinat pelanggan:
1. Filter Dinas/DINKES dari match RS (false positive).
2. Validasi FOUND_GEO: jarak ke pusat region <= 60 km, kalau jauh -> CITY_LEVEL.
3. Hitung jarak koordinat aktual vs koordinat data (level kota) untuk laporan A1.
Output: skrip/customers_geocoded_final.csv
"""
import math
import pandas as pd

IN = r"E:\Download\Jurnal\skrip\customers_geocoded_v2.csv"
CUST = r"E:\Download\Jurnal\skrip\customers_coords.csv"
OUT = r"E:\Download\Jurnal\skrip\customers_geocoded_final.csv"

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

res = pd.read_csv(IN)
orig = pd.read_csv(CUST, index_col=0).reset_index()
res = res.merge(orig[["pelanggan_nama", "lat", "lon"]].rename(columns={"lat": "lat_data", "lon": "lon_data"}),
                on="pelanggan_nama", how="left")

# 1. Fix false positive: pelanggan Dinas/DINKES tidak boleh match ke RS
dinas_mask = res["pelanggan_nama"].str.upper().str.contains("DINAS|DINKES", na=False)
rs_match_mask = res["status"] == "FOUND_SIRS"
fix = dinas_mask & rs_match_mask
print(f"Fix Dinas->RS: {fix.sum()} pelanggan")
res.loc[fix, "status"] = "CITY_LEVEL"
res.loc[fix, "sumber"] = "CITY_LEVEL"
res.loc[fix, "nama_sirs"] = None
res.loc[fix, "kabkota_sirs"] = None

# 2. Validasi jarak FOUND_GEO ke pusat region (mean pelanggan region dari data asli)
region_center = res.groupby("region")[["lat", "lon"]].mean()
geo_mask = res["status"] == "FOUND_GEO"
for i, row in res[geo_mask].iterrows():
    rc = region_center.loc[row["region"]]
    d = haversine(row["lat"], row["lon"], rc["lat"], rc["lon"])
    if d > 60:
        print(f"  JAUH ({d:.0f} km): {row['pelanggan_nama']} [{row['region']}] -> fallback CITY_LEVEL")
        res.loc[i, "status"] = "CITY_LEVEL"
        res.loc[i, "sumber"] = "CITY_LEVEL"
        res.loc[i, "lat"] = rc["lat"]
        res.loc[i, "lon"] = rc["lon"]

# 3. FOUND_SIRS dengan lat kosong -> CITY_LEVEL (koordinat RS tidak ditemukan)
sirs_nolat = (res["status"] == "FOUND_SIRS") & (res["lat"].isna())
print(f"FOUND_SIRS tanpa koordinat: {sirs_nolat.sum()} -> CITY_LEVEL")
for i, row in res[sirs_nolat].iterrows():
    rc = region_center.loc[row["region"]]
    res.loc[i, "status"] = "CITY_LEVEL"
    res.loc[i, "sumber"] = "CITY_LEVEL"
    res.loc[i, "lat"] = rc["lat"]
    res.loc[i, "lon"] = rc["lon"]

# 4. Hitung jarak koordinat final vs koordinat data (level kota)
res["dist_data_km"] = res.apply(
    lambda r: haversine(r["lat"], r["lon"], r["lat_data"], r["lon_data"]) if pd.notna(r["lat"]) else None,
    axis=1)

res.to_csv(OUT, index=False)
print("\n=== Ringkasan status final ===")
print(res["status"].value_counts().to_string())
print("\n=== Jarak koordinat aktual vs data (km) ===")
d = res["dist_data_km"].dropna()
print(f"n={len(d)} | median={d.median():.1f} | mean={d.mean():.1f} | p90={d.quantile(0.9):.1f} | max={d.max():.1f}")
print("\n=== FOUND_SIRS (dengan koordinat) ===")
print(res[res["status"] == "FOUND_SIRS"][["pelanggan_nama", "region", "n_inv", "lat", "lon", "nama_sirs"]].to_string())
print("\n=== CITY_LEVEL (top 40 by n_inv) ===")
print(res[res["status"] == "CITY_LEVEL"].sort_values("n_inv", ascending=False).head(40)[["pelanggan_nama", "region", "n_inv"]].to_string())