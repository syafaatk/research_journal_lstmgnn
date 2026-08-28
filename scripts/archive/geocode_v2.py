# -*- coding: utf-8 -*-
"""
Pipeline koordinat pelanggan v2:
1. Geocode 90 RS Sumsel (SIRS) via Photon dengan beberapa varian query + validasi bounding box Sumsel.
2. Match 156 pelanggan ke RS (validasi kabkota, hindari false positive).
3. Geocode pelanggan non-RS (nama + region).
4. Fallback: koordinat level kota (mean pelanggan per region dari data).
Output: skrip/customers_geocoded_v2.csv + skrip/not_found_list.csv
"""
import re
import time
import json
import requests
import pandas as pd

SIRS = r"C:\Users\ADMLAY~1\AppData\Local\Temp\opencode\sirs_ml.csv"
CUST = r"E:\Download\Jurnal\skrip\customers_coords.csv"
OUT = r"E:\Download\Jurnal\skrip\customers_geocoded_v2.csv"
OUT_NF = r"E:\Download\Jurnal\skrip\not_found_list.csv"
CACHE = r"E:\Download\Jurnal\skrip\photon_cache.json"

HEADERS = {"User-Agent": "zara-research/1.0 (research contact)"}
BBOX = {"lat_min": -5.5, "lat_max": -1.5, "lon_min": 102.0, "lon_max": 106.0}


def in_bbox(lat, lon):
    if lat is None or lon is None:
        return False
    return BBOX["lat_min"] <= lat <= BBOX["lat_max"] and BBOX["lon_min"] <= lon <= BBOX["lon_max"]


def norm(s, keep_tokens=False):
    s = str(s).upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    if not keep_tokens:
        s = re.sub(r"\b(RS|RSUD|RSUP|RUMKIT|RUMAH|SAKIT|KAB|KABUPATEN|KOTA|PROV|PROVINSI|DR|DOKTER|PT|CV|YAY|YAYASAN|TK|IV|II|III|DKT|RSIA|RSB|RSU|RSK|RSI|UMUM|DAERAH|PUSAT|ISLAM|IBU|ANAK|KHUSUS)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def photon(q, cache):
    if q in cache:
        return cache[q]
    for attempt in range(3):
        try:
            r = requests.get("https://photon.komoot.io/api/", params={"q": q, "limit": 3},
                             headers=HEADERS, timeout=30)
            if r.status_code == 200:
                res = r.json()
                feats = res.get("features", [])
                # pilih feature pertama yang valid dalam bbox
                for f in feats:
                    lon, lat = f["geometry"]["coordinates"]
                    if in_bbox(lat, lon):
                        props = f["properties"]
                        cache[q] = (lat, lon, props.get("name", ""), props.get("city", ""), props.get("state", ""))
                        return cache[q]
                cache[q] = (None, None, "", "", "")
                return cache[q]
            time.sleep(2)
        except Exception:
            time.sleep(2)
    cache[q] = (None, None, "", "", "")
    return cache[q]


# ---------- 1. Geocode RS Sumsel (multi-variant) ----------
df = pd.read_csv(SIRS, low_memory=False)
ss = df[df["provinsi"].str.upper().str.contains("SUMATERA SELATAN", na=False)].copy()
print(f"RS Sumsel: {len(ss)}")

try:
    with open(CACHE, "r", encoding="utf-8") as f:
        cache = json.load(f)
except Exception:
    cache = {}

rs_rows = []
for _, r in ss.iterrows():
    addr = str(r["alamat"]).replace("\n", " ").strip()
    nama = str(r["nama_rs"]).strip()
    kab = str(r["kabkota"]).strip()
    variants = [
        f"{nama}, {kab}",
        f"{nama}",
        f"{addr}, {kab}",
    ]
    best = (None, None, "", "", "")
    for v in variants:
        lat, lon, gn, gc, gs = photon(v, cache)
        if in_bbox(lat, lon):
            best = (lat, lon, gn, gc, gs)
            break
        time.sleep(0.5)
    rs_rows.append({"kode_rs": r["kode_rs"], "nama_rs": nama, "kabkota": kab,
                    "alamat": addr, "lat": best[0], "lon": best[1],
                    "geo_name": best[2], "geo_city": best[3]})
    time.sleep(0.8)

sirs_geo = pd.DataFrame(rs_rows)
sirs_geo.to_csv(r"E:\Download\Jurnal\skrip\sirs_sumsel_geocoded_v2.csv", index=False)
with open(CACHE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False)

n_ok = sirs_geo["lat"].notna().sum()
print(f"Geocode RS v2: {n_ok}/{len(sirs_geo)} berhasil (dalam bbox)")
print("Gagal:", sirs_geo[sirs_geo["lat"].isna()]["nama_rs"].tolist())

# ---------- 2. Match pelanggan ke RS ----------
cust = pd.read_csv(CUST, index_col=0).reset_index()
cust["key"] = cust["pelanggan_nama"].apply(norm)
sirs_geo["key"] = sirs_geo["nama_rs"].apply(norm)

matched = []
for _, c in cust.iterrows():
    # pass 1: exact key
    hit = sirs_geo[sirs_geo["key"] == c["key"]]
    # pass 2: contains (nama pelanggan ada di nama RS)
    if len(hit) == 0 and len(c["key"]) > 4:
        hit = sirs_geo[sirs_geo["key"].apply(lambda k: c["key"] in k if isinstance(k, str) else False)]
    # pass 3: nama RS ada di nama pelanggan
    if len(hit) == 0 and len(c["key"]) > 4:
        hit = sirs_geo[sirs_geo["key"].apply(lambda k: k in c["key"] if isinstance(k, str) else False)]
    if len(hit) > 0:
        r = hit.iloc[0]
        matched.append({"pelanggan_nama": c["pelanggan_nama"], "region": c["region"], "n_inv": c["n_inv"],
                        "lat": r["lat"], "lon": r["lon"], "nama_sirs": r["nama_rs"], "kabkota_sirs": r["kabkota"],
                        "sumber": "SIRS", "status": "FOUND_SIRS"})
    else:
        matched.append({"pelanggan_nama": c["pelanggan_nama"], "region": c["region"], "n_inv": c["n_inv"],
                        "lat": None, "lon": None, "nama_sirs": None, "kabkota_sirs": None,
                        "sumber": None, "status": "PENDING_GEO"})

res = pd.DataFrame(matched)
print(f"\nMatch SIRS: FOUND_SIRS {(res['status']=='FOUND_SIRS').sum()} | PENDING {(res['status']=='PENDING_GEO').sum()}")

# ---------- 3. Geocode pelanggan non-RS (nama + region) ----------
region_center = cust.groupby("region")[["lat", "lon"]].mean().to_dict("index")
pending = res[res["status"] == "PENDING_GEO"].copy()
for i, row in pending.iterrows():
    q = f"{row['pelanggan_nama']}, {row['region']}, Sumatera Selatan"
    lat, lon, gn, gc, gs = photon(q, cache)
    if in_bbox(lat, lon):
        res.loc[i, "lat"] = lat
        res.loc[i, "lon"] = lon
        res.loc[i, "sumber"] = "PHOTON"
        res.loc[i, "status"] = "FOUND_GEO"
    else:
        # fallback: koordinat level kota (mean pelanggan region)
        rc = region_center.get(row["region"])
        if rc:
            res.loc[i, "lat"] = rc["lat"]
            res.loc[i, "lon"] = rc["lon"]
            res.loc[i, "sumber"] = "CITY_LEVEL"
            res.loc[i, "status"] = "CITY_LEVEL"
        else:
            res.loc[i, "status"] = "NOT_FOUND"
    time.sleep(0.8)

with open(CACHE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False)

res.to_csv(OUT, index=False)
print("\n=== Ringkasan status ===")
print(res["status"].value_counts().to_string())
print("\n=== FOUND_SIRS dengan lat kosong (perlu cek manual) ===")
print(res[(res["status"] == "FOUND_SIRS") & (res["lat"].isna())][["pelanggan_nama", "nama_sirs"]].to_string())
print("\n=== FOUND_GEO (top 30) ===")
print(res[res["status"] == "FOUND_GEO"].sort_values("n_inv", ascending=False).head(30)[["pelanggan_nama", "region", "n_inv", "lat", "lon"]].to_string())
print("\n=== CITY_LEVEL (top 30) ===")
print(res[res["status"] == "CITY_LEVEL"].sort_values("n_inv", ascending=False).head(30)[["pelanggan_nama", "region", "n_inv"]].to_string())