# -*- coding: utf-8 -*-
"""
Geocode RS Sumsel dari SIRS (via alamat, Photon/komoot) lalu match pelanggan.
Output:
  skrip/sirs_sumsel_geocoded.csv  - 90 RS Sumsel + lat/lon
  skrip/customers_geocoded.csv    - 156 pelanggan + lat/lon + sumber
"""
import re
import time
import json
import requests
import pandas as pd

SIRS = r"C:\Users\ADMLAY~1\AppData\Local\Temp\opencode\sirs_ml.csv"
CUST = r"E:\Download\Jurnal\skrip\customers_coords.csv"
OUT_SIRS = r"E:\Download\Jurnal\skrip\sirs_sumsel_geocoded.csv"
OUT_CUST = r"E:\Download\Jurnal\skrip\customers_geocoded.csv"
CACHE = r"E:\Download\Jurnal\skrip\photon_cache.json"

HEADERS = {"User-Agent": "zara-research/1.0 (research contact)"}


def norm(s):
    s = str(s).upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\b(RS|RSUD|RSUP|RUMKIT|RUMAH|SAKIT|KAB|KABUPATEN|KOTA|PROV|PROVINSI|DR|DOKTER|PT|CV|YAY|YAYASAN|TK|IV|II|III|DKT|RSIA|RSB|RSU|RSK|RSI|UMUM|DAERAH|PUSAT|ISLAM|IBU|ANAK|KHUSUS)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def photon(q, cache):
    if q in cache:
        return cache[q]
    for attempt in range(3):
        try:
            r = requests.get("https://photon.komoot.io/api/", params={"q": q, "limit": 1},
                             headers=HEADERS, timeout=30)
            if r.status_code == 200:
                res = r.json()
                if res.get("features"):
                    f = res["features"][0]
                    lon, lat = f["geometry"]["coordinates"]
                    props = f["properties"]
                    cache[q] = (lat, lon, props.get("name", ""), props.get("city", ""), props.get("state", ""))
                else:
                    cache[q] = (None, None, "", "", "")
                return cache[q]
            time.sleep(2)
        except Exception:
            time.sleep(2)
    cache[q] = (None, None, "", "", "")
    return cache[q]


# ---------- 1. Geocode RS Sumsel via alamat ----------
df = pd.read_csv(SIRS, low_memory=False)
ss = df[df["provinsi"].str.upper().str.contains("SUMATERA SELATAN", na=False)].copy()
print(f"RS Sumsel: {len(ss)}")

try:
    with open(CACHE, "r", encoding="utf-8") as f:
        cache = json.load(f)
except Exception:
    cache = {}

rows = []
for _, r in ss.iterrows():
    addr = str(r["alamat"]).replace("\n", " ").strip()
    q = f"{addr}, {r['kabkota']}, Sumatera Selatan"
    lat, lon, name, city, state = photon(q, cache)
    rows.append({"kode_rs": r["kode_rs"], "nama_rs": r["nama_rs"], "kabkota": r["kabkota"],
                 "alamat": addr, "lat": lat, "lon": lon, "geo_name": name, "geo_city": city})
    time.sleep(1.0)

sirs_geo = pd.DataFrame(rows)
sirs_geo.to_csv(OUT_SIRS, index=False)
with open(CACHE, "w", encoding="utf-8") as f:
    json.dump(cache, f, ensure_ascii=False)

print(f"Geocode RS: {sirs_geo['lat'].notna().sum()}/{len(sirs_geo)} berhasil")
print(sirs_geo[sirs_geo["lat"].isna()][["nama_rs", "kabkota"]].to_string())

# ---------- 2. Match pelanggan ke RS ----------
cust = pd.read_csv(CUST, index_col=0).reset_index()
cust["key"] = cust["pelanggan_nama"].apply(norm)
sirs_geo["key"] = sirs_geo["nama_rs"].apply(norm)

matched = []
for _, c in cust.iterrows():
    hit = sirs_geo[sirs_geo["key"] == c["key"]]
    if len(hit) == 0 and len(c["key"]) > 4:
        hit = sirs_geo[sirs_geo["key"].str.contains(c["key"], regex=False, na=False)]
    if len(hit) == 0 and len(c["key"]) > 4:
        hit = sirs_geo[sirs_geo["key"].apply(lambda k: c["key"] in k if isinstance(k, str) else False)]
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
res.to_csv(OUT_CUST, index=False)
print(f"\nMatch SIRS: {len(res)} total | FOUND_SIRS: {(res['status']=='FOUND_SIRS').sum()} | PENDING: {(res['status']=='PENDING_GEO').sum()}")
print("\n=== PENDING (top 30 by n_inv) ===")
print(res[res["status"] == "PENDING_GEO"].sort_values("n_inv", ascending=False).head(30)[["pelanggan_nama", "region", "n_inv"]].to_string())