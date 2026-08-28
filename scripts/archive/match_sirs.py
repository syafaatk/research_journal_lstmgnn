# -*- coding: utf-8 -*-
"""
Match pelanggan (customers_coords.csv) dengan data SIRS Kemenkes (lat/lon).
Output: skrip/customers_geocoded.csv dengan kolom status.
"""
import re
import pandas as pd

SIRS = r"C:\Users\ADMLAY~1\AppData\Local\Temp\opencode\sirs_ml.csv"
CUST = r"E:\Download\Jurnal\skrip\customers_coords.csv"
OUT = r"E:\Download\Jurnal\skrip\customers_geocoded.csv"

def norm(s):
    """Normalisasi nama untuk matching."""
    s = str(s).upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\b(RS|RSUD|RSUP|RUMKIT|RUMAH|SAKIT|KAB|KABUPATEN|KOTA|PROV|PROVINSI|DR|DOKTER|PT|CV|YAY|YAYASAN|TK|IV|II|III|DKT|RSIA|RSB|RSU|RSK|RSI)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Pelanggan kita
cust = pd.read_csv(CUST, index_col=0)
cust = cust.reset_index()
cust["key"] = cust["pelanggan_nama"].apply(norm)

# Data SIRS Sumsel
sirs = pd.read_csv(SIRS, low_memory=False)
sirs = sirs[sirs["provinsi"].str.upper().str.contains("SUMATERA SELATAN", na=False)].copy()
print(f"RS Sumsel di SIRS: {len(sirs)}")
sirs["key"] = sirs["nama_rs"].apply(norm)

# Match: exact key, lalu contains (satu arah)
matched = []
for _, c in cust.iterrows():
    hit = sirs[sirs["key"] == c["key"]]
    if len(hit) == 0:
        hit = sirs[sirs["key"].str.contains(c["key"], regex=False, na=False)] if len(c["key"]) > 4 else sirs.iloc[0:0]
    if len(hit) == 0:
        hit = sirs[sirs["key"].str.contains(hit_key, regex=False, na=False)] if False else hit
    if len(hit) > 0:
        r = hit.iloc[0]
        matched.append({
            "pelanggan_nama": c["pelanggan_nama"],
            "region": c["region"],
            "n_inv": c["n_inv"],
            "lat_data": c["lat"],
            "lon_data": c["lon"],
            "lat_sirs": r["lat"],
            "lon_sirs": r["lon"],
            "nama_sirs": r["nama_rs"],
            "kabkota_sirs": r["kabkota"],
            "status": "FOUND_SIRS",
        })
    else:
        matched.append({
            "pelanggan_nama": c["pelanggan_nama"],
            "region": c["region"],
            "n_inv": c["n_inv"],
            "lat_data": c["lat"],
            "lon_data": c["lon"],
            "lat_sirs": None, "lon_sirs": None,
            "nama_sirs": None, "kabkota_sirs": None,
            "status": "NOT_FOUND",
        })

res = pd.DataFrame(matched)
res.to_csv(OUT, index=False)
print(f"\nTotal: {len(res)} | FOUND: {(res['status']=='FOUND_SIRS').sum()} | NOT_FOUND: {(res['status']=='NOT_FOUND').sum()}")
print("\n=== FOUND (top 40 by n_inv) ===")
print(res[res["status"] == "FOUND_SIRS"].sort_values("n_inv", ascending=False).head(40).to_string())
print("\n=== NOT_FOUND (top 40 by n_inv) ===")
print(res[res["status"] == "NOT_FOUND"].sort_values("n_inv", ascending=False).head(40).to_string())