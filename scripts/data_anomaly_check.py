# -*- coding: utf-8 -*-
"""Diagnostik anomali data penjualan alkes.

Memindai kemungkinan kesalahan pengukuran / pencatatan yang bisa mengubah
kesimpulan analisis. Output: results/data_anomaly_check.json
"""
import json
import os

import numpy as np
import pandas as pd

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(BASE, "results", "data_anomaly_check.json")

df = pd.read_excel(DATA)
n_raw = len(df)

# --- Normalisasi kolom numerik ---
for c in ["jumlah", "d_jual_qty", "d_jual_diskon", "jual_pajak", "jual_total_pajak",
          "jual_total_fak", "d_jual_barang_harjul", "d_barang_harga_total"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")

report = {}
report["n_baris_raw"] = int(n_raw)
report["n_faktur_unik"] = int(df["d_jual_nofak"].nunique())
report["n_pelanggan_unik"] = int(df["pelanggan_id"].nunique()) if "pelanggan_id" in df.columns else None
report["rentang_tanggal"] = [str(df["tanggal"].min()), str(df["tanggal"].max())]

# ============================================================
# 1. Harga per qty ekstrem (error input)
# ============================================================
h = df["jumlah"] / df["d_jual_qty"].replace(0, np.nan)
df["harga_per_qty"] = h
extreme = df[(df["harga_per_qty"] < 100) | (df["harga_per_qty"] > 50_000_000)]
report["harga_per_qty"] = {
    "n_baris_harga_<100": int((df["harga_per_qty"] < 100).sum()),
    "n_baris_harga_>50jt": int((df["harga_per_qty"] > 50_000_000).sum()),
    "n_baris_harga_<=1": int((df["harga_per_qty"] <= 1).sum()),
    "contoh_harga_<=1": extreme[extreme["harga_per_qty"] <= 1][
        ["barang_nama", "barang_spesifikasi", "d_jual_qty", "jumlah", "harga_per_qty", "tanggal"]]
        .head(10).to_dict(orient="records"),
}

# ============================================================
# 2. Qty negatif / retur
# ============================================================
neg_qty = df[df["d_jual_qty"] < 0]
report["qty_negatif"] = {
    "n_baris": int(len(neg_qty)),
    "n_faktur": int(neg_qty["d_jual_nofak"].nunique()),
    "total_qty_negatif": float(neg_qty["d_jual_qty"].sum()),
    "contoh": neg_qty[["d_jual_nofak", "barang_nama", "d_jual_qty", "jumlah", "tanggal"]]
        .head(10).to_dict(orient="records"),
}

# ============================================================
# 3. Baris jumlah=0 tapi qty>0 (sample/gratis)
# ============================================================
zero_val = df[(df["jumlah"] == 0) & (df["d_jual_qty"] > 0)]
report["jumlah_nol_qty_positif"] = {
    "n_baris": int(len(zero_val)),
    "n_faktur": int(zero_val["d_jual_nofak"].nunique()),
    "total_qty": float(zero_val["d_jual_qty"].sum()),
    "contoh": zero_val[["d_jual_nofak", "barang_nama", "d_jual_qty", "jumlah", "tanggal"]]
        .head(10).to_dict(orient="records"),
}

# ============================================================
# 4. Duplikasi faktur (nofak ganda)
# ============================================================
dup = df[df.duplicated("d_jual_nofak", keep=False)]
report["duplikasi_nofak"] = {
    "n_baris_duplikat": int(len(dup)),
    "n_faktur_terlibat": int(dup["d_jual_nofak"].nunique()),
    "contoh": dup[["d_jual_nofak", "tanggal", "pelanggan_nama", "jumlah"]]
        .assign(nofak_str=dup["d_jual_nofak"].astype(str))
        .sort_values("nofak_str").drop(columns="nofak_str").head(10).to_dict(orient="records"),
}

# ============================================================
# 5. Konsistensi satuan untuk produk kunci (pola masker)
# ============================================================
KEYWORDS = {
    "handscoon": "handscoon|sarung tangan|glove",
    "spuit": "spuit|syringe|syiringe",
    "infusion_set": "infusion set|infus set",
    "iv_catheter": "iv catheter|iv canula|catheter iv",
    "masker": "masker|n95|kn95|duckbill|face mask",
    "kassa": "kassa|gauze",
    "plester": "plester|plaster",
}
satuan_report = {}
for key, pat in KEYWORDS.items():
    sub = df[df["barang_nama"].str.contains(pat, case=False, na=False)]
    if len(sub) == 0:
        satuan_report[key] = {"n_baris": 0, "satuan": []}
        continue
    sats = (sub.groupby("satuan_nama").agg(
        n_baris=("jumlah", "count"),
        total_qty=("d_jual_qty", "sum"),
        total_nilai_jt=("jumlah", lambda x: round(x.sum() / 1e6, 1)),
    ).reset_index().rename(columns={"satuan_nama": "satuan"}).sort_values("n_baris", ascending=False))
    satuan_report[key] = {
        "n_baris": int(len(sub)),
        "n_satuan": int(sub["satuan_nama"].nunique()),
        "satuan": sats.to_dict(orient="records"),
    }
report["konsistensi_satuan_produk_kunci"] = satuan_report

# ============================================================
# 6. Barang sama nama berbeda (typo/casing/spasi) - potensi pecah
# ============================================================
norm = df["barang_nama"].str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
report["barang_nama_normalisasi"] = {
    "n_nama_asli": int(df["barang_nama"].nunique()),
    "n_nama_ternormalisasi": int(norm.nunique()),
    "selisih_pecah": int(df["barang_nama"].nunique() - norm.nunique()),
}

# ============================================================
# 7. Nilai gross vs net (pajak/diskon) - konsistensi jumlah vs total faktur
# ============================================================
if "jual_total_fak" in df.columns:
    # jumlah per baris vs total faktur (per baris item)
    diff = (df["jumlah"] - df["jual_total_fak"]).abs()
    report["jumlah_vs_total_fak"] = {
        "n_baris_selisih_>0": int((diff > 1).sum()),
        "pct_baris_selisih_>0": round(float((diff > 1).mean()) * 100, 2),
        "contoh": df[diff > 1][["d_jual_nofak", "barang_nama", "jumlah", "jual_total_fak"]]
            .head(10).to_dict(orient="records"),
    }

# ============================================================
# 8. Tanggal di luar rentang wajar / null
# ============================================================
report["tanggal"] = {
    "n_null": int(df["tanggal"].isna().sum()),
    "n_sebelum_2020": int((df["tanggal"] < "2020-01-01").sum()),
    "n_setelah_2025_12": int((df["tanggal"] > "2025-12-31").sum()),
}

# ============================================================
# 9. Wilayah vs koordinat (konsistensi geocoding)
# ============================================================
if "latitude" in df.columns and "longitude" in df.columns:
    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")
    report["koordinat"] = {
        "n_lat_null": int(lat.isna().sum()),
        "n_lon_null": int(lon.isna().sum()),
        "n_lat_di_luar_sumsel": int(((lat < -4.5) | (lat > -2.0)).sum()),
        "n_lon_di_luar_sumsel": int(((lon < 102.0) | (lon > 106.0)).sum()),
    }

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False, default=str)

# ============================================================
# Cetak ringkasan
# ============================================================
print("=" * 70)
print("DIAGNOSTIK ANOMALI DATA PENJUALAN ALKES")
print("=" * 70)
print(f"Baris mentah: {n_raw:,} | Faktur unik: {report['n_faktur_unik']:,} | Pelanggan: {report['n_pelanggan_unik']:,}")
print(f"Rentang tanggal: {report['rentang_tanggal'][0]} s.d. {report['rentang_tanggal'][1]}")

print("\n[1] HARGA PER QTY EKSTREM")
hp = report["harga_per_qty"]
print(f"  harga <= 1 rupiah : {hp['n_baris_harga_<=1']:,} baris")
print(f"  harga < 100        : {hp['n_baris_harga_<100']:,} baris")
print(f"  harga > 50 juta    : {hp['n_baris_harga_>50jt']:,} baris")

print("\n[2] QTY NEGATIF / RETUR")
qn = report["qty_negatif"]
print(f"  {qn['n_baris']:,} baris, {qn['n_faktur']:,} faktur, total qty negatif {qn['total_qty_negatif']:,.0f}")

print("\n[3] JUMLAH=0 TAPI QTY>0 (sample/gratis)")
zv = report["jumlah_nol_qty_positif"]
print(f"  {zv['n_baris']:,} baris, {zv['n_faktur']:,} faktur, total qty {zv['total_qty']:,.0f}")

print("\n[4] DUPLIKASI NOFAK")
du = report["duplikasi_nofak"]
print(f"  {du['n_baris_duplikat']:,} baris duplikat, {du['n_faktur_terlibat']:,} faktur terlibat")

print("\n[5] KONSISTENSI SATUAN PRODUK KUNCI")
for key, v in satuan_report.items():
    if v["n_baris"] == 0:
        print(f"  {key:<14}: 0 baris")
        continue
    sats = ", ".join(f"{s['satuan']}({s['n_baris']})" for s in v["satuan"])
    print(f"  {key:<14}: {v['n_baris']:,} baris, {v['n_satuan']} satuan -> {sats}")

print("\n[6] NAMA BARANG NORMALISASI")
nb = report["barang_nama_normalisasi"]
print(f"  nama asli {nb['n_nama_asli']:,} -> ternormalisasi {nb['n_nama_ternormalisasi']:,} (selisih {nb['selisih_pecah']:,})")

print("\n[7] JUMLAH vs TOTAL FAKTUR")
if "jumlah_vs_total_fak" in report:
    jv = report["jumlah_vs_total_fak"]
    print(f"  {jv['n_baris_selisih_>0']:,} baris ({jv['pct_baris_selisih_>0']}%) selisih > 1")

print("\n[8] TANGGAL")
tg = report["tanggal"]
print(f"  null: {tg['n_null']:,} | sebelum 2020: {tg['n_sebelum_2020']:,} | setelah 2025: {tg['n_setelah_2025_12']:,}")

print("\n[9] KOORDINAT")
if "koordinat" in report:
    k = report["koordinat"]
    print(f"  lat null: {k['n_lat_null']:,} | lon null: {k['n_lon_null']:,} | lat luar sumsel: {k['n_lat_di_luar_sumsel']:,} | lon luar sumsel: {k['n_lon_di_luar_sumsel']:,}")

print("\nSelesai. Disimpan ke", OUT)
