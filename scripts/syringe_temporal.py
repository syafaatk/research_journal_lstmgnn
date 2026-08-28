"""Analisis temporal pengadaan massal syringe (Disposable + Auto Destruct).
Distribusi per bulan-tahun, per pembeli, per region.
Output: results/syringe_temporal.json
"""
import json
import os
from collections import defaultdict
import pandas as pd

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(BASE, "results", "syringe_temporal.json")

SYRINGE_KEYWORDS = ["disposable syringe", "auto destruct syringe"]

df = pd.read_excel(DATA)
# Grain analisis = baris item. Satu faktur bisa berisi banyak baris item syringe
# dengan qty berbeda; JANGAN dedup per faktur (bug lama membuang ~50% volume).
# Nilai PAKAI kolom `jumlah` (nilai level-item = qty x harga satuan). Kolom
# _val berulang di tiap baris item dan TIDAK boleh dijumlahkan
# per baris (inflasi ~3,4x pada subset syringe).
df["_val"] = pd.to_numeric(df["jumlah"], errors="coerce").fillna(0.0)
df = df.dropna(subset=["d_jual_nofak", "tanggal", "barang_nama", "d_jual_qty"])

# Baris item syringe (bukan faktur) - filter nama barang
mask = df["barang_nama"].astype(str).str.lower().str.contains("syringe", na=False)
sy = df[mask].copy()
sy["tanggal"] = pd.to_datetime(sy["tanggal"])
sy["ym"] = sy["tanggal"].dt.to_period("M").astype(str)
sy["year"] = sy["tanggal"].dt.year
sy["month"] = sy["tanggal"].dt.month

# Klasifikasi disposable vs auto destruct
sy["jenis"] = sy["barang_nama"].astype(str).str.lower().apply(
    lambda s: "auto_destruct" if "auto destruct" in s else "disposable")

print("=== Total item syringe (semua qty) ===")
print("Faktur unik:", sy["d_jual_nofak"].nunique())
print("Qty total:", int(sy["d_jual_qty"].sum()))
print("Nilai total: {:.0f}".format(sy["_val"].sum()))
print()
print("Per jenis:")
for j, g in sy.groupby("jenis"):
    print("  {}: faktur={} qty={:,} nilai={:,.0f}".format(
        j, g["d_jual_nofak"].nunique(), int(g["d_jual_qty"].sum()), g["_val"].sum()))

# Fokus qty besar (>= 10.000 unit per baris item) - pengadaan massal
big = sy[sy["d_jual_qty"] >= 10000].copy()
print()
print("=== Pengadaan massal (qty >= 10.000 per item) ===")
print("Faktur unik:", big["d_jual_nofak"].nunique())
print("Qty total: {:,}".format(int(big["d_jual_qty"].sum())))
print("Nilai total: {:,.0f}".format(big["_val"].sum()))

# Distribusi per bulan-tahun
print()
print("=== Distribusi per bulan-tahun (qty, semua syringe) ===")
ym_qty = sy.groupby("ym")["d_jual_qty"].sum().sort_index()
ym_val = sy.groupby("ym")["_val"].sum().sort_index()
for ym in ym_qty.index:
    print("  {}: qty={:>10,} nilai={:>14,.0f}".format(ym, int(ym_qty[ym]), ym_val[ym]))

# Distribusi per bulan-tahun untuk pengadaan massal saja
print()
print("=== Distribusi per bulan-tahun (pengadaan massal >= 10.000) ===")
big_ym = big.groupby("ym")["d_jual_qty"].sum().sort_index()
big_ym_n = big.groupby("ym")["d_jual_nofak"].nunique()
for ym in big_ym.index:
    print("  {}: faktur={:>2} qty={:>10,} nilai={:>14,.0f}".format(
        ym, big_ym_n[ym], int(big_ym[ym]), big.groupby("ym")["_val"].sum()[ym]))

# Per pembeli (pengadaan massal)
print()
print("=== Pembeli pengadaan massal (qty >= 10.000) ===")
cust = big.groupby("pelanggan_nama").agg(
    faktur=("d_jual_nofak", "nunique"), qty=("d_jual_qty", "sum"),
    nilai=("_val", "sum")).sort_values("qty", ascending=False)
for name, row in cust.iterrows():
    print("  {:<45} faktur={:>2} qty={:>10,} nilai={:>14,.0f}".format(
        str(name)[:45], int(row["faktur"]), int(row["qty"]), row["nilai"]))

# Per region (pengadaan massal)
print()
print("=== Region pengadaan massal (qty >= 10.000) ===")
reg = big.groupby("wilayah").agg(
    faktur=("d_jual_nofak", "nunique"), qty=("d_jual_qty", "sum"),
    nilai=("_val", "sum")).sort_values("qty", ascending=False)
for name, row in reg.iterrows():
    print("  {:<30} faktur={:>2} qty={:>10,} nilai={:>14,.0f}".format(
        str(name)[:30], int(row["faktur"]), int(row["qty"]), row["nilai"]))

# Ringkasan per tahun
print()
print("=== Ringkasan per tahun (semua syringe) ===")
yr = sy.groupby("year").agg(faktur=("d_jual_nofak", "nunique"), qty=("d_jual_qty", "sum"),
                            nilai=("_val", "sum"))
for y, row in yr.iterrows():
    print("  {}: faktur={:>3} qty={:>10,} nilai={:>14,.0f}".format(
        int(y), int(row["faktur"]), int(row["qty"]), row["nilai"]))

# Simpan hasil
out = {
    "total_syringe": {"faktur": int(sy["d_jual_nofak"].nunique()),
                      "qty": int(sy["d_jual_qty"].sum()),
                      "nilai": float(sy["_val"].sum())},
    "per_jenis": {j: {"faktur": int(g["d_jual_nofak"].nunique()),
                      "qty": int(g["d_jual_qty"].sum()),
                      "nilai": float(g["_val"].sum())}
                  for j, g in sy.groupby("jenis")},
    "per_bulan_tahun": {ym: {"qty": int(ym_qty[ym]), "nilai": float(ym_val[ym])}
                        for ym in ym_qty.index},
    "massal_per_bulan_tahun": {ym: {"faktur": int(big_ym_n[ym]), "qty": int(big_ym[ym]),
                                    "nilai": float(big.groupby("ym")["_val"].sum()[ym])}
                               for ym in big_ym.index},
    "pembeli_massal": {str(k): {"faktur": int(v["faktur"]), "qty": int(v["qty"]),
                                "nilai": float(v["nilai"])} for k, v in cust.iterrows()},
    "region_massal": {str(k): {"faktur": int(v["faktur"]), "qty": int(v["qty"]),
                               "nilai": float(v["nilai"])} for k, v in reg.iterrows()},
    "per_tahun": {str(int(y)): {"faktur": int(row["faktur"]), "qty": int(row["qty"]),
                                "nilai": float(row["nilai"])} for y, row in yr.iterrows()},
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print("\nSelesai. Disimpan ke " + OUT)