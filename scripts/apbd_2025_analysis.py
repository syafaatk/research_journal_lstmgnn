"""Analisis APBD Provinsi Sumsel 2025 (realisasi sampai Oktober) dari data_apbd.xls.
File adalah XML Spreadsheet 2003 (bukan .xls biner). Data agregat provinsi,
bukan per kab/kota - tidak bisa dipetakan ke 16 region.
Tujuan: (1) simpan nilai anggaran/realisasi per akun; (2) perbarui korelasi
realisasi belanja barang&jasa provinsi vs total penjualan dengan nilai 2025
yang tersedia saat test set berakhir (Oktober 2025, anti-leakage).
Output: results/apbd_2025_analysis.json
"""
import json
import os
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = r"E:\Download\Jurnal"
APBD_XLS = os.path.join(BASE, "data", "data_apbd.xls")
BPS_JSON = os.path.join(BASE, "results", "bps_analysis.json")
OUT = os.path.join(BASE, "results", "apbd_2025_analysis.json")

NS = "{urn:schemas-microsoft-com:office:spreadsheet}"


def parse_apbd_xml(path):
    tree = ET.parse(path)
    rows = []
    for row in tree.getroot().iter(NS + "Row"):
        cells = []
        for cell in row.iter(NS + "Cell"):
            data = cell.find(NS + "Data")
            cells.append(data.text if data is not None and data.text else "")
        if cells:
            rows.append(cells)
    df = pd.DataFrame(rows)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    for c in ["Anggaran", "Realisasi", "Persentase"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main():
    df = parse_apbd_xml(APBD_XLS)
    print("=== APBD Provinsi Sumsel 2025 (realisasi s.d. Oktober) ===")
    print(df.to_string(index=False, max_colwidth=60), flush=True)

    # Nilai kunci
    belanja = df[df["Akun"].str.strip() == "Belanja Daerah"].iloc[0]
    bj = df[df["Akun"].str.strip() == "Belanja Barang dan Jasa"].iloc[0]
    pendapatan = df[df["Akun"].str.strip() == "Pendapatan Daerah"].iloc[0]
    print("\nBelanja Daerah: anggaran {:,.0f} M, realisasi {:,.0f} M ({:.1f}%)".format(
        belanja["Anggaran"] / 1e6, belanja["Realisasi"] / 1e6, belanja["Persentase"]), flush=True)
    print("Belanja Barang & Jasa: anggaran {:,.0f} M, realisasi {:,.0f} M ({:.1f}%)".format(
        bj["Anggaran"] / 1e6, bj["Realisasi"] / 1e6, bj["Persentase"]), flush=True)

    # Korelasi realisasi barang&jasa vs total penjualan
    res = json.load(open(BPS_JSON))
    realisasi = {int(y): float(v) for y, v in res["realisasi_belanja"].items()}
    sales = res["sales_by_year_region"]
    total_sales = {}
    for r, yd in sales.items():
        for y, v in yd.items():
            total_sales[int(y)] = total_sales.get(int(y), 0) + float(v)

    # 2025 dari file baru (rupiah -> ribu rupiah), tersedia saat test berakhir (Okt)
    realisasi[2025] = float(bj["Realisasi"]) / 1000.0

    yrs = sorted(y for y in realisasi if y in total_sales)
    b = [realisasi[y] for y in yrs]
    s = [total_sales[y] for y in yrs]
    r_c, p_c = spearmanr(b, s)
    print("\nKorelasi realisasi belanja barang&jasa provinsi vs total penjualan (n={}): r={:.3f} p={:.4f}".format(
        len(yrs), r_c, p_c), flush=True)
    for y in yrs:
        print("  {}: belanja {:,.2f} T vs penjualan {:,.0f} jt (rasio {:.2f})".format(
            y, realisasi[y] / 1e9, total_sales[y] / 1e6, realisasi[y] / max(total_sales[y], 1)), flush=True)

    out = {
        "source": "data_apbd.xls (XML Spreadsheet 2003), APBD Provinsi Sumsel 2025, realisasi s.d. Oktober 2025",
        "akun": df.to_dict("records"),
        "kunci": {
            "belanja_daerah": {"anggaran": float(belanja["Anggaran"]), "realisasi": float(belanja["Realisasi"]),
                               "persen": float(belanja["Persentase"])},
            "belanja_barang_jasa": {"anggaran": float(bj["Anggaran"]), "realisasi": float(bj["Realisasi"]),
                                    "persen": float(bj["Persentase"])},
            "pendapatan_daerah": {"anggaran": float(pendapatan["Anggaran"]), "realisasi": float(pendapatan["Realisasi"]),
                                  "persen": float(pendapatan["Persentase"])},
        },
        "korelasi_realisasi_vs_penjualan": {
            "n": len(yrs), "spearman": float(r_c), "p": float(p_c),
            "tahun": yrs, "realisasi_ribu_rupiah": {str(y): realisasi[y] for y in yrs},
            "total_penjualan": {str(y): total_sales[y] for y in yrs},
        },
        "catatan": "Data agregat provinsi (satu nilai), tidak bisa dipetakan ke 16 region. "
                   "Nilai 2025 dari file baru = realisasi s.d. Oktober (saat test set berakhir), "
                   "berbeda dari file lama yang berisi realisasi akhir tahun (2,20 T). "
                   "Korelasi tetap tidak signifikan.",
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSelesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()