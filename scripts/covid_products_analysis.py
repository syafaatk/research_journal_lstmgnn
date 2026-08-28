"""Analisis produk COVID mahal: Airvo Unit Optiflow Therapy dan
Hyper-Hypothermia System Blanketrol III. Kontribusi per tahun terhadap
total penjualan, puncak pandemi vs pasca-pandemi.
Output: results/covid_products_analysis.json
"""
import json
import os
import numpy as np
import pandas as pd

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(BASE, "results", "covid_products_analysis.json")

PRODUCTS = ["Airvo Unit Optiflow Therapy", "Hyper-Hypothermia System Blanketrol III"]
ACCESSORY = "Junior Optiflow Nasal Cannula"


def main():
    df = pd.read_excel(DATA)
    df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
    df = df.drop_duplicates("d_jual_nofak")
    df["year"] = df["tanggal"].dt.year

    total_by_year = df.groupby("year")["jual_total_fak"].sum()
    print("Total penjualan per tahun (juta):")
    print((total_by_year / 1e6).round(0).to_string(), flush=True)

    out = {"total_by_year": (total_by_year / 1e6).round(0).to_dict()}
    for prod in PRODUCTS + [ACCESSORY]:
        sub = df[df["barang_nama"] == prod]
        by_year = sub.groupby("year")["jual_total_fak"].sum()
        share = (by_year / total_by_year * 100).round(2)
        print("\n=== {} ===".format(prod), flush=True)
        print("Total: {:,.0f} IDR ({:,.1f} jt)".format(sub["jual_total_fak"].sum(),
                                                        sub["jual_total_fak"].sum() / 1e6), flush=True)
        print("Faktur: {} | Pelanggan unik: {} | Region: {}".format(
            len(sub), sub["pelanggan_nama"].nunique(), sub["wilayah"].nunique()), flush=True)
        t = pd.DataFrame({"penjualan_jt": (by_year / 1e6).round(1),
                          "share_persen": share})
        print(t.to_string(), flush=True)
        out[prod] = {
            "total": float(sub["jual_total_fak"].sum()),
            "n_invoices": int(len(sub)),
            "n_customers": int(sub["pelanggan_nama"].nunique()),
            "by_year_jt": (by_year / 1e6).round(1).to_dict(),
            "share_persen": share.to_dict(),
        }

    # Kontribusi kedua produk utama terhadap penurunan 2021->2025
    main_prods = df[df["barang_nama"].isin(PRODUCTS)]
    main_by_year = main_prods.groupby("year")["jual_total_fak"].sum()
    print("\n=== Kontribusi 2 produk utama ===")
    for y in sorted(total_by_year.index):
        if y in main_by_year.index:
            print("  {}: {:,.1f} jt dari {:,.1f} jt ({:.1f}%)".format(
                y, main_by_year[y] / 1e6, total_by_year[y] / 1e6,
                main_by_year[y] / total_by_year[y] * 100), flush=True)
    out["main_products_by_year"] = (main_by_year / 1e6).round(1).to_dict()

    # Puncak: bulan penjualan tertinggi per produk
    print("\n=== Bulan puncak per produk ===")
    for prod in PRODUCTS:
        sub = df[df["barang_nama"] == prod]
        monthly = sub.groupby(sub["tanggal"].dt.to_period("M"))["jual_total_fak"].sum()
        top = monthly.sort_values(ascending=False).head(5)
        print("{}:".format(prod), flush=True)
        for m, v in top.items():
            print("   {}: {:,.1f} jt".format(m, v / 1e6), flush=True)
        out[prod]["top_months_jt"] = {str(m): round(v / 1e6, 1) for m, v in top.items()}

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSelesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()