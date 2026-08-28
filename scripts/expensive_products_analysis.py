"""Analisis produk bernilai tinggi dan pembelian qty besar - grain benar.
Grain analisis = baris item. Nilai item = kolom `jumlah` (qty x harga satuan);
kolom jual_total_fak berulang di tiap baris dan tidak boleh dijumlahkan per
baris, dan dedup per faktur membuang baris item (bug lama).
1. Produk dengan harga satuan tertinggi (jumlah/qty per baris item).
2. Produk neonatal (Neopuff, Infant Warmer, nCPAP, dll): total, pembeli, region, tahun.
3. Pembelian qty besar (baris item dengan d_jual_qty >= 10.000).
Output: results/expensive_products_analysis.json
"""
import json
import os
import pandas as pd

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT = os.path.join(BASE, "results", "expensive_products_analysis.json")

NEONATAL_KEYWORDS = ["Neopuff", "Infant", "Neonatal", "Bubble", "Resuscitator", "Warmer"]


def main():
    df = pd.read_excel(DATA)
    df = df.dropna(subset=["d_jual_nofak", "tanggal", "barang_nama", "d_jual_qty"])
    df["_val"] = pd.to_numeric(df["jumlah"], errors="coerce").fillna(0.0)
    df["year"] = pd.to_datetime(df["tanggal"]).dt.year

    # --- 1. Harga satuan tertinggi (per baris item: jumlah / qty) ---
    it = df[df["d_jual_qty"] > 0].copy()
    it["unit_price"] = it["_val"] / it["d_jual_qty"]
    top_unit = it.nlargest(25, "unit_price")[
        ["tanggal", "barang_nama", "d_jual_qty", "_val", "unit_price", "wilayah", "pelanggan_nama"]]
    print("=== 25 item dengan harga satuan tertinggi ===")
    for _, r in top_unit.iterrows():
        print("  {:>14,.0f}/unit  qty={:>4,.0f}  total={:>14,.0f}  {} | {} | {} | {}".format(
            r["unit_price"], r["d_jual_qty"], r["_val"],
            r["barang_nama"], pd.Timestamp(r["tanggal"]).date(), r["wilayah"], r["pelanggan_nama"]), flush=True)

    # Produk unik dengan harga satuan max > 50 jt
    prod_max = it.groupby("barang_nama")["unit_price"].max().sort_values(ascending=False)
    expensive = prod_max[prod_max > 50e6]
    print("\n=== Produk dengan harga satuan max > 50 jt ({} produk) ===".format(len(expensive)))
    for p, v in expensive.head(30).items():
        print("  {:>14,.0f}  {}".format(v, p), flush=True)

    # --- 2. Produk neonatal (level item) ---
    mask = df["barang_nama"].str.contains("|".join(NEONATAL_KEYWORDS), case=False, na=False)
    neo = df[mask]
    print("\n=== Produk neonatal ({} faktur, {:,.0f} IDR) ===".format(
        neo["d_jual_nofak"].nunique(), neo["_val"].sum()))
    neo_by_prod = neo.groupby("barang_nama").agg(
        total=("_val", "sum"), n=("d_jual_nofak", "nunique"),
        qty=("d_jual_qty", "sum")).sort_values("total", ascending=False)
    for p, r in neo_by_prod.iterrows():
        print("  {:>13,.0f}  n={:<3} qty={:<6} {}".format(r["total"], r["n"], r["qty"], p), flush=True)

    neo_by_year = neo.groupby("year")["_val"].sum()
    print("\nNeonatal per tahun (juta):")
    print((neo_by_year / 1e6).round(1).to_string(), flush=True)

    neo_by_region = neo.groupby("wilayah")["_val"].sum().sort_values(ascending=False)
    print("\nNeonatal per region (juta):")
    print((neo_by_region / 1e6).round(1).to_string(), flush=True)

    # Pembeli neonatal
    neo_cust = neo.groupby("pelanggan_nama").agg(
        total=("_val", "sum"), n=("d_jual_nofak", "nunique"),
        region=("wilayah", "first")).sort_values("total", ascending=False).head(15)
    print("\nPembeli neonatal terbesar:")
    for c, r in neo_cust.iterrows():
        print("  {:>13,.0f}  n={:<3} {} | {}".format(r["total"], r["n"], r["region"], c), flush=True)

    # --- 3. Pembelian qty besar (baris item >= 10.000 unit) ---
    top_qty = df.nlargest(20, "d_jual_qty")[
        ["tanggal", "barang_nama", "d_jual_qty", "_val", "wilayah", "pelanggan_nama"]]
    print("\n=== 20 item dengan qty terbanyak ===")
    for _, r in top_qty.iterrows():
        print("  qty={:>8,.0f}  total={:>13,.0f}  {} | {} | {} | {}".format(
            r["d_jual_qty"], r["_val"], r["barang_nama"],
            pd.Timestamp(r["tanggal"]).date(), r["wilayah"], r["pelanggan_nama"]), flush=True)

    big_qty = df[df["d_jual_qty"] >= 10000]
    print("\n=== Item qty >= 10.000 unit ({} baris, {:,} faktur, {:,.0f} IDR) ===".format(
        len(big_qty), big_qty["d_jual_nofak"].nunique(), big_qty["_val"].sum()))
    big_by_prod = big_qty.groupby("barang_nama").agg(
        total=("_val", "sum"), n=("d_jual_nofak", "nunique"),
        qty=("d_jual_qty", "sum")).sort_values("qty", ascending=False).head(15)
    for p, r in big_by_prod.iterrows():
        print("  qty={:>10,.0f}  total={:>13,.0f}  n={:<3} {}".format(
            r["qty"], r["total"], r["n"], p), flush=True)

    out = {
        "catatan": "grain = baris item; nilai = kolom jumlah",
        "top_unit_price_items": top_unit.to_dict("records"),
        "products_max_unit_price_gt_50jt": {p: float(v) for p, v in expensive.head(30).items()},
        "neonatal_total": {"faktur": int(neo["d_jual_nofak"].nunique()),
                           "nilai": float(neo["_val"].sum()),
                           "qty": int(neo["d_jual_qty"].sum())},
        "neonatal_by_product": neo_by_prod.to_dict(),
        "neonatal_by_year_jt": (neo_by_year / 1e6).round(1).to_dict(),
        "neonatal_by_region_jt": (neo_by_region / 1e6).round(1).to_dict(),
        "neonatal_top_customers": neo_cust.to_dict(),
        "top_qty_items": top_qty.to_dict("records"),
        "big_qty_summary": {"baris": int(len(big_qty)),
                            "faktur": int(big_qty["d_jual_nofak"].nunique()),
                            "qty": int(big_qty["d_jual_qty"].sum()),
                            "nilai": float(big_qty["_val"].sum())},
        "big_qty_by_product": big_by_prod.to_dict(),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSelesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()
