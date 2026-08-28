# -*- coding: utf-8 -*-
"""Tampilkan daftar pelanggan tanpa koordinat aktual (CITY_LEVEL), terkelompok."""
import pandas as pd

nf = pd.read_csv(r"E:\Download\Jurnal\skrip\not_found_list.csv")
nf = nf.sort_values("n_inv", ascending=False)
total_inv = nf["n_inv"].sum()
print(f"Total belum ada koordinat aktual: {len(nf)} pelanggan | {total_inv} faktur ({100*total_inv/10878:.1f}%)")
print()


def tipe(nama):
    n = nama.upper()
    if any(k in n for k in ("RS", "RSUD", "RSUP", "RSIA", "RUMKIT", "RUMAH SAKIT")):
        return "RS / Operator RS"
    if any(k in n for k in ("DINAS", "DINKES", "BPBD", "BADAN", "KANWIL", "PUSKESMAS")):
        return "Instansi pemerintah"
    if "APOTEK" in n:
        return "Apotek"
    if any(k in n for k in ("PT.", "CV.", "YAY", "YAYASAN")):
        return "Perusahaan / Yayasan"
    return "Individu / Klinik / Lain"


nf["tipe"] = nf["pelanggan_nama"].apply(tipe)
for t, g in nf.groupby("tipe"):
    print(f"--- {t} ({len(g)}) ---")
    for _, r in g.iterrows():
        print(f"  {r['n_inv']:>4} | {r['pelanggan_nama']} [{r['region']}]")
    print()