"""Normalisasi file Data BPS: salin ke data\\bps\\ dengan nama pendek.
Source path panjang (>260 char) dibaca via prefix \\?\.
Output: data/bps/penduduk_YYYY.csv, dbd_YYYY.csv, rs_YYYY.csv, sarana_YYYY.csv
"""
import os
import shutil

SRC = r"E:\Download\Jurnal\data\Data BPS"
DST = r"E:\Download\Jurnal\data\bps"
os.makedirs(DST, exist_ok=True)

RULES = [
    ("Jumlah Penduduk Menurut Kabupaten_Kota", "penduduk"),
    ("Kasus Penyakit Menurut Kabupaten_Kota dan Jenis Penyakit", "dbd"),
    ("Jumlah Rumah Sakit Umum", "rs"),
    ("Jumlah Desa_Kelurahan Yang Memiliki Sarana Kesehatan", "sarana"),
]


def long_path(p):
    return "\\\\?\\" + os.path.abspath(p)


def main():
    copied = []
    for root, _, files in os.walk(SRC):
        for f in files:
            if not f.endswith(".csv"):
                continue
            year = None
            for y in range(2020, 2026):
                if ", " + str(y) + ".csv" in f or " " + str(y) + ".csv" in f:
                    year = y
                    break
            if year is None:
                continue
            prefix = None
            for key, name in RULES:
                if key in f:
                    prefix = name
                    break
            if prefix is None:
                continue
            src = os.path.join(root, f)
            dst = os.path.join(DST, "{}_{}.csv".format(prefix, year))
            shutil.copy(long_path(src), dst)
            copied.append(os.path.basename(dst))
    print("Tersalin {} file ke {}".format(len(copied), DST))
    for c in sorted(copied):
        print("  " + c)


if __name__ == "__main__":
    main()