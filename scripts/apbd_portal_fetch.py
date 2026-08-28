"""Ambil data APBD per kab/kota Sumsel dari portal DJPK Kemenkeu.
URL: https://djpk.kemenkeu.go.id/portal/data/apbd?periode={P}&tahun={Y}&provinsi=06&pemda=XX
Realisasi kumulatif s.d. periode P tahun Y. Pemakaian:
  python apbd_portal_fetch.py [tahun] [periode]
  contoh: python apbd_portal_fetch.py 2025 10   (realisasi s.d. Oktober 2025)
          python apbd_portal_fetch.py 2024 12   (realisasi akhir tahun 2024)
Output: results/apbd_{tahun}_per_pemda.json
"""
import json
import os
import re
import sys
import time
import urllib.request
from html.parser import HTMLParser

BASE = r"E:\Download\Jurnal"

YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
PERIODE = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUT = os.path.join(BASE, "results", "apbd_{}_per_pemda.json".format(YEAR))

URL = ("https://djpk.kemenkeu.go.id/portal/data/apbd?periode={}&tahun={}"
       "&provinsi=06&pemda={}").format(PERIODE, YEAR, "{}")

# Kode pemda -> nama (dari dropdown portal, urutan DJPK)
PEMDA = {
    "01": "Kab. Lahat", "02": "Kab. Musi Banyuasin", "03": "Kab. Musi Rawas",
    "04": "Kab. Muara Enim", "05": "Kab. Ogan Komering Ilir", "06": "Kab. Ogan Komering Ulu",
    "07": "Kota Palembang", "08": "Kota Prabumulih", "09": "Kota Pagar Alam",
    "10": "Kota Lubuk Linggau", "11": "Kab. Banyuasin", "12": "Kab. Ogan Ilir",
    "13": "Kab. OKU Timur", "14": "Kab. OKU Selatan", "15": "Kab. Empat Lawang",
    "16": "Kab. Penukal Abab Lematang Ilir", "17": "Kab. Musi Rawas Utara",
}

# Map pemda -> region kita (MRU digabung ke Musi Rawas)
REGION_MAP = {
    "Kab. Lahat": "Lahat", "Kab. Musi Banyuasin": "Musi Banyuasin",
    "Kab. Musi Rawas": "Musi Rawas", "Kab. Muara Enim": "Muara Enim",
    "Kab. Ogan Komering Ilir": "Ogan Komering Ilir", "Kab. Ogan Komering Ulu": "Ogan Komering Ulu",
    "Kota Palembang": "Palembang", "Kota Prabumulih": "Prabumulih",
    "Kota Pagar Alam": "Pagar Alam", "Kota Lubuk Linggau": "Lubuk Linggau",
    "Kab. Banyuasin": "Banyuasin", "Kab. Ogan Ilir": "Ogan Ilir",
    "Kab. OKU Timur": "Ogan Komering Ulu Timur", "Kab. OKU Selatan": "Ogan Komering Ulu Selatan",
    "Kab. Empat Lawang": "Empat Lawang",
    "Kab. Penukal Abab Lematang Ilir": "Penukal Abab Lematang Ilir",
    "Kab. Musi Rawas Utara": "Musi Rawas",
}


class TblParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_td = False
        self.rows = []
        self.cur = []

    def handle_starttag(self, tag, attrs):
        if tag == "td":
            self.in_td = True
            self.cur.append("")

    def handle_data(self, d):
        if self.in_td:
            self.cur[-1] += d.strip()

    def handle_endtag(self, tag):
        if tag == "td":
            self.in_td = False
        if tag == "tr" and self.cur:
            self.rows.append(self.cur)
            self.cur = []


def parse_money(s):
    """'1.418,08 M' -> 1418080000000 (rupiah). 'M' = miliar, 'Jt' = juta."""
    s = s.strip().replace(" ", "")
    m = re.match(r"^([\d.,]+)\s*(M|Jt|Rb)?$", s)
    if not m:
        return None
    num = float(m.group(1).replace(".", "").replace(",", "."))
    unit = m.group(2) or ""
    if unit == "M":
        return num * 1e9
    if unit == "Jt":
        return num * 1e6
    if unit == "Rb":
        return num * 1e3
    return num


def parse_persen(s):
    if not s:
        return None
    try:
        return float(s.replace(",", "."))
    except ValueError:
        return None


def fetch_pemda(code):
    req = urllib.request.Request(URL.format(code), headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", errors="replace")
    p = TblParser()
    p.feed(html)
    data = {}
    for row in p.rows:
        if len(row) >= 5 and row[1]:
            data[row[1].strip()] = {
                "anggaran": parse_money(row[2]),
                "realisasi": parse_money(row[3]),
                "persen": parse_persen(row[4]),
            }
    return data


def main():
    out = {}
    for code, name in PEMDA.items():
        try:
            data = fetch_pemda(code)
            out[code] = {"pemda": name, "region": REGION_MAP[name], "akun": data}
            bj = data.get("Belanja Barang dan Jasa", {})
            bd = data.get("Belanja Daerah", {})
            print("{} {}: Belanja Daerah realisasi {:,.0f} M | Barang&Jasa realisasi {:,.0f} M".format(
                code, name,
                (bd.get("realisasi") or 0) / 1e6,
                (bj.get("realisasi") or 0) / 1e6), flush=True)
        except Exception as e:
            print("{} {}: ERROR {}".format(code, name, e), flush=True)
        time.sleep(0.5)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nSelesai. Disimpan ke " + OUT)


if __name__ == "__main__":
    main()