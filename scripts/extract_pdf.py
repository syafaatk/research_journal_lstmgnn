import PyPDF2
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def extract(path):
    text = []
    with open(path, "rb") as f:
        r = PyPDF2.PdfReader(f)
        for i, page in enumerate(r.pages):
            t = page.extract_text() or ""
            text.append("===PAGE {}===\n{}".format(i + 1, t))
    return "\n".join(text)

# Cari tabel ringkasan jenis belanja: pola "Pegawai" + "Barang" + "Modal" berdekatan
for name, path in [
    ("LAKIP 2024", r"E:\Download\Jurnal\data\lakip_dinkes_2024.pdf"),
    ("LAKIP 2023", r"E:\Download\Jurnal\data\lakip_dinkes_2023.pdf"),
]:
    t = extract(path)
    print("\n########## {} ##########".format(name))
    # cari semua kemunculan "Pegawai" dengan konteks
    for m in list(re.finditer(r"Pegawai", t))[:8]:
        s = max(0, m.start() - 200)
        e = min(len(t), m.end() + 400)
        snippet = t[s:e].replace("\n", " | ")
        # hanya tampilkan yang mengandung angka besar (jutaan+)
        if re.search(r"\d{3}\.\d{3}", snippet):
            print("\n--- Pegawai ctx ---")
            print("  ...", snippet[:600])