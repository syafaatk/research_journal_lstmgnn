import requests

h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
urls = {
    "monev_dinkes_2022": "https://pdf2.sumselgo.id/ppiddinkes/unggah/35049042-Lap.%20Monev%20TW%20IV%20Dinkes%202022.pdf",
    "lakip_dinkes_2024": "https://pdf2.sumselgo.id/ppiddinkes/unggah/2025/52117252-lakip%202024.pdf",
    "lakip_dinkes_2023": "https://pdf2.sumselgo.id/ppiddinkes/unggah/96837133-LAKIP_2023.pdf",
}
for name, url in urls.items():
    r = requests.get(url, headers=h, timeout=120)
    print(name, "status:", r.status_code, "len:", len(r.content))
    if r.status_code == 200 and len(r.content) > 5000:
        out = r"E:\Download\Jurnal\data\{}.pdf".format(name)
        open(out, "wb").write(r.content)
        print("  saved", out)