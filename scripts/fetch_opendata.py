import requests

h = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# CKAN API: cek resource
for rid in ["40f2b908-90f1-4cb4-87f2-ba5b86ae73af", "0df14659-f836-4623-9290-4b60a033e1a2"]:
    url = "https://opendata.sumselprov.go.id/api/3/action/resource_show?id={}".format(rid)
    r = requests.get(url, headers=h, timeout=30)
    print(rid, "status:", r.status_code)
    if r.status_code == 200:
        j = r.json()
        if j.get("success"):
            res = j["result"]
            print("  name:", res.get("name"))
            print("  url:", res.get("url"))
            print("  format:", res.get("format"))
            # coba download via url
            r2 = requests.get(res["url"], headers=h, timeout=60)
            print("  download status:", r2.status_code, "len:", len(r2.content))
            if r2.status_code == 200 and len(r2.content) > 1000:
                y = "2022" if "2022" in res["name"] else "2023"
                out = r"E:\Download\Jurnal\data\realisasi_belanja_prov_{}.xlsx".format(y)
                open(out, "wb").write(r2.content)
                print("  saved", out)