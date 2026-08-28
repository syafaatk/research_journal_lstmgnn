import json
import numpy as np
import pandas as pd

with open(r"C:\Users\ADMLAY~1\AppData\Local\Temp\opencode\experiment_results.json", encoding="utf-8") as f:
    r = json.load(f)

print("=== PANEL ===")
print("n_months:", r["panel"]["n_months"], "| n_regions:", r["panel"]["n_regions"])
print("n_records_mapped:", r["panel"]["n_records_mapped"])
print("months:", r["panel"]["months"][0], "->", r["panel"]["months"][-1])
print("desc (per region per bulan):")
for k, v in r["panel"]["desc"].items():
    print(f"  {k}: {v}")

print("\n=== SPLIT ===")
print(r["split"]["train"], r["split"]["val"], r["split"]["test"])
print("test_months:", r["split"]["test_months"])

print("\n=== ZERO ACTUALS TEST ===", r["zero_actuals_test"], "dari 51")

print("\n=== GRAPH ===")
print("edges:", r["graph"]["n_edges"], "| degrees:", r["graph"]["degrees"])
print("knn_dists:", [round(x, 2) for row in r["graph"]["knn_dists"] for x in row])
print("coords:", r["graph"]["coords"])

# Rebuild test actuals from df to inspect
df = pd.read_excel(r"E:\Download\Jurnal\view_penjualan_detail data hingga oktober.xlsx")
df = df.dropna(subset=["d_jual_nofak", "tanggal", "jual_total_fak"])
df = df.drop_duplicates("d_jual_nofak")
df = df.rename(columns={"tanggal": "jual_tanggal", "wilayah": "pelanggan_kota",
                        "jual_total_fak": "jual_total", "latitude": "Latitude"})
CITY_MAP = {
    "Palembang": "Palembang", "Lahat": "Lahat", "Baturaja": "Ogan Komering Ulu",
    "Oku (Ogan Komering Ulu)": "Ogan Komering Ulu Timur", "Musi Banyuasin": "Musi Banyuasin",
    "Muara Enim": "Muara Enim", "Tanjung Enim": "Muara Enim", "Muara Lawai": "Muara Enim",
    "Lubuklinggau": "Lubuk Linggau", "Ogan Ilir": "Ogan Ilir", "Oku Selatan": "Ogan Komering Ulu Selatan",
    "Prabumulih": "Prabumulih", "Prabumulih Timur": "Prabumulih",
    "Pali (Penukal Abab Lematang Ilir)": "Penukal Abab Lematang Ilir", "Talang Ubi": "Penukal Abab Lematang Ilir",
    "Musi Rawas": "Musi Rawas", "Oku Timur": "Ogan Komering Ulu Timur", "Martapura": "Ogan Komering Ulu Timur",
    "Empat Lawang": "Empat Lawang", "Banyuasin": "Banyuasin", "Pagaralam": "Pagar Alam",
    "Muara Rupit": "Musi Rawas Utara", "Rupit": "Musi Rawas Utara",
    "Oki (Ogan Komering Ilir)": "Ogan Komering Ilir",
}
REGIONS = r["panel"]["desc"].keys()  # order from describe? no - use json order
# desc dict keys are region names in panel column order
regions = list(r["panel"]["desc"].keys())
print("\nregions order:", regions)

df = df[df["pelanggan_kota"].isin(CITY_MAP)].copy()
df["region"] = df["pelanggan_kota"].map(CITY_MAP)
df["month"] = df["jual_tanggal"].dt.to_period("M")
panel = df.pivot_table(index="month", columns="region", values="jual_total", aggfunc="sum")
panel = panel.reindex(columns=regions).fillna(0.0).sort_index()

test_months = [pd.Period(m, "M") for m in r["split"]["test_months"]]
test_actual = panel.loc[test_months].values  # (3, 17)
print("\n=== TEST ACTUALS (3 bulan x 17 region) ===")
print(pd.DataFrame(test_actual, index=[str(m) for m in test_months], columns=regions).to_string())

# predictions from HybridTuned (mean over seeds)
preds = np.array(r["models"]["HybridTuned"]["preds_original"])  # (3, 17)
print("\n=== PREDICTIONS HybridTuned (mean 5 seeds) ===")
print(pd.DataFrame(preds, index=[str(m) for m in test_months], columns=regions).to_string())

print("\n=== per-region test actual stats ===")
for i, reg in enumerate(regions):
    a = test_actual[:, i]
    p = preds[:, i]
    print(f"  {reg}: actual {a.tolist()} | pred {[round(x) for x in p.tolist()]}")

print("\n=== per-region model metrics ===")
for reg, m in r["per_region"].items():
    print(f"  {reg}: RMSE={m['RMSE']:.2f} MAE={m['MAE']:.2f} R2={m['R2']:.3f} MAPE={m['MAPE']:.1f}")

print("\n=== ABLATION ===")
for k, v in r["ablation"].items():
    print(f"  {k}: RMSE={v['RMSE']:.2f} MAE={v['MAE']:.2f} R2={v['R2']:.3f}")