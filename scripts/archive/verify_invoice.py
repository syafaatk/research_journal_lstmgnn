import pandas as pd

df = pd.read_excel(r"E:\Download\Jurnal\df.xlsx")

grp = df.groupby("d_jual_nofak")
city_n = grp["pelanggan_kota"].nunique()
date_n = grp["jual_tanggal"].nunique()
lat_n = grp["Latitude"].nunique()
lon_n = grp["longitude"].nunique()
print("invoices with >1 pelanggan_kota:", (city_n > 1).sum(), "dari", len(grp))
print("invoices with >1 jual_tanggal:", (date_n > 1).sum(), "dari", len(grp))
print("invoices with >1 Latitude:", (lat_n > 1).sum(), "dari", len(grp))
print("invoices with >1 longitude:", (lon_n > 1).sum(), "dari", len(grp))

inv = df.drop_duplicates("d_jual_nofak").copy()
inv["month"] = inv["jual_tanggal"].dt.to_period("M")
monthly = inv.groupby("month")["jual_total"].sum()
print("\nmonthly invoice sums (dedup):")
print(monthly.to_string())
print("\nmin:", f"{monthly.min():,.0f}", "max:", f"{monthly.max():,.0f}",
      "mean:", f"{monthly.mean():,.0f}")

cnt = inv.groupby("month").size()
print("\ninvoices per month: min", cnt.min(), "max", cnt.max(), "mean", round(cnt.mean(), 1))

print("\nrows:", len(df), "| unique d_jual_nofak:", df["d_jual_nofak"].nunique(),
      "| unique d_jual_id:", df["d_jual_id"].nunique())
print("sum jual_total unique-invoice:", f"{inv['jual_total'].sum():,.0f}")
print("sum jual_total unique-d_jual_id:", f"{df.drop_duplicates('d_jual_id')['jual_total'].sum():,.0f}")