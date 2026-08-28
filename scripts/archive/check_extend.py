import pandas as pd

df = pd.read_excel(r"E:\Download\Jurnal\df.xlsx")
view = pd.read_excel(r"E:\Download\Jurnal\view_penjualan_detail.xlsx")

# view-only invoices (d_jual_id not in df's d_jual_id set)
df_ids = set(df["d_jual_id"])
view_new = view[~view["d_jual_id"].isin(df_ids)].copy()
print("view rows:", len(view), "| view-only (not in df):", len(view_new))

view_new["month"] = view_new["jual_tanggal"].dt.to_period("M")
cnt = view_new.groupby("month").size()
print("\nview-only invoices per month:")
print(cnt.to_string())

# focus Apr-Oct 2025
apr_oct = view_new[view_new["month"] >= pd.Period("2025-04", "M")]
print("\nview-only Apr-Oct 2025:", len(apr_oct))
print("by city:")
print(apr_oct["pelanggan_kota"].value_counts().to_string())

# check: does view have invoices for ALL months Apr-Oct 2025?
all_view = view.copy()
all_view["month"] = all_view["jual_tanggal"].dt.to_period("M")
print("\nALL view invoices per month (2025):")
print(all_view[all_view["month"] >= pd.Period("2025-01", "M")].groupby("month").size().to_string())

# combined: df invoices (dedup by d_jual_nofak) + view-only
df_inv = df.drop_duplicates("d_jual_nofak").copy()
print("\ndf invoices (dedup):", len(df_inv), "| range:", df_inv["jual_tanggal"].min(), "->", df_inv["jual_tanggal"].max())
print("view-only range:", view_new["jual_tanggal"].min(), "->", view_new["jual_tanggal"].max())

# check overlap of view-only with df invoices by date+city+amount (sanity)
# view-only from 2021-2024: are they really missing from df?
early = view_new[view_new["month"] < pd.Period("2025-04", "M")]
print("\nview-only BEFORE Apr 2025:", len(early))
print(early.groupby("month").size().to_string())