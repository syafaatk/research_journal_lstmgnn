import pandas as pd

path = r"E:\Download\Jurnal\view_penjualan_detail data hingga oktober.xlsx"
df = pd.read_excel(path)

print("rows:", len(df), "| unique d_jual_nofak:", df["d_jual_nofak"].nunique(),
      "| nulls:", df["d_jual_nofak"].isna().sum())

# wilayah unique values
print("\n=== WILAYAH unique values ===")
for v in sorted(df["wilayah"].dropna().unique(), key=str):
    print(f"  {v!r}: {(df['wilayah'] == v).sum()}")

# monthly distribution
df["month"] = df["tanggal"].dt.to_period("M")
print("\n=== per month ===")
print(df.groupby("month").size().to_string())

# invoices per month
inv = df.drop_duplicates("d_jual_nofak")
inv["month"] = inv["tanggal"].dt.to_period("M")
print("\n=== invoices per month ===")
print(inv.groupby("month").size().to_string())

# jual_total_fak per month (dedup invoice)
monthly = inv.groupby("month")["jual_total_fak"].sum()
print("\n=== monthly invoice sums (dedup) ===")
print(monthly.to_string())
print("\nmin:", f"{monthly.min():,.0f}", "max:", f"{monthly.max():,.0f}",
      "mean:", f"{monthly.mean():,.0f}")

# compare wilayah vs df.xlsx pelanggan_kota
dfx = pd.read_excel(r"E:\Download\Jurnal\df.xlsx")
print("\ndf.xlsx pelanggan_kota unique:", sorted(dfx["pelanggan_kota"].dropna().unique(), key=str))
print("\nwilayah unique:", sorted(df["wilayah"].dropna().unique(), key=str))

# check d_jual_id overlap with df.xlsx
print("\ndf.xlsx d_jual_id nunique:", dfx["d_jual_id"].nunique())
print("new file d_jual_id nunique:", df["d_jual_id"].nunique())
print("overlap:", len(set(df["d_jual_id"]) & set(dfx["d_jual_id"])))