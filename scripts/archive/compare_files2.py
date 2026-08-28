import pandas as pd

df = pd.read_excel(r"E:\Download\Jurnal\df.xlsx")
view = pd.read_excel(r"E:\Download\Jurnal\view_penjualan_detail.xlsx")

merged = df.merge(
    view[["d_jual_id", "jual_total_fak", "pelanggan_kota"]],
    on="d_jual_id",
    how="inner",
    suffixes=("_df", "_view"),
)
print("matched:", len(merged))

mism = merged[merged["pelanggan_kota_df"] != merged["pelanggan_kota_view"]]
print("city mismatch rows:", len(mism))
if len(mism):
    mm = mism[["pelanggan_kota_df", "pelanggan_kota_view"]].drop_duplicates()
    print("mismatch pairs (df -> view):")
    for _, r in mm.iterrows():
        print(f"  {r['pelanggan_kota_df']!r} -> {r['pelanggan_kota_view']!r}")

# also: rows in df NOT matched by d_jual_id -> which cities?
unmatched = df[~df["d_jual_id"].isin(view["d_jual_id"])]
print("\nunmatched df rows:", len(unmatched))
print("unmatched by city:")
print(unmatched["pelanggan_kota"].value_counts().to_string())

# date range of unmatched
print("\nunmatched date range:", unmatched["jual_tanggal"].min(), "->", unmatched["jual_tanggal"].max())
print("matched date range:", merged["jual_tanggal"].min(), "->", merged["jual_tanggal"].max())

# jual_total_fak vs jual_total on matched rows: how different?
diff = (merged["jual_total"] - merged["jual_total_fak"]).abs()
print("\n|jual_total - jual_total_fak|: mean={:.2f} max={:,.0f} pct_equal={:.4f}%".format(
    diff.mean(), diff.max(), 100 * (diff < 1).mean()))
big = merged[diff >= 1]
print("rows with diff >= 1:", len(big))
if len(big):
    print(big[["d_jual_id", "jual_total", "jual_total_fak", "jual_pajak", "jual_total_pajak"]].head(10).to_string())