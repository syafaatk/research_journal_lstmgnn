import pandas as pd

df = pd.read_excel(r"E:\Download\Jurnal\df.xlsx")
view = pd.read_excel(r"E:\Download\Jurnal\view_penjualan_detail.xlsx")

merged = df.merge(
    view[["d_jual_id", "jual_total_fak", "pelanggan_kota"]],
    on="d_jual_id",
    how="inner",
    suffixes=("_df", "_view"),
)

# cross-tab: df city -> view city (matched rows)
ct = pd.crosstab(merged["pelanggan_kota_df"], merged["pelanggan_kota_view"])
print("=== CROSS-TAB df city x view city (matched rows) ===")
print(ct.to_string())

# view rows not in df
view_only = view[~view["d_jual_id"].isin(df["d_jual_id"])]
print("\n=== view rows NOT in df:", len(view_only))
print(view_only["pelanggan_kota"].value_counts().to_string())
print("date range:", view_only["jual_tanggal"].min(), "->", view_only["jual_tanggal"].max())

# df rows not in view by city (already have, but recheck counts)
df_only = df[~df["d_jual_id"].isin(view["d_jual_id"])]
print("\n=== df rows NOT in view:", len(df_only))

# For each df city: how many matched, and what view city dominates?
print("\n=== per df city: matched count, dominant view city ===")
for city in sorted(df["pelanggan_kota"].unique(), key=str):
    sub = merged[merged["pelanggan_kota_df"] == city]
    if len(sub) == 0:
        print(f"  {city!r}: 0 matched (all {int((df['pelanggan_kota']==city).sum())} rows unmatched)")
        continue
    vc = sub["pelanggan_kota_view"].value_counts()
    dom = vc.index[0]
    dom_pct = 100 * vc.iloc[0] / len(sub)
    others = ", ".join(f"{k}({v})" for k, v in vc.items() if k != dom)
    print(f"  {city!r}: matched={len(sub)}/{int((df['pelanggan_kota']==city).sum())} -> {dom!r} ({dom_pct:.1f}%)" + (f" | others: {others}" if others else ""))