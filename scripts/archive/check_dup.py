import pandas as pd

df = pd.read_excel(r"E:\Download\Jurnal\df.xlsx")
view = pd.read_excel(r"E:\Download\Jurnal\view_penjualan_detail.xlsx")

print("df rows:", len(df), "| d_jual_id nunique:", df["d_jual_id"].nunique())
print("view rows:", len(view), "| d_jual_id nunique:", view["d_jual_id"].nunique())

dup = df["d_jual_id"].duplicated().sum()
print("df duplicate d_jual_id rows:", dup)

if dup:
    # how many distinct invoices, and how many line items per invoice
    vc = df["d_jual_id"].value_counts()
    print("distinct invoices:", len(vc))
    print("items per invoice: min", vc.min(), "max", vc.max(), "mean", round(vc.mean(), 2))
    print("max repeat invoice:", vc.idxmax(), "count", vc.max())
    # check: does jual_total repeat within an invoice? (i.e., same value on each line)
    sample = df[df["d_jual_id"] == vc.idxmax()]
    print(sample[["d_jual_id", "jual_total", "d_jual_qty", "d_barang_brg_id", "barang_nama"]].head(10).to_string())

# compare sum of jual_total in df vs view for matched invoices
m = df.merge(view[["d_jual_id", "jual_total_fak"]], on="d_jual_id", how="inner")
print("\nmatched rows:", len(m))
print("sum jual_total (df, matched):", f"{m['jual_total'].sum():,.0f}")
print("sum jual_total_fak (view):", f"{m['jual_total_fak'].sum():,.0f}")
print("ratio df/view:", round(m["jual_total"].sum() / m["jual_total_fak"].sum(), 3))