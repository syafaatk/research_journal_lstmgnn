import pandas as pd

df = pd.read_excel(r"E:\Download\Jurnal\df.xlsx")

print("rows:", len(df))
print("d_jual_id nunique:", df["d_jual_id"].nunique())
print("d_jual_nofak nunique:", df["d_jual_nofak"].nunique())
print("d_jual_nofak nulls:", df["d_jual_nofak"].isna().sum())

# items per invoice (d_jual_nofak)
vc = df["d_jual_nofak"].value_counts()
print("\nitems per invoice (d_jual_nofak): min", vc.min(), "max", vc.max(),
      "mean", round(vc.mean(), 2), "median", vc.median())
print("invoices with >1 item:", (vc > 1).sum(), "dari", len(vc))
print("rows in multi-item invoices:", vc[vc > 1].sum())

# is jual_total repeated per invoice?
sample_nofak = vc.idxmax()
sub = df[df["d_jual_nofak"] == sample_nofak]
print("\nsample invoice", sample_nofak, "dengan", len(sub), "baris:")
print(sub[["d_jual_nofak", "d_jual_id", "jual_total", "d_jual_qty", "d_barang_brg_id", "barang_nama", "jual_tanggal"]].to_string())

# check: within each invoice, is jual_total constant?
grp = df.groupby("d_jual_nofak")["jual_total"].nunique()
print("\ninvoices where jual_total varies within invoice:", (grp > 1).sum(), "dari", len(grp))

# compare: sum of all rows vs sum of unique invoices (per month)
df["month"] = df["jual_tanggal"].dt.to_period("M")
sum_rows = df.groupby("month")["jual_total"].sum()
sum_inv = df.drop_duplicates("d_jual_nofak").groupby("month")["jual_total"].sum()
cmp = pd.DataFrame({"sum_rows": sum_rows, "sum_inv": sum_inv})
cmp["ratio"] = cmp["sum_rows"] / cmp["sum_inv"]
print("\nper-bulan: sum semua baris vs sum per faktur unik")
print(cmp.head(10).to_string())
print("...")
print("ratio mean:", round(cmp["ratio"].mean(), 4), "min:", round(cmp["ratio"].min(), 4),
      "max:", round(cmp["ratio"].max(), 4))

# how much of jual_total is double counted?
print("\ntotal jual_total (semua baris):", f"{df['jual_total'].sum():,.0f}")
print("total jual_total (faktur unik):", f"{df.drop_duplicates('d_jual_nofak')['jual_total'].sum():,.0f}")
print("selisih (overcount):", f"{df['jual_total'].sum() - df.drop_duplicates('d_jual_nofak')['jual_total'].sum():,.0f}")

# check d_jual_id vs d_jual_nofak relationship
print("\nd_jual_id unique per d_jual_nofak?",
      df.groupby("d_jual_nofak")["d_jual_id"].nunique().eq(1).all())