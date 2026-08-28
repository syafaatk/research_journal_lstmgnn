import pandas as pd

new = pd.read_excel(r"E:\Download\Jurnal\view_penjualan_detail data hingga oktober.xlsx")
dfx = pd.read_excel(r"E:\Download\Jurnal\df.xlsx")
view = pd.read_excel(r"E:\Download\Jurnal\view_penjualan_detail.xlsx")

# 1) new file wilayah vs df.xlsx pelanggan_kota on overlapping d_jual_id
m1 = new.merge(dfx[["d_jual_id", "pelanggan_kota"]], on="d_jual_id", how="inner", suffixes=("_new", "_df"))
print("overlap new-df:", len(m1))
mm = m1[m1["wilayah"] != m1["pelanggan_kota"]]
print("wilayah != pelanggan_kota:", len(mm))
if len(mm):
    print(mm[["wilayah", "pelanggan_kota"]].drop_duplicates().to_string())

# 2) new file wilayah vs view_penjualan_detail.xlsx pelanggan_kota on overlapping d_jual_id
m2 = new.merge(view[["d_jual_id", "pelanggan_kota"]], on="d_jual_id", how="inner", suffixes=("_new", "_view"))
print("\noverlap new-view:", len(m2))
mm2 = m2[m2["wilayah"] != m2["pelanggan_kota"]]
print("wilayah != view pelanggan_kota:", len(mm2))
if len(mm2):
    print(mm2[["wilayah", "pelanggan_kota"]].drop_duplicates().to_string())

# 3) within-invoice consistency of wilayah in new file
grp = new.groupby("d_jual_nofak")["wilayah"].nunique()
print("\ninvoices with >1 wilayah:", (grp > 1).sum(), "dari", len(grp))

# 4) check the 2 null d_jual_nofak rows
nulls = new[new["d_jual_nofak"].isna()]
print("\nnull d_jual_nofak rows:", len(nulls))
print(nulls[["d_jual_id", "tanggal", "jual_total_fak", "wilayah"]].to_string())

# 5) does new file have Oku (Ogan Komering Ulu) mapped to Oku Timur anywhere?
oku = new[new["wilayah"].isin(["Oku (Ogan Komering Ulu)", "Oku Timur"])]
print("\nOku rows:", len(oku), "| Oku Timur rows:", len(new[new["wilayah"] == "Oku Timur"]))