import pandas as pd

df = pd.read_excel(r"E:\Download\Jurnal\df.xlsx")
view = pd.read_excel(r"E:\Download\Jurnal\view_penjualan_detail.xlsx")

# find city columns
df_city_col = [c for c in df.columns if "kota" in c.lower()]
view_city_col = [c for c in view.columns if "kota" in c.lower()]
print("df city cols:", df_city_col)
print("view city cols:", view_city_col)

df_city = df_city_col[0]
view_city = view_city_col[0]

df_vals = set(df[df_city].dropna().unique())
view_vals = set(view[view_city].dropna().unique())

print("\n=== df.xlsx unique cities:", len(df_vals))
for v in sorted(df_vals, key=str):
    print(f"  {v!r}: {int((df[df_city]==v).sum())}")

print("\n=== view unique cities:", len(view_vals))
for v in sorted(view_vals, key=str):
    print(f"  {v!r}: {int((view[view_city]==v).sum())}")

print("\n=== in df but NOT view:")
for v in sorted(df_vals - view_vals, key=str):
    print(f"  {v!r}")
print("\n=== in view but NOT df:")
for v in sorted(view_vals - df_vals, key=str):
    print(f"  {v!r}")

# check jual_total_fak vs jual_total in df: is there a join key?
print("\ndf columns:", list(df.columns))
print("view columns:", list(view.columns))

# try joining on d_jual_id if present in df
if "d_jual_id" in df.columns:
    merged = df.merge(view[["d_jual_id", "jual_total_fak", view_city]], on="d_jual_id", how="inner", suffixes=("_df", "_view"))
    print("\nmerge on d_jual_id: matched", len(merged), "of", len(df))
    if len(merged):
        # compare jual_total vs jual_total_fak
        jt = [c for c in df.columns if c.lower().startswith("jual_total")]
        print("jual_total cols in df:", jt)
        for c in jt:
            if c in merged.columns:
                diff = (merged[c] - merged["jual_total_fak"]).abs()
                print(f"  |{c} - jual_total_fak|: mean={diff.mean():,.0f} max={diff.max():,.0f} pct_equal={100*(diff<1).mean():.2f}%")
        # check city consistency
        mism = (merged[df_city] != merged[view_city]).sum()
        print(f"  city mismatch on matched rows: {mism}/{len(merged)}")
        if mism:
            mm = merged[merged[df_city] != merged[view_city]][[df_city, view_city]].drop_duplicates()
            print("  mismatch pairs:")
            for _, r in mm.iterrows():
                print(f"    {r[df_city]!r} -> {r[view_city]!r}")
else:
    print("\nno d_jual_id in df")