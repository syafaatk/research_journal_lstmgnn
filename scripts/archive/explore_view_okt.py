import pandas as pd

path = r"E:\Download\Jurnal\view_penjualan_detail data hingga oktober.xlsx"
xl = pd.ExcelFile(path)
print("SHEETS:", xl.sheet_names)

df = pd.read_excel(path, sheet_name=xl.sheet_names[0])
print("SHAPE:", df.shape)
print("COLUMNS:", list(df.columns))
print("DTYPES:")
print(df.dtypes)

for c in df.columns:
    if "tanggal" in c.lower() or "date" in c.lower():
        try:
            print(f"DATE COL {c}: min={df[c].min()} max={df[c].max()} nunique={df[c].nunique()}")
        except Exception as e:
            print(f"DATE COL {c}: err {e}")

for c in df.columns:
    if "fak" in c.lower() or "jual" in c.lower():
        print(f"--- {c}: nunique={df[c].nunique()} nulls={df[c].isna().sum()} dtype={df[c].dtype}")
        if pd.api.types.is_numeric_dtype(df[c]):
            print(f"    min={df[c].min()} max={df[c].max()} mean={df[c].mean():,.0f} sum={df[c].sum():,.0f}")

for c in df.columns:
    if "kota" in c.lower():
        vals = df[c].dropna().unique()
        print(f"--- {c}: {len(vals)} unique values")
        for v in sorted(vals, key=str):
            print(f"    {v!r}: {(df[c] == v).sum()}")

# monthly distribution
df["month"] = df["jual_tanggal"].dt.to_period("M")
print("\nper month (2025):")
print(df[df["month"] >= pd.Period("2025-01", "M")].groupby("month").size().to_string())
print("\nper month (all):")
print(df.groupby("month").size().to_string())