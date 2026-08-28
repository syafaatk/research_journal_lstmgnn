# -*- coding: utf-8 -*-
"""
Interrupted Time Series (ITS): pengaruh program vaksinasi COVID-19 terhadap
penjualan syringe (disposable + auto destruct), Jul 2020 - Okt 2025.

Desain:
- Segmented regression bulanan dengan 3 intervensi:
    D1 Feb 2021 : vaksinasi dimulai di daerah (OKU Selatan 3 Feb 2021,
                  Muara Enim awal Feb 2021)
    D2 Jul 2021 : fase vaksinasi masif + gelombang Delta
    D3 Jan 2023 : transisi pasca-pandemi (PPKM dicabut akhir Des 2022)
  Spesifikasi: Y_t = b0 + b1*t + b2*D1 + b3*D2 + b4*D3 + dummy bulan + e
- Kontrol musiman: dummy bulan kalender (baseline = Desember).
- Autokorelasi: SE Newey-West HAC (lag 3); Durbin-Watson dilaporkan.
- CITS (kontrol): seri pembeli RS/rumah sakit sebagai pembanding - kanal
  distribusi dan siklus anggaran sama, tetapi permintaannya klinis, bukan
  kampanye imunisasi. Efek diferensial = koef intervensi dinkes dikurangi RS.
- Placebo: seri produk infus/catheter (tidak terdampak vaksinasi).

Catatan data: TIDAK ada dedup per faktur agar qty per baris item utuh.

Output: results/its_syringe.json, figures/its_syringe.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT_JSON = os.path.join(BASE, "results", "its_syringe.json")
OUT_FIG = os.path.join(BASE, "figures", "its_syringe.png")

INTERVENTIONS = {
    "D1_vaksinasi_mulai": "2021-02",
    "D2_fase_masif": "2021-07",
    "D3_pasca_pandemi": "2023-01",
}
HAC_LAGS = 3


def buyer_group(name):
    s = str(name).upper()
    if "DINKES" in s or "DINAS KESEHATAN" in s or "DIN KES" in s:
        return "dinkes"
    if "RS" in s or "RUMKIT" in s or "RUMAH SAKIT" in s:
        return "rs"
    return "lainnya"


# ---------------------------------------------------------------------------
# 1. Muat data dan bangun seri bulanan
# ---------------------------------------------------------------------------
df = pd.read_excel(DATA)
df = df.dropna(subset=["d_jual_nofak", "tanggal", "barang_nama", "d_jual_qty"])
df["barang_nama"] = df["barang_nama"].astype(str)

sy = df[df["barang_nama"].str.lower().str.contains("syringe", na=False)].copy()
sy["tanggal"] = pd.to_datetime(sy["tanggal"])
sy["ym"] = sy["tanggal"].dt.to_period("M")
sy["jenis"] = sy["barang_nama"].str.lower().apply(
    lambda s: "ad" if "auto destruct" in s else "disposable")
sy["grup"] = sy["pelanggan_nama"].map(buyer_group)

infus = df[df["barang_nama"].str.lower().str.contains(
    "infus|catheter|kateter", na=False)].copy()
infus["tanggal"] = pd.to_datetime(infus["tanggal"])
infus["ym"] = infus["tanggal"].dt.to_period("M")

full_idx = pd.period_range("2020-07", "2025-10", freq="M")


def monthly(series_df, col="d_jual_qty"):
    s = series_df.groupby("ym")[col].sum()
    return s.reindex(full_idx).fillna(0.0)


Y = pd.DataFrame({
    "total": monthly(sy),
    "ad": monthly(sy[sy["jenis"] == "ad"]),
    "disposable": monthly(sy[sy["jenis"] == "disposable"]),
    "dinkes": monthly(sy[sy["grup"] == "dinkes"]),
    "rs": monthly(sy[sy["grup"] == "rs"]),
    "lainnya": monthly(sy[sy["grup"] == "lainnya"]),
    "infus_placebo": monthly(infus),
})
print(f"Periode: {full_idx[0]} s.d. {full_idx[-1]} ({len(full_idx)} bulan)", flush=True)
print("Pembeli syringe per grup (qty total):", flush=True)
for g, gdf in sy.groupby("grup"):
    print(f"  {g:<8} qty={int(gdf['d_jual_qty'].sum()):>10,}", flush=True)


# ---------------------------------------------------------------------------
# 2. Matriks desain ITS
# ---------------------------------------------------------------------------
def build_X(idx):
    cols = {"t": np.arange(len(idx), dtype=float)}
    for name, start in INTERVENTIONS.items():
        cols[name] = (idx >= pd.Period(start)).astype(float)
    for m in range(1, 12):  # baseline = Desember
        cols[f"m{m}"] = (idx.month == m).astype(float)
    return pd.DataFrame(cols, index=idx.astype(str))


X = build_X(full_idx)


def fit_its(y, label):
    Xc = sm.add_constant(X)
    res = sm.OLS(y.values, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    plain = sm.OLS(y.values, Xc).fit()
    dw = durbin_watson(plain.resid)
    print(f"\n=== ITS {label} === adjR2={plain.rsquared_adj:.3f} DW={dw:.2f}", flush=True)
    tab = {}
    for k in ["const", "t"] + list(INTERVENTIONS):
        row = {
            "coef": float(res.params[k]),
            "se_hac": float(res.bse[k]),
            "ci_lo": float(res.conf_int().loc[k][0]),
            "ci_hi": float(res.conf_int().loc[k][1]),
            "p": float(res.pvalues[k]),
        }
        tab[k] = row
        star = "***" if row["p"] < 0.01 else "**" if row["p"] < 0.05 else \
               "*" if row["p"] < 0.1 else ""
        print(f"  {k:<20} coef={row['coef']:>12,.1f} se={row['se_hac']:>10,.1f} "
              f"p={row['p']:.4f} {star}", flush=True)
    return {"params": tab, "adj_r2": float(plain.rsquared_adj),
            "dw": float(dw), "n": int(len(y))}


results = {}
results["total"] = fit_its(Y["total"], "Total syringe (qty)")
results["ad"] = fit_its(Y["ad"], "Auto destruct saja")
results["disposable"] = fit_its(Y["disposable"], "Disposable saja")


# ---------------------------------------------------------------------------
# 3. CITS: dinkes vs rs (interaksi penuh)
# ---------------------------------------------------------------------------
stack = []
for grp in ["dinkes", "rs"]:
    d = X.copy()
    d["treat"] = 1.0 if grp == "dinkes" else 0.0
    d["y"] = Y[grp].values
    stack.append(d)
pool = pd.concat(stack, ignore_index=True)
tr = pool["treat"]
Xc_pool = pd.DataFrame({"const": 1.0, "treat": tr}, index=pool.index)
base_cols = ["t"] + list(INTERVENTIONS) + [c for c in X.columns if c.startswith("m")]
for c in base_cols:
    Xc_pool[c] = pool[c]
    Xc_pool[f"{c}:treat"] = pool[c] * tr
res_cits = sm.OLS(pool["y"].values, Xc_pool).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
print("\n=== CITS dinkes vs rs (efek diferensial = interaksi :treat) ===", flush=True)
tab_cits = {}
for k in list(INTERVENTIONS) + ["t"]:
    kk = f"{k}:treat"
    row = {"coef": float(res_cits.params[kk]), "se_hac": float(res_cits.bse[kk]),
           "p": float(res_cits.pvalues[kk])}
    tab_cits[k] = row
    star = "***" if row["p"] < 0.01 else "**" if row["p"] < 0.05 else \
          "*" if row["p"] < 0.1 else ""
    print(f"  {kk:<28} coef={row['coef']:>12,.1f} se={row['se_hac']:>10,.1f} "
          f"p={row['p']:.4f} {star}", flush=True)
results["cits_dinkes_vs_rs"] = {"differential": tab_cits,
                                "adj_r2": float(res_cits.rsquared_adj),
                                "dw": float(durbin_watson(res_cits.resid))}


# ---------------------------------------------------------------------------
# 4. Placebo: infus/catheter
# ---------------------------------------------------------------------------
results["placebo_infus"] = fit_its(Y["infus_placebo"], "Placebo infus/catheter")


# ---------------------------------------------------------------------------
# 5. Estimasi unit ekstra (deskriptif, dari model total & AD)
#    Kontrafaktual di-floor ke 0 agar tren linier tidak mengekstrapolasi
#    nilai negatif yang menggelembungkan estimasi.
# ---------------------------------------------------------------------------
def excess_estimate(series_name):
    y = Y[series_name].values
    Xc_ = sm.add_constant(X)
    fit = sm.OLS(y, Xc_).fit()
    X_counter = X.copy()
    for k in INTERVENTIONS:
        X_counter[k] = 0.0
    cf = np.maximum(fit.predict(sm.add_constant(X_counter)), 0.0)
    mask_vax = (full_idx >= pd.Period("2021-02")) & (full_idx <= pd.Period("2022-12"))
    mask_all = full_idx >= pd.Period("2021-02")
    ex_vax = float((y[mask_vax] - cf[mask_vax]).sum())
    ex_all = float((y[mask_all] - cf[mask_all]).sum())
    act_vax = float(y[mask_vax].sum())
    act_all = float(y[mask_all].sum())
    print(f"\n[{series_name}] aktual Feb21-Des22: {act_vax:,.0f} | "
          f"ekstra vs kontrafaktual: {ex_vax:,.0f} ({100 * ex_vax / act_vax:.0f}% dari aktual)", flush=True)
    print(f"[{series_name}] aktual Feb21-Okt25: {act_all:,.0f} | "
          f"ekstra: {ex_all:,.0f} ({100 * ex_all / act_all:.0f}% dari aktual)", flush=True)
    assert ex_vax <= act_vax and ex_all <= act_all, "Ekstra melebihi aktual - kontrafaktual invalid"
    return {"aktual_feb21_des22": act_vax, "ekstra_feb21_des22": ex_vax,
            "aktual_feb21_okt25": act_all, "ekstra_feb21_okt25": ex_all}


results["excess_units_total"] = excess_estimate("total")
results["excess_units_ad"] = excess_estimate("ad")


# ---------------------------------------------------------------------------
# 6. Figur
# ---------------------------------------------------------------------------
Xc_fig = sm.add_constant(X)
fit_full = sm.OLS(Y["total"].values, Xc_fig).fit()
X_counter_fig = X.copy()
for k in INTERVENTIONS:
    X_counter_fig[k] = 0.0
pred_cf = np.maximum(fit_full.predict(sm.add_constant(X_counter_fig)), 0.0)

fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
xnum = np.arange(len(full_idx))

ax = axes[0]
ax.bar(xnum, Y["total"], color="#9db9d6", label="Penjualan syringe (unit)")
ax.plot(xnum, fit_full.fittedvalues, color="#1a4a7a", lw=2, label="Fitted ITS")
ax.plot(xnum, pred_cf, color="#c0392b", ls="--", lw=1.5, label="Kontrafaktual tanpa intervensi")
for name, start in INTERVENTIONS.items():
    xi = list(full_idx).index(pd.Period(start))
    ax.axvline(xi, color="#555555", ls=":", lw=1)
    ax.text(xi, ax.get_ylim()[1] * 0.95, name.split("_")[0], fontsize=8, ha="left")
ax.set_ylabel("Unit / bulan")
ax.set_title("ITS Penjualan Syringe vs Intervensi Vaksinasi COVID-19, Sumsel (Jul 2020 - Okt 2025)")
ax.legend(fontsize=8, loc="upper right")

ax = axes[1]
w = 0.4
ax.bar(xnum - w / 2, Y["dinkes"], width=w, color="#2e7d32", label="Dinkes (kab/kota/prov)")
ax.bar(xnum + w / 2, Y["rs"], width=w, color="#8e44ad", label="RS/rumah sakit")
for name, start in INTERVENTIONS.items():
    xi = list(full_idx).index(pd.Period(start))
    ax.axvline(xi, color="#555555", ls=":", lw=1)
ax.set_ylabel("Unit / bulan")
ax.set_xlabel("Bulan")
ticks = list(range(0, len(full_idx), 3))
ax.set_xticks(ticks)
ax.set_xticklabels([str(full_idx[i]) for i in ticks], rotation=45, ha="right", fontsize=7)
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT_FIG, dpi=150)
print(f"\nFigur disimpan: {OUT_FIG}", flush=True)

with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1, ensure_ascii=False)
print(f"Hasil disimpan: {OUT_JSON}", flush=True)
