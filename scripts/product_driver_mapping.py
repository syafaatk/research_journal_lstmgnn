# -*- coding: utf-8 -*-
"""Pemetaan produk (barang_nama + barang_spesifikasi) ke grup pemicu permintaan.

Revisi besar atas versi lama (lihat git/logs untuk versi sebelumnya):
  1. Klasifikasi LEVEL BARIS (nama + spesifikasi), bukan nama saja.
     Contoh: "Disposable Syringe" spesifikasi 1 cc -> IMUNISASI,
     spesifikasi 3/5/10 cc -> SPUIT_INJEKSI (klinis rutin).
  2. Normalisasi nama (spasi/kapital) sehingga varian tidak menyebar.
  3. Typo yang sebelumnya lolos ke default ditutup: Syiringe, IV Canula,
     Blood Tranfusion, Polypropylane, Fostryl, Rapid/Rapit, Plain x/0, Nylon.
  4. Taksonomi diperluas dengan grup lini layanan agar OPERASIONAL_RS
     hanya menjadi residual untuk produk yang benar-benar generik:
     SPUIT_INJEKSI, PERAWATAN_LUKA, RESPIRATORIUM, UROLOGI, LABORATORIUM,
     STERILISASI_CSSD, OBGYN, APD_HIGIENE.
  5. Grup lama (PANDEMI_COVID, IMUNISASI, KARHUTLA, DBD, dst.) dipertahankan
     demi kompatibilitas konsumen downstream.

Output (skema sama seperti versi lama):
  - results/product_driver_mapping.json
  - results/product_driver_monthly.csv
  - results/product_event_spikes.json
"""
import json
import os
import re

import pandas as pd

BASE = r"E:\Download\Jurnal"
DATA = os.path.join(BASE, "data", "view_penjualan_detail data hingga oktober.xlsx")
OUT_JSON = os.path.join(BASE, "results", "product_driver_mapping.json")
OUT_CSV = os.path.join(BASE, "results", "product_driver_monthly.csv")
OUT_SPIKE = os.path.join(BASE, "results", "product_event_spikes.json")

GROUPS = {
    "PANDEMI_COVID": "Respirasi akut / COVID-19 (tes, APD darurat, terapi high-flow)",
    "IMUNISASI": "Program imunisasi / vaksinasi (ADS, safety box, spuit 1cc)",
    "KARHUTLA": "Asap karhutla (masker pernapasan)",
    "DBD": "Terapi cairan IV / transfusi (pemicu musiman DBD)",
    "NEONATAL": "Perawatan neonatal / NICU",
    "BEDAH_OK": "Operasi / kamar bedah (instrumen, jahitan, sarung tangan steril)",
    "DIAGNOSTIK": "Diagnostik / penunjang (ECG, USG, film, strip)",
    "FARMASI": "Farmasi / racikan (kapsul kosong, antiseptik)",
    "CAPITAL_EQUIP": "Alat kesehatan modal (monitor, meja, alat terapi)",
    "OPERASIONAL_RS": "Operasional harian RS (residual produk generik)",
    "SPUIT_INJEKSI": "Spuit injeksi klinis non-imunisasi (3cc s.d. 50cc)",
    "PERAWATAN_LUKA": "Perawatan luka (kassa, plester, balutan, kapas)",
    "RESPIRATORIUM": "Jalan napas & oksigenasi (ET tube, masker O2, suksi)",
    "UROLOGI": "Perawatan kemih / stoma (kateter, urine bag)",
    "LABORATORIUM": "Laboratorium (tabung, pipet, reagen wadah)",
    "STERILISASI_CSSD": "Sterilisasi / CSSD (pouches, indikator, enzimatik)",
    "OBGYN": "Obstetri-ginekologi (partus, sectio, speculum)",
    "APD_HIGIENE": "APD & higienitas rutin (handscoon, masker, penutup kepala)",
}

# Spesifikasi spuit 1 cc (imunisasi/insulin/tuberkulin) vs klinis besar
SPEC_1CC = ["1 cc", "1cc", "1 ml", "1ml", "tuberculin", "insulin"]

RULES = [
    # (grup, keyword-list). Dievaluasi berurutan; satu baris boleh dapat >1 grup.
    # Catatan: "swab" generik sengaja tidak dipakai agar Gauze/Alcohol Swab
    # tidak salah masuk COVID; dipakai kata spesifik saja.
    ("PANDEMI_COVID", ["genose", "ge nose", "virus sampling", "nasal swab",
                       "one swabs", "rapid", "rapit", "antigen", "pcr"]),
    ("PANDEMI_COVID", ["airvo", "optiflow", "ventilator"]),
    ("PANDEMI_COVID", ["baju apd", "coverall", "hazmat", "cocerall"]),
    ("IMUNISASI", ["auto destruct", "safety box"]),
    ("NEONATAL", ["neopuff", "infant warmer", "ncpap", "neonatal", "umbilical cord",
                  "tali pusat", "resuscitator", "bubble"]),
    ("CAPITAL_EQUIP", ["blanketrol", "patient monitor", "examination table",
                       "stethoscope", "tensimeter", "warmer", "monitor heyer",
                       "mindray", "termometer", "thermometer", "thermogun",
                       "vein finder", "doppler", "otoscope", "suction unit",
                       "kursi roda", "hearing aid", "beneFusion", "perfusor",
                       "infuse pump", "terufusion"]),
    ("RESPIRATORIUM", ["endotracheal", "intubation", "stylet", "laryngeal mask",
                       "lma ", "lma supreme", "guedel", "ambu bag", "rebreathing bag",
                       "kantung nafas", "suction catheter", "close suction",
                       "closed suction", "oxygen mask", "masker o2", "nasal cannula",
                       "oxygen nasal", "oksigen", "oxygen tubing", "tabung oxygen",
                       "nebulizer", "tracheostomy", "tracheastomy", "catrheter mount",
                       "catheter mount", "laryngoscope", "breathing circuit",
                       "jackson rees", "jeckson", "hmef", "hmes",
                       "nasal prong", "nasal silicon",
                       "supraglottic", "mouthpiece", "bougie", "mf cath"]),
    ("UROLOGI", ["foley", "folley", "catherer", "urine bag", "urine bags",
                 "pot urine", "urinal", "colostomy", "ureteral stent",
                 "hemodialysis", "condom kateter", "condom"]),
    ("LABORATORIUM", ["tabung", "pipet", "beaker", "erlenmeyer", "edta",
                      "gel & clot", "clot activator", "spesimen", "specimen",
                      "sample cotainer", "litmus", "urinometer", "haemocytometer",
                      "ose bulat", "corong kaca", "rak tabung", "rak pewarna",
                      "centrifuge", "golongan darah", "kapiler", "nierbeken"]),
    ("STERILISASI_CSSD", ["pouches", "tyvek", "gusset", "reels", "indicator tape",
                          "sterilization", "enzymatic", "cosmozyme", "virusolve",
                          "biocide", "baki instrumen", "autoclave", "sofnolime"]),
    ("OBGYN", ["partus", "paket sc", "paket sectio", "cesarean", "episiotomi",
               "speculum", "uterine", "laminaria", "pesarium", "obgyn"]),
    ("APD_HIGIENE", ["handscoon", "handscun", "glove", "sarung tangan",
                     "masker", "face mask", "n95", "kn95", "duckbill",
                     "bouffant", "mob cap", "mobcap", "mocap", "nurse cap",
                     "topi oprasi", "surgical hat", "visor", "kacamata",
                     "shoe cover", "sepatu ", "apron", "long sleeve",
                     "hand sanitizer", "handsanitaizer", "handsanitizer",
                     "single use bonnet", "id band", "gelas ukur"]),
    ("KARHUTLA", ["n95", "kn95"]),
    ("PANDEMI_COVID", ["n95", "kn95", "surgical face mask", "face mask"]),
    ("DBD", ["iv cannula", "iv catheter", "iv canula", "surflo", "infusion set",
             "infus", "blood tranfusion", "blood transfusion", "transfusion set",
             "wing infusion", "stop cock", "stopcock", "extention tube",
             "extension tube", "spike", "tiang infus", "cvc", "certofix",
             "bionector", "plastik obat", "intrafix", "romovac", "romo vac"]),
    ("PANDEMI_COVID", ["oxygen mask", "nasal oxygen", "oxygen nasal"]),
    ("FARMASI", ["kapsul kosong", "povidine", "alkohol", "alcohol", "rivanol",
                 "antiseptic", "hand wash", "vaselin", "vaseline swab",
                 "chlorhexidine", "xylocaine", "cathejell"]),
    ("BEDAH_OK", ["electro surgical", "surgical conecting", "surgical connecting",
                  "steril surgical glove", "gammex", "silk", "chromic",
                  "polypropylene", "polypropylane", "nylon", "assucryl",
                  "needle", "spalak", "gypsona", "kassa steril", "kassa steril",
                  "lateks", "latex", "pinset", "forcep", "magil", "metzenbaum",
                  "debakey", "scissors", "gunting episiotomi", "hecting",
                  "fostryl", "scalpel", "ceratome", "crescent knife", "stab knife",
                   "cauter", "pencil cauter", "grounding pad", "equispon",
                   "lap sponge", "universal set", "bisturi",
                   "spinocan", "mesh"]),
    ("DIAGNOSTIK", ["ecg", "usg", "ultrasound", "film", "dihl", "drystar",
                    "blood lancet", "accu-check", "accu chek", "accu-chek",
                    "glucose", "oximeter",
                    "finger", "electrode", "elektroda", "tes narkoba", "hcg"]),
    ("PERAWATAN_LUKA", ["kassa", "gauze", "gauze swab", "bandage", "plester",
                        "plaster", "tape", "hypafix", "leukotape", "dermafix",
                        "cutimed", "soffban", "dresing", "dressing", "wound",
                        "kapas", "underpad", "equispon", "hydrocolloid",
                        "transparent film", "finger bandage", "stikpan",
                        "high elastic", "zinc oxide", "adhesive",
                        "kasa hidrofil"]),
]


def norm(s):
    return " ".join(str(s).split()).lower()


def classify_row(nama, spek):
    """Klasifikasi satu baris transaksi -> daftar grup."""
    n = norm(nama)
    s = norm(spek)
    # --- Keluarga spuit: keputusan berbasis spesifikasi ---
    if "syringe" in n or "syiringe" in n:
        if "auto destruct" in n:
            return ["IMUNISASI", "SPUIT_INJEKSI"]
        if any(k in s for k in SPEC_1CC):
            return ["IMUNISASI", "SPUIT_INJEKSI"]
        return ["SPUIT_INJEKSI"]
    if "safety box" in n:
        return ["IMUNISASI"]

    groups = []
    for g, kws in RULES:
        if any(k in n for k in kws):
            if g not in groups:
                groups.append(g)
    # Jahitan "Plain <angka>" (Plain 0, Plain 2/0, ...) -> BEDAH_OK
    if not groups and re.search(r"\bplain\s*\d", n):
        groups.append("BEDAH_OK")
    if not groups:
        groups.append("OPERASIONAL_RS")
    return groups


EVENTS = [
    ("delta_2021", "Delta COVID (Jul-Agu 2021)", ["2021-07", "2021-08"]),
    ("omicron_2022", "Omicron COVID (Feb-Mar 2022)", ["2022-02", "2022-03"]),
    ("bian_2022", "BIAN imunisasi (Agu 2022)", ["2022-08"]),
    ("karhutla_2023", "Karhutla El Nino (Sep-Okt 2023)", ["2023-09", "2023-10"]),
    ("dbd_season", "Musim DBD (Jan-Apr, rata 2021-2025)",
     ["2021-01", "2021-02", "2021-03", "2021-04",
      "2022-01", "2022-02", "2022-03", "2022-04",
      "2023-01", "2023-02", "2023-03", "2023-04",
      "2024-01", "2024-02", "2024-03", "2024-04",
      "2025-01", "2025-02", "2025-03", "2025-04"]),
]


def main():
    df = pd.read_excel(DATA)
    df["jual_total_fak"] = df["jual_total_fak"].fillna(0)
    df = df.dropna(subset=["d_jual_nofak", "tanggal", "barang_nama", "d_jual_qty"])
    df["month"] = df["tanggal"].dt.to_period("M").astype(str)
    spek = df["barang_spesifikasi"].fillna("") if "barang_spesifikasi" in df.columns else ""

    # --- Atribusi nilai per grup (grain diperbaiki) ---
    # jual_total_fak BERULANG di tiap baris item satu faktur (rata-rata 3,43
    # item/faktur) sehingga TIDAK boleh dijumlahkan per baris (inflasi ~5,4x).
    # Nilai item = kolom `jumlah` (qty x harga satuan; cocok ~99,9% dengan
    # total faktur). Setiap faktur diatribusikan SEKALI ke grup dominan
    # menurut bobot nilai item di dalamnya -> jumlah total antar grup ==
    # total nilai faktur unik (tanpa double counting).
    if "jumlah" in df.columns:
        df["_val"] = pd.to_numeric(df["jumlah"], errors="coerce").fillna(0.0)
    else:
        df["_val"] = 0.0

    # --- Klasifikasi level baris ---
    df["groups"] = [classify_row(a, b) for a, b in zip(df["barang_nama"], spek)]
    ex = df.explode("groups")

    fv = ex.groupby(["d_jual_nofak", "groups"])["_val"].sum().unstack(fill_value=0.0)
    dom = fv.idxmax(axis=1).rename("dom_group")
    dom[fv.sum(axis=1) <= 0] = "OPERASIONAL_RS"

    inv = (df.drop_duplicates("d_jual_nofak")[["d_jual_nofak", "jual_total_fak", "month"]]
             .merge(dom, left_on="d_jual_nofak", right_index=True, how="left"))
    inv["dom_group"] = inv["dom_group"].fillna("OPERASIONAL_RS")

    # --- Statistik per grup ---
    group_stats = {}
    for g, sub in inv.groupby("dom_group"):
        group_stats[g] = {"total": float(sub["jual_total_fak"].sum()),
                          "n_produk": int(ex[ex["groups"] == g]["barang_nama"].nunique()),
                          "n_faktur": int(sub["d_jual_nofak"].nunique())}
    grand = sum(s["total"] for s in group_stats.values())

    print("=== Statistik per grup pemicu (total nilai) ===", flush=True)
    for g, s in sorted(group_stats.items(), key=lambda kv: -kv[1]["total"]):
        print("  {:16s} {:>12,.0f} M ({:>4.1f}%)  produk={:<4} faktur={:>6,}  {}".format(
            g, s["total"] / 1e6, 100 * s["total"] / grand,
            s["n_produk"], s["n_faktur"], GROUPS.get(g, "?")), flush=True)
    resid = group_stats.get("OPERASIONAL_RS", {}).get("total", 0.0)
    print("\nPorsi residual OPERASIONAL_RS: {:.1f}% dari total nilai".format(
        100 * resid / grand), flush=True)

    # --- Mapping produk -> union grup (skema JSON tetap sama) ---
    mapping = {}
    for name, sub in df.groupby("barang_nama"):
        gs = []
        for gl in sub["groups"]:
            for g in gl:
                if g not in gs:
                    gs.append(g)
        mapping[name] = gs

    # --- Penjualan bulanan per grup (atribusi dominan per faktur) ---
    monthly = (inv.groupby(["month", "dom_group"])["jual_total_fak"].sum()
                 .reset_index().rename(columns={"dom_group": "group", "jual_total_fak": "total"}))
    monthly.to_csv(OUT_CSV, index=False)

    # --- Deteksi spike empiris per event (nilai level-item `jumlah`) ---
    prod_month = df.groupby(["barang_nama", "month"])["_val"].sum().reset_index()
    spike_out = {}
    for key, label, months in EVENTS:
        hits = []
        for name, psub in prod_month.groupby("barang_nama"):
            in_win = psub[psub["month"].isin(months)]["_val"].sum()
            out_win = psub[~psub["month"].isin(months)]
            if len(out_win) == 0 or in_win <= 0:
                continue
            baseline = out_win["_val"].median()
            if baseline <= 0:
                continue
            ratio = in_win / baseline
            if ratio >= 3.0 and in_win >= 50e6:
                hits.append({"barang_nama": name, "total_window": float(in_win),
                             "baseline_median": float(baseline), "ratio": float(ratio),
                             "groups": mapping[name]})
        hits.sort(key=lambda h: -h["ratio"])
        spike_out[key] = {"label": label, "months": months, "hits": hits[:20]}
        print("\n=== Spike {} ({} bulan) ===".format(label, ", ".join(months)), flush=True)
        for h in hits[:20]:
            print("  x{:>6.1f}  {:>12,.0f} M  (baseline {:>9,.0f} M)  {}  [{}]".format(
                h["ratio"], h["total_window"] / 1e6, h["baseline_median"] / 1e6,
                h["barang_nama"], ", ".join(h["groups"])), flush=True)

    out = {
        "groups": GROUPS,
        "group_stats": group_stats,
        "product_mapping": mapping,
        "event_spikes": spike_out,
        "catatan_revisi": (
            "Klasifikasi level baris (nama+spesifikasi); spuit dipilah per "
            "spesifikasi (1cc/Auto Destruct -> IMUNISASI, sisanya SPUIT_INJEKSI); "
            "typo ditutup; taksonomi diperluas agar OPERASIONAL_RS murni residual. "
            "Grain nilai diperbaiki: jual_total_fak berulang per baris item dan "
            "tidak boleh dijumlahkan per baris; setiap faktur diatribusikan sekali "
            "ke grup dominan menurut bobot nilai item (kolom jumlah), sehingga "
            "total antar grup == total nilai faktur unik; spike memakai nilai "
            "level-item."
        ),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    with open(OUT_SPIKE, "w", encoding="utf-8") as f:
        json.dump(spike_out, f, indent=2, default=str)
    print("\nSelesai. Disimpan ke " + OUT_JSON + ", " + OUT_CSV + ", " + OUT_SPIKE)


if __name__ == "__main__":
    main()
