# Checklist Anomali Data Penjualan Alkes

Dibuat: 28 Agustus 2026. Hasil diagnostik otomatis dari `scripts/data_anomaly_check.py` (output `results/data_anomaly_check.json`). Tujuan: mendokumentasikan potensi kesalahan pengukuran/pencatatan yang bisa mengubah kesimpulan analisis, dan status tindak lanjutnya.

Data: `view_penjualan_detail data hingga oktober.xlsx` — 46.822 baris, 13.661 faktur unik, 230 pelanggan, rentang 2020-07-01 s.d. 2025-10-27.

---

## Ringkasan Status

| # | Anomali | Jumlah | Status | Dampak |
|---|---|---|---|---|
| 1 | Harga per qty ekstrem | 102 baris | **SELESAI** (sample/gratis + produk modal sah) | Diabaikan |
| 2 | Qty negatif / retur | 0 | Bersih | - |
| 3 | Jumlah=0 tapi qty>0 | 8 baris (6 faktur) | **SELESAI** (sample/gratis) | Diabaikan |
| 4 | Duplikasi nofak | 42.170 baris | Normal (multi-baris item) | - |
| 5 | Satuan campur produk kunci | 7 produk | **PERLU NORMALISASI** | Tinggi |
| 6 | Nama barang pecah | 174 nama | **SELESAI** (normalisasi aman; typo kandidat perlu kurasi manual) | Sedang |
| 7 | Jumlah vs total faktur | 90% selisih | Normal (total diulang) | - |
| 8 | Tanggal null / di luar rentang | 2 null | Minor | Rendah |
| 9 | Koordinat di luar Sumsel | 2.313 baris | Normal (di luar 16 region) | - |

---

## 1. Harga per qty ekstrem — SELESAI (tidak berdampak material)

**Temuan:** 81 baris harga per qty <= 1 rupiah; 102 baris < 100 rupiah; 6 baris > 50 juta rupiah.

**Keputusan (28 Agu 2026):**
- **81 baris harga <= 1 rupiah = sample/gratis** (bukan error input). Pola: Masker 1 rupiah/box, Hand Sanitizer 1 rupiah/botol, BloodLancets 3000 pcs jumlah=0. Ini barang diberikan dengan harga nominal/simbolis.
  - 48 faktur terlibat; 40 di antaranya **sample murni** (semua baris nominal).
  - Nilai total faktur sample murni = **8.376 rupiah** dari total 99,97 M = **0,000%**. Dampak ke agregasi nilai diabaikan.
- **6 baris harga > 50 juta = produk modal (CAPITAL EQUIPMENT) yang sah**, bukan error: GeNose C19 (88,5 jt), Bubble Infant nCPAP (179 jt), Infant Warmer Giraffe (420 jt), Airvo Optiflow (152 jt), Blanketrol III (441 jt), Servis Juli 2024 (77,8 jt). Semua satuan Unit, kategori modal (GE, Medin, Fisher & Paykel, Gentherm). Sudah ditangani `scripts/expensive_products_analysis.py`.

**Status:** Tidak perlu mengubah pipeline agregasi. Sample/gratis berdampak 0,000% nilai; produk modal adalah penjualan riil yang sah. Tidak ada baris yang perlu di-exclude/dikoreksi untuk analisis nilai.

---

## 2. Qty negatif / retur — BERSIH

**Temuan:** 0 baris qty negatif. Tidak ada retur yang tercatat sebagai qty negatif.

**Status:** Tidak ada tindakan.

---

## 3. Jumlah=0 tapi qty>0 — SELESAI (sample/gratis, dampak kecil)

**Temuan:** 8 baris (6 faktur) dengan `jumlah` = 0 tapi `d_jual_qty` > 0, total qty 6.211. Ini barang sample/gratis (diberikan tanpa nilai): BloodLancets 3000, One Swabs 3000, Kantung Nafas 200, dll.

**Keputusan (28 Agu 2026):** Baris ini sample/gratis, bukan penjualan riil. Dampak ke agregasi qty kecil (6.211 unit dari total qty yang sangat besar) dan tidak berpengaruh ke agregasi nilai (jumlah=0). Tidak perlu di-exclude untuk analisis nilai; untuk analisis qty per produk, dampaknya diabaikan.

---

## 4. Duplikasi nofak — NORMAL

**Temuan:** 42.170 baris duplikat nofak, 9.009 faktur terlibat. Ini karena **satu faktur punya banyak baris item** (median 2, maks 85 baris per faktur).

**Status:** Normal. Pipeline sudah `drop_duplicates("d_jual_nofak")` sebelum agregasi. Tidak ada tindakan.

---

## 5. Satuan campur produk kunci — PERLU NORMALISASI

**Temuan:** banyak produk kunci dijual campur satuan (pola yang sama dengan masker):

| Produk | Satuan (n baris) |
|---|---|
| handscoon | Box (1132), Pair (500), Pack (12) |
| spuit | Pcs (3234), Box (243) |
| infusion_set | Pcs (980) |
| iv_catheter | Pcs (886) |
| masker | Box (629), Pcs (147) |
| kassa | Roll (326), Box (288), Pack (144) |
| plester | Box (45), Roll (10) |

**Risiko:** qty mentah antar satuan TIDAK bisa dijumlahkan langsung (1 box bisa berisi puluhan pcs). Analisis qty per produk yang tidak menormalkan satuan bisa menyesatkan.

**Tindak lanjut:**
- [ ] Untuk setiap produk kunci, tentukan isi per kemasan (pcs/box, pcs/pack, pcs/roll, dll).
- [ ] Normalisasi qty ke satuan dasar (pcs) sebelum agregasi qty.
- [ ] Sudah dilakukan untuk masker (surgical=50, N95/KN95=10) — perlu diperluas ke handscoon, spuit, kassa, plester.

**Status per produk (28 Agu 2026):**

| Produk | Status | Catatan |
|---|---|---|
| masker | SELESAI (konversi ke pcs) | surgical=50, N95/KN95=10 |
| handscoon | SELESAI (per-satuan) | isi box tidak diketahui dari data; normalisasi merek; qty per Box/Pair/Pack terpisah |
| spuit | SELESAI (konversi ke pcs) | isi 100 pcs/box (rasio harga ~90-100x); produk khusus (pre-filled/insulin) di-exclude; normalisasi merek |
| kassa | SELESAI (per produk) | Roll/Box/Pack adalah produk BERBEDA (Kassa roll, Kassa Steril box, Gauze Swab pack), bukan isi box sama; tidak ada konversi valid; analisis per nama x satuan |
| plester | SELESAI (per produk) | Roll/Box produk/format BERBEDA (Hypafix roll, Zinc Oxide box); normalisasi merek (OneMed./.BSN.); analisis per nama x satuan |
| infusion_set | SELESAI (Pcs konsisten) | sudah Pcs, tidak perlu konversi; normalisasi merek (Meddis../G E A); qty per merek |
| iv_catheter | SELESAI (Pcs konsisten) | sudah Pcs, tidak perlu konversi; normalisasi merek (GEA/Gea./G E A/GeA, HEALTHCARE., Meddis); qty per ukuran G |

---

## 6. Nama barang pecah — SELESAI (normalisasi aman; typo kandidat perlu kurasi)

**Temuan:** 1.318 nama barang asli -> 1.144 setelah normalisasi basic (strip, lowercase, spasi tunggal). Selisih 174 nama = produk yang sama tercatat dengan nama berbeda (typo, casing, spasi).

**Keputusan (28 Agu 2026):**
- **Normalisasi basic** (strip + lowercase + spasi tunggal) sudah diterapkan di pipeline: 1.318 -> 1.144.
- **Normalisasi tanda baca** (hapus trailing `-`, `.`, `,`, `;`, `:` dan normalisasi spasi `%`): menggabung 9 kelompok lagi -> 1.135 nama. Aman, tidak mengubah makna. Contoh: `alkohol 70 %`->`alkohol 70%`, `kit 3a -`->`kit 3a`, `film island dresing.`->`film island dresing`.
- **Typo kandidat (170 pasangan fuzzy >= 0.90) TIDAK digabung otomatis.** Banyak pasangan mirip adalah produk/ukuran/tipe BERBEDA (mis. `polypropylene 3/0` vs `4/0` ukuran benang jahit; `2-way` vs `3-way foley`; `non-sterile` vs `sterile`; `n95` vs `kn95`; `monopolar` vs `bipolar`; ukuran `fr/G/s/m/l`). Menggabung otomatis berisiko menciptakan error agregasi.
- **Kandidat typo murni** (produk sama, hanya salah eja, tanpa perbedaan ukuran/tipe) didokumentasikan di `results/nama_barang_normalisasi.json` untuk **kurasi manual** oleh peneliti: `handscoon non steril/sterile`, `2 way foley/folley/catherer`, `kassa/kasa hidrofil`, `mask nebulizer/nebulize`, `syringe/syiringe`, `surflo/surfloo`, `nasal cannula/canula`, `alcohol/alkohol swab`, `povidine/povidone`, `rapit/rapid tes`, `oxygen mask non rebreathing/rebrithing`, `n95 mask/maks`, `blood transfusion/tranfusion`, dll.

**Status:** Normalisasi aman (basic + tanda baca) diterapkan. Penggabungan typo murni menunggu kurasi manual peneliti untuk menghindari error agregasi.

---

## 7. Jumlah vs total faktur — NORMAL

**Temuan:** 90,03% baris punya selisih `jumlah` vs `jual_total_fak` > 1.

**Status:** Normal. `jual_total_fak` adalah **total per faktur yang diulang di setiap baris item** (semua 13.661 faktur punya 1 nilai total). Contoh: faktur 601210009 punya 2 baris (486rb + 1.134rb = 1.620rb = jual_total_fak). Tidak ada tindakan.

---

## 8. Tanggal null / di luar rentang — MINOR

**Temuan:** 2 baris tanggal null; 0 sebelum 2020; 0 setelah 2025.

**Tindak lanjut:**
- [ ] Periksa 2 baris tanggal null — apakah di-exclude atau diisi.

---

## 9. Koordinat di luar Sumsel — NORMAL

**Temuan:** 2.313 baris lat di luar rentang Sumsel, 1.628 baris lon di luar rentang.

**Status:** Normal. Ini pelanggan **di luar 16 region studi** (Jakarta, Jambi, Sungailiat/Bangka, OKU Selatan, dll) yang ada di data mentah tapi tidak termasuk panel 16 kab/kota Sumsel. Pipeline sudah memfilter ke 16 region inti. Tidak ada tindakan.

---

## Prioritas Tindak Lanjut

1. **Tinggi:** Normalisasi satuan produk kunci (#5) — SELESAI untuk semua 7 produk (masker, handscoon, spuit, kassa, plester, infusion_set, iv_catheter).
2. **Tinggi:** Periksa harga per qty ekstrem (#1) — SELESAI. Sample/gratis (0,000% nilai) + produk modal sah; tidak perlu filter.
3. **Sedang:** Normalisasi nama barang (#6) — SELESAI (normalisasi aman basic+tanda baca). Typo murni kandidat menunggu kurasi manual peneliti.
4. **Rendah:** Keputusan baris sample/gratis (#3) — SELESAI (sample/gratis, dampak kecil) dan tanggal null (#8) — 2 baris, minor.

---

## Catatan

- Diagnostik ini otomatis dan dapat dijalankan ulang: `E:\pyvenv_geo\Scripts\python.exe scripts\data_anomaly_check.py`
- Output JSON: `results/data_anomaly_check.json`
- Temuan yang sudah ditindaklanjuti: masker (satuan Box/Pcs dikonversi ke pcs, lihat `scripts/masker_karhutla_analysis.py`).
- Normalisasi nama barang: `scripts/nama_barang_normalisasi.py` -> `results/nama_barang_normalisasi.json` (daftar typo kandidat untuk kurasi manual).
