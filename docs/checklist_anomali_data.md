# Checklist Anomali Data Penjualan Alkes

Dibuat: 28 Agustus 2026. Hasil diagnostik otomatis dari `scripts/data_anomaly_check.py` (output `results/data_anomaly_check.json`). Tujuan: mendokumentasikan potensi kesalahan pengukuran/pencatatan yang bisa mengubah kesimpulan analisis, dan status tindak lanjutnya.

Data: `view_penjualan_detail data hingga oktober.xlsx` — 46.822 baris, 13.661 faktur unik, 230 pelanggan, rentang 2020-07-01 s.d. 2025-10-27.

---

## Ringkasan Status

| # | Anomali | Jumlah | Status | Dampak |
|---|---|---|---|---|
| 1 | Harga per qty ekstrem | 102 baris | **PERLU DITINDAKLANJUTI** | Tinggi |
| 2 | Qty negatif / retur | 0 | Bersih | - |
| 3 | Jumlah=0 tapi qty>0 | 8 baris (6 faktur) | **PERLU KEPUTUSAN** | Rendah |
| 4 | Duplikasi nofak | 42.170 baris | Normal (multi-baris item) | - |
| 5 | Satuan campur produk kunci | 7 produk | **PERLU NORMALISASI** | Tinggi |
| 6 | Nama barang pecah | 174 nama | **PERLU NORMALISASI** | Sedang |
| 7 | Jumlah vs total faktur | 90% selisih | Normal (total diulang) | - |
| 8 | Tanggal null / di luar rentang | 2 null | Minor | Rendah |
| 9 | Koordinat di luar Sumsel | 2.313 baris | Normal (di luar 16 region) | - |

---

## 1. Harga per qty ekstrem — PERLU DITINDAKLANJUTI

**Temuan:** 81 baris harga per qty <= 1 rupiah; 102 baris < 100 rupiah; 6 baris > 50 juta rupiah.

**Risiko:** harga per qty ekstrem bisa jadi salah input (harga 1 rupiah jelas error) atau produk bernilai sangat tinggi yang sah. Jika dipakai untuk analisis produk bernilai tinggi / outlier, bisa menyesatkan.

**Tindak lanjut:**
- [ ] Periksa 81 baris harga <= 1 rupiah — apakah error input (qty salah, harga salah) atau sample/gratis.
- [ ] Periksa 6 baris harga > 50 juta — apakah produk modal (CAPITAL_EQUIP) yang sah.
- [ ] Putuskan apakah baris error di-exclude atau dikoreksi sebelum analisis produk.

---

## 2. Qty negatif / retur — BERSIH

**Temuan:** 0 baris qty negatif. Tidak ada retur yang tercatat sebagai qty negatif.

**Status:** Tidak ada tindakan.

---

## 3. Jumlah=0 tapi qty>0 — PERLU KEPUTUSAN

**Temuan:** 8 baris (6 faktur) dengan `jumlah` = 0 tapi `d_jual_qty` > 0, total qty 6.211. Ini kemungkinan barang sample/gratis (diberikan tanpa nilai).

**Risiko:** jika dihitung dalam agregasi qty, menambah qty tanpa nilai; jika dihitung nilai, tidak berpengaruh.

**Tindak lanjut:**
- [ ] Konfirmasi apakah baris ini sample/gratis.
- [ ] Putuskan apakah di-exclude dari analisis qty (karena bukan penjualan riil).

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
| kassa | BELUM | - |
| plester | BELUM | - |

---

## 6. Nama barang pecah — PERLU NORMALISASI

**Temuan:** 1.317 nama barang asli -> 1.143 setelah normalisasi (strip, lowercase, spasi tunggal). Selisih 174 nama = produk yang sama tercatat dengan nama berbeda (typo, casing, spasi).

**Risiko:** produk yang sama bisa terpecah menjadi beberapa entitas, mengaburkan agregasi per produk.

**Tindak lanjut:**
- [ ] Terapkan normalisasi nama (strip + lowercase + spasi tunggal) di semua analisis per produk.
- [ ] Periksa 174 nama yang pecah untuk typo yang perlu dipetakan manual.

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

1. **Tinggi:** Normalisasi satuan produk kunci (#5) — pola masker terulang di handscoon, spuit, kassa, plester. Ini berdampak langsung pada analisis qty.
2. **Tinggi:** Periksa harga per qty ekstrem (#1) — bisa mengganggu analisis produk bernilai tinggi.
3. **Sedang:** Normalisasi nama barang (#6) — 174 nama pecah.
4. **Rendah:** Keputusan baris sample/gratis (#3) dan tanggal null (#8).

---

## Catatan

- Diagnostik ini otomatis dan dapat dijalankan ulang: `E:\pyvenv_geo\Scripts\python.exe scripts\data_anomaly_check.py`
- Output JSON: `results/data_anomaly_check.json`
- Temuan yang sudah ditindaklanjuti: masker (satuan Box/Pcs dikonversi ke pcs, lihat `scripts/masker_karhutla_analysis.py`).
