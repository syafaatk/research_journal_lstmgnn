# Temuan Analisis — Revisi Paper LSTM-GNN Alkes Sumsel

Dokumen ini merangkum SEMUA temuan analisis yang telah diverifikasi (per 27 Agustus 2026).
Setiap temuan menyertakan script dan file hasil untuk jejak audit.

---

## 1. Eksperimen Fitur Kalender (Temuan Utama Model)

**Script:** `scripts/exp_features.py` (V1/V2), `scripts/v3_rs_features.py` (V3)
**Hasil:** `results/exp_features.json`, `results/v3_rs_features.json`, cache `results/v3_preds_seed{n}.npy`
**Setup:** 16 region, window 30 hari, train s.d. 2023, validasi 2024, test 2025, ensemble 3 seeds (42/7/123). Libur nasional resmi SKB 3 Menteri 2020-2025 di `scripts/holidays.py`.

| Varian | Fitur tambahan | Kanal | R2 | RMSE | DM vs base | p | Palembang RMSE |
|---|---|---|---|---|---|---|---|
| Base | - | 2 | 0,0523 | 8.002.603 | - | - | 23.964.716 |
| V1 | weekend + libur + fase pandemi | 6 | 0,0528 | 7.990.984 | -1,740 | 0,082 | 23.923.979 |
| V2 | day-of-week one-hot + libur + fase | 12 | 0,0612 | 7.762.318 | -6,655 | <0,001 | 22.689.018 (-5,3%) |
| **V3** | rs config (hl=219, hg=107, do=0,1734, lr=1,52e-3, bs=32) + fitur V2 | 12 | **0,0666** | **7.609.122** | **-6,307** | **<0,001** | **21.842.944 (-8,9%)** |

**Temuan:**
1. **Day-of-week one-hot jauh lebih informatif daripada weekend biner.** V2 (dow one-hot) signifikan (p<0,001) sementara V1 (weekend biner) tidak (p=0,082). Konsisten dengan analisis residual per hari: model overpredict hari kerja (Senin +22,1 jt) — pola hari kerja vs akhir pekan tidak cukup ditangkap biner weekend.
2. **V3 (random-search config + fitur V2) adalah model terbaik sejauh ini:** R2 0,0666, RMSE 7.609.122, MAE 1.774.426, bootstrap CI RMSE [6.506.764, 8.795.473]. Per seed: 42=0,0694, 7=0,0649, 123=0,0652 (stabil).
3. **Perbaikan terbesar di Palembang** (region dominan): RMSE turun 8,9% ke 21.842.944.
4. Fitur kalender menangkap variasi temporal harian yang tidak bisa ditangkap model base.

---

## 2. Faktor Eksternal Data BPS (Tiga Lapis Bukti)

### 2.1 Korelasi pooled (semua region-tahun)

**Script:** `scripts/bps_analysis.py` | **Hasil:** `results/bps_analysis.json`

| Faktor | r | p | n | Kesimpulan |
|---|---|---|---|---|
| Belanja pemerintah kab/kota | 0,401 | 0,001 | 64 | signifikan (efek skala) |
| Penduduk | 0,164 | 0,111 | 96 | tidak |
| DBD per 100.000 | -0,051 | 0,619 | 96 | tidak |
| Jarak dari Palembang | -0,099 | 0,716 | 16 | tidak |
| Luas daerah | -0,341 | 0,196 | 16 | tidak |
| Realisasi belanja barang&jasa provinsi | -0,143 | 0,787 | 6 | tidak |

Catatan: file belanja BPS berisi 4 kolom (2020-2023) - script sebelumnya salah slice 3 kolom, sudah dikoreksi. Format: 2020/2022/2023 = ribu rupiah, 2021 = rupiah (dikoreksi /1000).

### 2.2 Korelasi within-region (menghilangkan efek skala)

**Analisis:** demean per region. Data 2020-2023 (BPS) + 2024-2025 (DJPK), semua dalam rupiah.

| Analisis | r | p | n |
|---|---|---|---|
| Pooled 2020-2025 | +0,380 | <0,001 | 96 |
| **Within-region (demeaned)** | **+0,058** | **0,576** | 96 |

**Temuan:** korelasi belanja yang tampak signifikan adalah **murni efek level antar region** (region besar = belanja besar = penjualan besar). Variasi belanja antar tahun dalam region yang sama TIDAK memprediksi penjualan, bahkan dengan data lengkap 6 tahun. Level statis sudah ditangkap model dari data penjualan itu sendiri.

### 2.3 Korelasi residual model V3 vs faktor BPS 2025

**Script:** `scripts/bps_residual_analysis.py` | **Hasil:** `results/bps_residual_analysis.json`
Residual = y_true - y_pred model V3 (test 2025), rata-rata per region (n=16).

| Faktor | r residual | p |
|---|---|---|
| Penduduk | +0,066 | 0,807 |
| DBD/100k | +0,274 | 0,305 |
| RS/puskesmas | +0,478 | 0,061 |
| Sarana kesehatan | +0,510 | 0,044* |
| Jarak | -0,134 | 0,621 |
| Luas | -0,341 | 0,196 |

*Artefak: file sarana_2025 hanya berisi 6 region bernilai (Palembang 25, sisanya 0-6); tanpa Palembang r=-0,158 p=0,800.

**Temuan:** TIDAK ADA faktor BPS yang menjelaskan kesalahan prediksi model.

### 2.4 APBD per kab/kota (portal DJPK Kemenkeu) — fokus Belanja Barang & Jasa

**Script:** `scripts/apbd_portal_fetch.py` (fetch 2020-2025), `scripts/apbd_bj_analysis.py` (analisis BJ), `scripts/apbd_2025_corr.py`, `scripts/apbd_combined_analysis.py`
**Hasil:** `results/apbd_2020..2025_per_pemda.json`, `results/apbd_bj_analysis.json`, `results/apbd_2025_corr.json`, `results/apbd_combined_analysis.json`
**Sumber:** djpk.kemenkeu.go.id, provinsi=06, realisasi 2020-2024 = akhir tahun (periode=12), 2025 = s.d. Oktober (periode=10), data SIKD per 19 Agu 2026. File user `data/data_apbd.xls` terverifikasi identik dengan portal.

Belanja Barang & Jasa (paling relevan untuk pengadaan alkes), semua tahun dari DJPK (konsisten), n=96:

| Analisis | r | p |
|---|---|---|
| Pooled 2020-2025 | +0,345 | 0,0006 (efek skala) |
| **Within-region (demeaned)** | **+0,103** | **0,318 (TIDAK signifikan)** |
| Cross-sectional 2020 | +0,496 | 0,051 (marginal) |
| Cross-sectional 2021 | +0,409 | 0,116 |
| Cross-sectional 2022 | +0,471 | 0,066 (marginal) |
| Cross-sectional 2023 | +0,162 | 0,550 |
| Cross-sectional 2024 | +0,384 | 0,142 |
| Cross-sectional 2025 | +0,252 | 0,347 |

Belanja Daerah total (konfirmasi): pooled r=0,380 p<0,001; within-region r=0,058 p=0,576.

**Temuan:** SEMUA cross-sectional tidak signifikan di 0,05; within-region tidak signifikan. Belanja APBD (baik total maupun barang&jasa) bukan prediktor temporal penjualan — hanya efek skala antar region. Data belanja LENGKAP 2020-2025 tersedia, tapi keputusan TIDAK ada eksperimen ulang dengan fitur BPS tetap valid dengan alasan murni statistik.

### 2.5 Kesimpulan Faktor X

**Belanja APBD bukan prediktor penjualan.** Empat lapis bukti konsisten: (1) korelasi pooled = efek skala antar region; (2) within-region (6 tahun, data lengkap) tidak signifikan; (3) cross-sectional tidak ada tahun yang signifikan; (4) residual model tidak berkorelasi dengan faktor BPS. Penjualan alat kesehatan didorong pengadaan/program (anggaran spesifik), bukan anggaran umum daerah. **Keputusan: TIDAK ada eksperimen ulang dengan fitur BPS** — fitur tahunan/statis hampir konstan dalam window 30 hari, tidak memberi informasi baru ke LSTM, dan berisiko overfitting.

### 2.6 Kandidat faktor X: struktur fasilitas kesehatan (indikator kinerja RS 2024)

**Sumber:** `data/Indikator kinerja pelayanan di RS Provinsi Sumatera Selatan.json` (88 RS, tahun 2024) — jumlah tempat tidur, pasien keluar, hari perawatan, BOR, BTO, TOI, ALOS per RS.
**Script:** `scripts/rs_kinerja_2024.py` | **Hasil:** `results/rs_kinerja_2024.json`
**Metode:** 88 RS dipetakan ke 16 region (8 nama RS ambigu diverifikasi via web: Karunia Indah Medika dan Trijaya → Muara Enim, Adellia → Lahat, Sukajadi → Banyuasin, Musi Medika Cendikia → Palembang, Petanang → Lubuk Linggau, Mahyuzahra dan Ar-Royyan → Ogan Ilir; Hermina OPI dan Bunda Medika Jakabaring → Palembang karena operasional di Jakabaring). Agregasi per region: jumlah RS, total tempat tidur, total pasien keluar, hari perawatan, BOR/BTO/TOI/ALOS rata-rata tertimbang tempat tidur. Total 88 RS, 10.161 tempat tidur (direktori Kemenkes mencatat 89 RS / 10.491 TT — JSON tidak mencakup 1 RS baru).

**Korelasi vs penjualan 2024 (cross-sectional, n=16):**

| Faktor | r | p | Kesimpulan |
|---|---|---|---|
| **Jumlah RS** | **+0,669** | **0,005** | **signifikan** |
| Tempat tidur | +0,577 | 0,019 | signifikan |
| Hari perawatan | +0,574 | 0,020 | signifikan |
| Pasien keluar | +0,522 | 0,038 | signifikan |
| BOR | +0,478 | 0,061 | marginal |
| BTO | +0,396 | 0,129 | tidak |
| TOI | +0,046 | 0,867 | tidak |
| ALOS | -0,233 | 0,386 | tidak |

**Korelasi vs residual model V3 (test 2025, n=16):**

| Faktor | r | p |
|---|---|---|
| **Jumlah RS** | **+0,697** | **0,003** |
| Pasien keluar | +0,643 | 0,007 |
| Hari perawatan | +0,618 | 0,011 |
| Tempat tidur | +0,615 | 0,011 |
| BTO | +0,514 | 0,042 |
| BOR | +0,458 | 0,075 (marginal) |

**Sensitivitas:**
- Tanpa Palembang (n=15): jumlah RS vs penjualan 2024 r=+0,596 p=0,019; vs residual r=+0,631 p=0,012 — tetap signifikan, bukan efek satu region.
- Kontrol penduduk (korelasi parsial, n=10): jumlah RS r=+0,648 p=0,043; tempat tidur r=+0,648 p=0,043 — signifikan setelah mengontrol populasi (penduduk sendiri tidak signifikan, r=0,164).

**Konsistensi lintas sumber:** BPS rs_2020-2025 (pooled): RS umum r=+0,298 p=0,003; RS khusus r=+0,422 p<0,001. Residual BPS 2025: RS khusus r=+0,572 p=0,021; desa dengan RS r=+0,510 p=0,044.

**Temuan:** struktur fasilitas kesehatan — jumlah RS, kapasitas tempat tidur, dan volume pasien — adalah kandidat faktor X TERKUAT yang pernah diuji di proyek ini. Konsisten di semua lapis (cross-sectional, residual, tanpa Palembang, kontrol penduduk, lintas sumber data). Mekanisme: RS adalah pembeli alat kesehatan terbesar; lebih banyak RS dan lebih banyak pasien = lebih banyak pengadaan. Efisiensi operasional (TOI, ALOS) tidak relevan; yang penting keberadaan dan skala layanan. Ini memperkuat kesimpulan naskah: penjualan didorong struktur layanan kesehatan, bukan anggaran umum daerah.

**Keterbatasan:** data kinerja RS hanya 1 tahun (2024) sehingga bukti bersifat struktural cross-sectional, bukan temporal; jumlah RS hampir statis antar tahun sehingga tidak bisa menjelaskan variasi harian; n=16 kecil dan banyak faktor diuji (risiko false positive, perlu koreksi multiple testing). Data ini TIDAK dipakai sebagai fitur model (konsisten dengan keputusan 2.5: fitur statis hampir konstan dalam window 30 hari).

### 2.7 Anggaran barang & jasa Dinkes Provinsi Sumsel (uji faktor anggaran)

**Sumber:** 3 file JSON dari user (alokasi 2022, realisasi 2023, alokasi 2024 — tingkat Dinkes Provinsi) + verifikasi silang Laporan Monev TW IV Dinkes 2022 (`data/monev_dinkes_2022.pdf`), LKjIP Dinkes 2023 (`data/lakip_dinkes_2023.pdf`), LKjIP Dinkes 2024 (`data/lakip_dinkes_2024.pdf`), satudata1.sumselprov.go.id (realisasi belanja provinsi 2022).
**Script:** `scripts/anggaran_compare.py` | **Log:** `logs/anggaran_compare.log`, `logs/anggaran_bj.log`

| Tahun | Belanja barang & jasa Dinkes | Penjualan (PT Parit Panjang) |
|---|---|---|
| 2022 | **Belanja Barang dan Jasa Rp36,39 M** (DPPA TA 2022 setelah perubahan; sebelum perubahan Rp44,37 M); belanja pegawai Rp181,42 M pagu / Rp171,73 M realisasi; belanja hibah Rp5,11 M; belanja modal Rp23,60 M pagu / Rp22,31 M realisasi | Rp6,96 M |
| 2023 | **Program Sediaan Farmasi, Alkes & Makanan Rp36,14 M alokasi / Rp28,12 M realisasi (77,81%)** — komponen paling relevan alkes; kegiatan pengadaan obat/vaksin/alkes Rp15,83 M realisasi; total Rp304,38 M / Rp253,26 M (83,20%) | Rp8,02 M |
| 2024 | **Belanja Barang dan Jasa Rp283,04 M** (eksplisit, 58% dari total Rp486,52 M; mencakup program sediaan farmasi/alkes: BHP medis, pemeliharaan alkes, kalibrasi rutin); pegawai Rp164,52 M; **belanja modal Rp16,19 M = pengadaan aset alkes** (faskes rujukan provinsi, UPTD/BLKP/RS pemprov); hibah Rp4,08 M; DAK BOK Rp11,59 M | Rp5,82 M |

**Breakdown program 2023 (alokasi setelah perubahan / realisasi):** Penunjang Urusan Pemerintahan (gaji, operasional, BJ kantor) Rp94,13 M / Rp82,52 M (87,66%); Pemenuhan Upaya Kesehatan (layanan masyarakat & rujukan, termasuk JKN Rp122,47 M) Rp158,46 M / Rp129,54 M (81,75%); Pencegahan & Pengendalian Penyakit Rp10,75 M / Rp8,87 M (82,56%); Peningkatan Kapasitas SDM Kesehatan Rp4,89 M / Rp4,20 M (85,84%); **Sediaan Farmasi, Alkes & Makanan Rp36,14 M / Rp28,12 M (77,81%)** — komponen paling relevan alkes, di dalamnya kegiatan pengadaan obat/vaksin/alkes Rp15,83 M realisasi. Mayoritas anggaran = transfer JKN + gaji, bukan pengadaan alkes langsung.

**Mekanisme pengadaan 2024 (konteks dari user):** Dinkes Provinsi membeli BMHP, obat program (TBC, malaria, HIV), dan alkes tertentu secara TERPUSAT di tingkat provinsi (dari BJ Rp283,04 M), lalu mendistribusikan fisik ke Gudang Farmasi Kabupaten/Kota. Dalam laporan keuangan provinsi, nilai ini dicatat sebagai satu kesatuan BJ operasional provinsi, bukan di-breakdown per kabupaten. Implikasi: (1) penjualan distributor ke kab/kota/RS tidak tercermin langsung di belanja provinsi — korelasi penjualan dengan anggaran provinsi lemah karena jalur pengadaannya berbeda; (2) permintaan riil tetap ditentukan fasilitas kesehatan di kab/kota (struktur RS), konsisten dengan 2.6.

**Temuan:** arah BERLAWANAN dengan penjualan — total belanja barang & jasa naik 7,78x (Rp36,39 M → Rp283,04 M, 2022→2024), penjualan turun 16% (Rp6,96 M → Rp5,82 M). CATATAN PENTING: Rp283,04 M = TOTAL BJ Dinkes (operasional, jasa, perjalanan dinas, dll), BUKAN khusus alkes. Komponen alkes 2024 yang pasti hanya belanja modal aset alkes Rp16,19 M; program sediaan farmasi/alkes (BHP, pemeliharaan, kalibrasi) ada di dalam BJ tapi tidak eksplisit. Pembanding: program sediaan farmasi/alkes/makanan 2023 = Rp36,14 M alokasi / Rp28,12 M realisasi. Dengan komponen alkes yang lebih akurat (2023: Rp28,12 M realisasi program alkes; 2024: Rp16,19 M modal alkes + bagian BJ), arah tetap tidak sejalan dengan penjualan (2023 Rp8,02 M → 2024 Rp5,82 M). Belanja barang & jasa 2024 (Rp283 M) = 49x penjualan 1 distributor (wajar: mencakup jasa, pemeliharaan, perjalanan dinas, bukan hanya alkes; Dinkes belanja dari banyak distributor). Data satudata (realisasi belanja provinsi 2022: barang & jasa Rp2,04 triliun) adalah seluruh APBD provinsi (semua OPD), terlalu makro untuk dikorelasikan dengan penjualan alkes.

**Keterbatasan:** belanja barang & jasa 2023 total tidak eksplisit (hanya per program; program sediaan farmasi/alkes/makanan Rp36,14 M adalah bagian darinya); komponen alkes 2024 tidak eksplisit (hanya belanja modal aset alkes Rp16,19 M yang pasti; program sediaan farmasi/alkes ada di dalam BJ Rp283,04 M tanpa rincian); data tingkat provinsi (1 nilai/tahun, n=3) → hanya deskriptif, tidak bisa korelasi statistik; penjualan = 1 distributor.

**Kesimpulan:** komponen belanja yang relevan (barang & jasa, pengadaan obat/alkes) juga tidak sejalan dengan penjualan — memperkuat posisi struktur fasilitas kesehatan (2.6) sebagai kandidat faktor X terkuat.

### 2.8 Konteks epidemiologi & bencana Sumsel 2020-2026 (data user)

**Sumber:** data konteks dari user (Dinkes Sumsel, BPBD Sumsel) — agregat/kualitatif, belum diuji korelasi statistik per region.

**Wabah & penyakit menular:**
- **COVID-19 (2020-2023):** pandemi, puncak 2020 & pertengahan 2021 (Delta), BOR naik di Palembang, endemi pertengahan 2023. Konsisten dengan 3.5 (uji COVID vs syringe tidak signifikan) dan temuan Airvo Agustus 2021 (event tunggal Delta).
- **DBD (2024-2025):** lonjakan signifikan; 2025 tercatat 4.130-6.265 kasus, tren naik September-November, Palembang tertinggi. Konsisten dengan BPS (2024: 70,4/100k, tertinggi sepanjang 2020-2025); uji DBD vs syringe tidak signifikan (3.5).
- **ISPA:** volume kasus tertinggi tiap tahun; puncak musim kemarau/karhutla; per Sep 2025 akumulasi 390.354 kasus; konsentrasi Palembang, Muara Enim, Banyuasin, Musi Banyuasin, Lahat.
- **TBC:** 2025 ditemukan 24.748 kasus aktif; pelacakan lanjut awal 2026.
- **Diare:** Jul 2026 Kota Palembang 14.248 kasus (5.570 balita).
- **HIV/AIDS:** Jan-Mei 2026: 380 kasus baru, 28 kematian.

**Bencana hidrometeorologi:**
- **Banjir (Jan-Mei tiap tahun):** banjir besar 2020-2024 di Muratara, PALI, Muara Enim, Banyuasin, Ogan Ilir; 2026: 90 kejadian (32 banjir, 30 angin kencang, 14 longsor); terbanyak Ogan Ilir (16), OKU Selatan (12), Muara Enim (9), Prabumulih (8).
- **Karhutla (Jun-Okt):** puncak 2023 (El Nino ekstrem, kabut asap Palembang); 2026: 7.050 hotspot Jan-Agu, 1.077 kejadian (120 Juni → 456 Juli); terparah OKI, Banyuasin, Musi Banyuasin.

**Status kedaruratan resmi & anggaran BTT (2020-2025):**
- Status siaga banjir/longsor Des-Apr (luapan Musi, Rawas, Ogan; Muratara, PALI, Muara Enim, Banyuasin; longsor Lahat, Pagar Alam, OKU Selatan); status siaga asap karhutla Mei/Juni-Nov (2023 & 2025 terpanjang; SK Gubernur No. 366/KPTS/BPBD-SS/2025, 17 Jun-30 Nov 2025; hotspot terpusat OKI & Musi Banyuasin).
- Anggaran BTT: puncak 2020-2021 (ratusan miliar: APD, insentif nakes, rumah isolasi, bansos COVID); normalisasi 2022-2024 (logistik bencana, beras, jembatan, satgas karhutla); 2025 realisasi efisien (pemadaman darat, bansos banjir; heli water bombing dari APBN BNPB).
- Insight: 2021 = puncak COVID + puncak BTT + puncak penjualan (Rp13,03 M) — satu-satunya wabah yang terlihat di data penjualan, lewat pengadaan darurat (Airvo, GeNose, APD); DBD 2024-2025 dan karhutla 2023 tidak berkorelasi dengan penjualan.

**Implikasi:**
- Volume penyakit tinggi & konsisten (ISPA 390 ribu, DBD, TBC 24 ribu, diare) = volume pasien tinggi = permintaan alkes stabil melalui fasilitas kesehatan — konsisten dengan 2.6 (pasien keluar r=+0,522).
- Wabah spesifik (COVID, DBD) tidak berkorelasi dengan penjualan syringe (3.5); hanya event tunggal COVID (Airvo, Delta 2021) yang terlihat di data.
- Bencana (banjir, karhutla) menambah kebutuhan logistik/alkes tapi tidak mengubah struktur permintaan.
- Konteks Muara Enim (pembeli syringe terbesar): ISPA, TBC, PD3I tinggi — pengadaan terkait program imunisasi & layanan rutin, bukan wabah.

### 2.9 Mekanisme pembayaran: lag 30 hari (domain knowledge, 27 Agustus 2026)

**Sumber:** informasi dari user — mekanisme pembayaran B2B di industri alkes Sumsel.

Dataset mencatat **tanggal faktur (order/pembelian)**, bukan tanggal pembayaran. Pembayaran oleh pelanggan menggunakan mekanisme **tempo 30 hari / tanggal yang sama bulan berikutnya** (contoh: faktur 3 Maret 2024, pembayaran 3 April 2024).

**Implikasi untuk interpretasi:**
- Model memprediksi **pola order**, bukan cash flow — lebih relevan untuk supply chain planning (kapan barang dipesan).
- Push order akhir tahun (Des) tercatat di data sebagai order Desember, meskipun payment masuk Januari. Fluktuasi akhir tahun yang kita amati mencerminkan perilaku order, bukan penundaan pembayaran.
- Hari kerja kosong mencerminkan **tidak adanya order baru** di hari tersebut, bukan penundaan pembayaran. Sumber hari kosong: jembatan long-weekend, PPKM, pull-forward tutup buku, atau fluktuasi B2B normal (3.6).

**Keterbatasan:** data agregat kualitatif tanpa rincian temporal per region yang bisa diuji korelasi; TIDAK dipakai sebagai fitur model.

---

## 3. Produk Bernilai Tinggi

### 3.1 Produk COVID mahal

**Script:** `scripts/covid_products_analysis.py` | **Hasil:** `results/covid_products_analysis.json`

| Produk | Nilai | Faktur | Waktu | Kontribusi |
|---|---|---|---|---|
| Airvo Unit Optiflow Therapy | 762,5 jt | 1 | Agu 2021 (gelombang Delta) | 3,2% total 2021 |
| Hyper-Hypothermia Blanketrol III | 441,5 jt | 1 | Nov 2023 | 2,3% total 2023 |
| Junior Optiflow Nasal Cannula | 10,0 jt | 4 | - | kecil |

**Temuan:** transaksi tunggal bernilai besar = outlier yang berkontribusi pada volatilitas harian dan residual. Airvo Agustus 2021 = bukti dampak pandemi (gelombang Delta).

### 3.2 Produk harga satuan > 50 jt

**Script:** `scripts/expensive_products_analysis.py` | **Hasil:** `results/expensive_products_analysis.json`

| Produk | Harga/unit | Pembeli | Waktu |
|---|---|---|---|
| Blanketrol III | 441,5 jt | RSUD Siti Fatimah (Palembang) | Nov 2023 |
| Infant Warmer | 420 jt | PT Bank Tabungan Negara (Jakarta Pusat)* | Agu 2021 |
| Bubble Infant nCPAP | 179,4 jt | Yay. Rafflesia (Bengkulu)* | Agu 2021 |
| Airvo Optiflow | 152,5 jt x 5 | Rumkit Bhayangkara (Palembang) | Agu 2021 |
| GeNose | 88,5 jt x 2 | Rumkit Bhayangkara (Palembang) | Jun 2021 |
| Servis Juli (jasa servis alat) | 77,8 jt | Rumkit Bhayangkara (Jambi) | Jul 2024 |

*Di luar Sumsel — tidak masuk panel 16 region.

Catatan koreksi grain: harga satuan kini dihitung per baris item (`jumlah`/qty). Patient Monitor maks 40,1 jt/unit dan Neopuff 42,2 jt/unit berada di bawah ambang 50 jt; angka lama "Patient Monitor 74,6 jt" adalah artefak dedup (nilai seluruh faktur dibagi qty baris pertama).

### 3.3 Produk neonatal

100 faktur, 940,5 jt IDR (~0,9% total penjualan).

- Dalam Sumsel: Neonatal Nasal Ventilation Double Tube 8Fr 98,3 jt (56 faktur), 10Fr 92,0 jt (55 faktur), Neopuff Infant Resuscitator 84,4 jt (Prabumulih)
- Pembeli: RS AR Bunda Prabumulih (118,3 jt, 17 faktur — RS ibu & anak), RSUD Rabain Muara Enim (105,9 jt, 11 faktur); pembeli terbesar di luar Sumsel: Infant Warmer Jakarta (420 jt), Bubble nCPAP Bengkulu (179,4 jt)
- Puncak 2021 (612,9 jt) didominasi pembeli luar Sumsel

**Keputusan data kelahiran:** produk neonatal hanya ~0,9% total dan pembeli terbesarnya di luar Sumsel — data kelahiran per kab/kota TIDAK wajib ditambahkan, dampaknya marginal.

### 3.4 Pembelian qty besar

293 baris item dengan qty >= 10.000 unit (177 faktur), total 7,38 M IDR.

| Produk | Qty | Nilai | Faktur |
|---|---|---|---|
| Disposable Syringe | 6,60 jt unit | 5,78 M IDR | 161 |
| Auto Destruct Syringe | 885 rb unit | 1,00 M IDR | 13 |

Di luar syringe: ECG Red Roll 135 rb unit, Masker N95 12,5 rb unit, Infusion Set, IV Catheter. Pembeli dominan: DINKES Kab. Muara Enim, RSUD Sekayu, RSUD Palembang Bari, RSUD Rabain, DINKES Musi Banyuasin. **Pola: pengadaan massal jarum suntik oleh Dinas Kesehatan (program layanan/imunisasi)** — qty besar, nilai besar, pembeli institusi pemerintah.

### 3.5 Distribusi temporal pengadaan syringe + uji wabah (DBD/COVID)

**Script:** `scripts/syringe_temporal.py` (hasil `results/syringe_temporal.json`), `scripts/syringe_wabah.py` (hasil `results/syringe_wabah.json`; rekonstruksi uji DBD/COVID dengan grain baris-item — skrip lama `syringe_dbd.py`/`syringe_covid.py` terhapus dan memakai qty hasil dedup yang basi)

**Distribusi temporal (semua syringe, 1.756 faktur, 12.372.163 unit, Rp12,34 M):**

| Tahun | Qty massal (>=10.000) | Puncak bulan |
|---|---|---|
| 2020 | 773.200 | Des (385.800) |
| 2021 | 1.908.768 (tertinggi) | Jun (399.800) |
| 2022 | 1.677.800 | Jul (629.000) |
| 2023 | 1.496.500 | Jul (435.800) |
| 2024 | 1.189.100 | Jul (238.000) |
| 2025 | 436.000 | Okt (126.700) |

Pengadaan massal (qty >= 10.000 per baris item): 168 faktur, 7.481.368 unit, Rp6,78 M. Pembeli dominan: DINKES Muara Enim (1,31 jt unit), RSUD Sekayu (1,18 jt), RSUD Palembang Bari (763 rb), RSUD dr. H.M. Rabain (720 rb). **BUKAN 1 event tunggal** — tersebar 2020-2025 dengan puncak konsisten di pertengahan tahun Jun-Jul (awal tahun fiskal) plus lonjakan Desember (akhir tahun anggaran) = pola pengadaan pemerintah. Top region pembeli syringe massal (Muara Enim, Musi Banyuasin, Palembang); Palembang #1 penjualan tapi #3 pembeli syringe massal.

**Uji DBD** (qty syringe vs kasus DBD per 100.000, region-tahun, n=77): pooled r=+0,048 p=0,677; within-region r=+0,072 p=0,536 — TIDAK ada hubungan. Tahun DBD tertinggi (2024: 70,4/100k menurut baris provinsi BPS) bukan tahun syringe tertinggi; 2021 syringe tertinggi justru DBD terendah (13,4/100k).

**Uji COVID** (qty syringe per bulan vs fase pandemi): pandemi (Jul 2020-Des 2022) mean 220.398/bulan vs endemi (2023+) 169.418/bulan — Mann-Whitney p=0,614 (tidak berbeda); Spearman qty vs pandemi r=+0,064 p=0,613. Puncak tertinggi justru Jul 2022 (709.779); Omicron (Jan-Feb 2022) syringe terendah (37.659 dan 57.854).

**Konteks Muara Enim (pembeli syringe terbesar):** kasus kesehatan 2023 didominasi ISPA, TB Paru, dan PD3I — 6 bayi pertusis dan 13 campak akibat penurunan cakupan imunisasi dasar. Pengadaan syringe DINKES terkait **program imunisasi rutin/kejar imunisasi**, bukan respons wabah DBD/COVID.

**Kesimpulan:** pengadaan massal syringe = kebutuhan rutin fasilitas kesehatan (imunisasi, pelayanan), didorong siklus anggaran, bukan wabah. Konsisten dengan 2.6: struktur fasilitas kesehatan adalah kandidat faktor X terkuat.

### 3.6 Hari kerja tanpa penjualan + koreksi kalender libur

**Script:** analisis ad-hoc + `scripts/holidays.py` | **Hasil:** `results/tanggal_kosong_kerja.csv`, `results/exp_libur.json`

Dari rentang data 1 Jul 2020 s.d. 27 Okt 2025 (1.945 hari): 1.509 hari ada transaksi; 436 hari kosong = 322 Sabtu/Minggu + 93 hari kerja libur nasional + **21 hari kerja murni tanpa penjualan**.

**Koreksi `holidays.py` (26 Agustus 2026):** daftar lama kehilangan 33 tanggal cuti bersama/libur nasional SKB 3 Menteri (Idulfitri CB 2021-2025, Imlek/Nyepi/Waisak/Iduladha/Natal CB, Pemilu 14 Feb 2024, Pilkada 27 Nov 2024) dan memuat 2 tanggal salah (Tahun Baru Islam 1443 H digeser 10 s.d. 11 Agustus 2021; Maulid Nabi digeser 19 s.d. 20 Oktober 2021). Semua dikoreksi sehingga fitur `is_hol` pada eksperimen berikutnya otomatis benar.

**Klasifikasi 21 hari (forensik D-3 s.d. D+3 untuk setiap tanggal):**
1. **Jembatan long-weekend / libur tidak resmi:** 11 Mei 2021 (awal mudik menjelang Idulfitri CB 6-17 Mei); 2-3 Des 2021 Kam-Jum (mulai libur panjang 4 hari setelah Hari Disabilitas Nasional 3 Des); 28 Des 2023 Kamis (jembatan antara Natal 25 Des dan Tahun Baru 1 Jan); 12 & 15 Jul 2024 Jum-Sen (push order 174 jt pada 11 Jul, lalu lubang 4 hari); 4 Jan 2021 Senin (hari kerja pertama tahun baru; awal data masih turbulen).
2. **Konteks pandemi:** 31 Des 2021 Jumat (PPKM Nataru - cuti nasional dilarang, perayaan malam tahun baru dilarang); 4, 7 & 10 Agustus 2020 Sel-Jum-Sen (era PSBB transisi Palembang - PSBB pertama disetujui Menkes 12 Mei 2020, berlaku 20 Mei, berakhir bertahap Juni-Juli; bulan-bulan awal data juga masih turbulen).
3. **Pull-forward akhir bulan:** 1 April 2022 Kamis - D-1 (31 Mar 2022) mencatat 139 jt (6x median, push tutup buku/target kuartal I), lalu Kamis-Senin mati (awal Ramadan 1443 H).
4. **Tanpa pola khusus (fluktuasi B2B normal):** 9 Feb 2022 Rabu; 6 Jan 2023 Jumat; 19 Jun 2023 Senin; 21 Des 2023 Kamis; 13 Jun 2024 Kamis; 14 & 20 Nov 2024 Kamis-Rabu; 17 Sep 2025 Rabu. Tidak ditemukan libur nasional, event kesehatan, atau kebijakan pembatasan di tanggal-tanggal ini; lebih mungkin merupakan variasi siklus operasional B2B normal.

**Uji dampak ke model (`exp_libur.py`, hasil `exp_libur.json`):**
- Eks-ante: flag jendela Idulfitri +/-7 hari + Natal-Tahun Baru membuat model **lebih buruk signifikan** dari baseline: R2 menurun dari 0,0670 s.d. 0,0661; DM = +4,107, p < 0,0001 (uji Diebold-Mariano dua arah, suku bebas Newey-West 7).
- Flag hardcoded 54 tanggal tidak valid untuk model: panel hanya memuat hari dengan transaksi (hasil pivot table); hari tanpa penjualan **tidak ada sebagai sampel** di data train/validation/test sehingga flag-nya bernilai nol di seluruh split. Hari kerja kosong adalah persoalan cakupan kalender operasional, bukan fitur prediksi dalam arsitektur LSTM-GNN saat ini.

### 3.7 Atribut node GCN: jumlah RS (27 Agustus 2026)

**Script:** `scripts/exp_rolling_nodefeat.py` (base config), `scripts/exp_nodeattr_tuned.py` (tuned config) | **Hasil:** `results/exp_rolling_nodefeat.json`, `results/exp_nodeattr_tuned.json`

**Pertanyaan:** apakah informasi infrastruktur kesehatan regional (jumlah RS per kabupaten/kota) memperbaiki prediksi model jika dimasukkan sebagai atribut node pada jalur GCN?

**Hasil — Hybrid base (128/128/do0, 3 seeds):**
- W2 (+jumlah RS node): R2 0,0609 vs W0 0,0592; DM = -2,861, p = 0,004 — signifikan.
- W3 (+jumlah RS + luas daerah): R2 0,0609 = W2 — luas daerah tidak menambah.

**Hasil — HybridTuned (128/64/do0.2, 3 seeds, paired init):**
- T1 (+jumlah RS node): R2 **0,0528** vs V2_ref 0,0508; DM = **-6,037, p < 0,0001** — signifikan kuat.

**Temuan:** jumlah RS secara konsisten dan signifikan memperbaiki model pada kedua konfigurasi. Mekanisme: Palembang memiliki 33 RS (44% total provinsi), kabupaten lain 1-8 RS. Jumlah RS berfungsi sebagai proxy kapasitas kesehatan regional yang mempengaruhi pola pengadaan alat kesehatan. Informasi ini masuk via jalur GCN sehingga embedding node mencerminkan struktur infrastruktur kesehatan, bukan hanya kedekatan geografis.

**Fitur yang tidak membantu:** rolling mean/std 7 & 30 hari (DM p = 0,763 — LSTM sudah menangkap smoothing secara internal); luas daerah (korelasi tidak signifikan r = -0,341, p = 0,196).

---

## 4. Temuan Data & Struktur

1. **Libur nasional resmi** SKB 3 Menteri 2020-2025 di-hardcode di `scripts/holidays.py` (2020: 25 hari incl. cuti bersama; 2021-2025 daftar resmi).
2. **Struktur folder dirapikan:** `data/` (xlsx + Data BPS + bps/ + geojson + csv), `scripts/` (13 aktif + archive/ 47 usang), `docs/`, `manuscript/`, `logs/`, `results/`, `figures/`. Folder `skrip/` dibuang. Runner log otomatis `scripts/run.ps1`.
3. **Path >260 karakter** di `data/Data BPS/` menyebabkan error Windows — dinormalisasi ke `data/bps/` nama pendek via `scripts/normalize_bps.py` (22 file).
4. **Bug data BPS dikoreksi:** kolom belanja 2021 = rupiah bukan ribu rupiah; jarak MRU pakai max bukan sum; mapping ganda nama kab/kota (Banyu Asin/Banyuasin, Kota Palembang/Palembang, Pali/Penukal Abab, OKU Selatan/Timur).
5. **File `data_apbd.xls` = XML Spreadsheet 2003** (bukan .xls biner), APBD Provinsi Sumsel agregat, realisasi s.d. Oktober 2025: Pendapatan 8,26 T (78,5%), Belanja Daerah 6,30 T (59,2%), Barang & Jasa 1,42 T (62,1%). Nilai identik dengan portal DJPK.
6. **Anti-leakage temporal:** file lama realisasi 2025 = 2,20 T (akhir tahun), file baru = 1,42 T (s.d. Oktober, saat test set berakhir) — file baru yang benar untuk konteks test 2025.
7. **Koreksi grain analisis produk (21 Agustus):** `syringe_temporal.py`, `expensive_products_analysis.py`, dan uji wabah lama memakai qty/nilai hasil dedup per faktur atau menjumlahkan `jual_total_fak` per baris item -> angka basi. Grain benar = baris item; nilai = kolom `jumlah`. Uji wabah direkonstruksi sebagai `syringe_wabah.py`. Semua angka syringe/neonatal/qty besar di dokumen ini sudah dikoreksi.
8. **Koreksi `holidays.py` + temuan panel gap (26 Agustus):** daftar libur sebelumnya kehilangan 33 cuti bersama/libur nasional SKB 3 Menteri dan memuat 2 tanggal salah (TB Islam 1443 H + Maulid Nabi 1443 H geser ±1 hari). Dikoreksi sehingga eksperimen berikutnya otomatis benar. Ditemukan 21 hari kerja murni tanpa penjualan (forensik di 3.6). Eksperimen `exp_libur.py` membuktikan hari-hari ini **tidak terjangkau arsitektur model saat ini**: panel hanya memuat hari transaksi; hari tanpa penjualan bukan sampel sehingga fitur apapun untuk hari-hari itu bernilai nol di semua split train/val/test.

---

## 6. Pemetaan Produk ke Pemicu Permintaan (nama+spesifikasi -> grup klinis)

**Script:** `scripts/product_driver_mapping.py` | **Hasil:** `results/product_driver_mapping.json`, `results/product_driver_monthly.csv`, `results/product_event_spikes.json`
**Metode:** (1) klasifikasi level baris (nama + spesifikasi; spuit dipilah: 1cc/Auto Destruct -> IMUNISASI, sisanya SPUIT_INJEKSI); (2) atribusi nilai: setiap faktur masuk grup dominan menurut bobot nilai item (`jumlah`), sehingga total antar grup == total nilai faktur unik **Rp99,97 M** tanpa double counting; (3) deteksi spike empiris pada nilai level-item: produk >= 3x median bulanan baseline dan >= 50 jt per jendela event. Event: Delta (Jul-Agu 2021), Omicron (Feb-Mar 2022), BIAN (Agu 2022), karhutla El Nino (Sep-Okt 2023), musim DBD (Jan-Apr 2021-2025).

**Koreksi grain (penting):** `jual_total_fak` berulang di tiap baris item satu faktur (rata-rata 3,43 item/faktur). Versi lama menjumlahkannya per baris -> inflasi ~5,4x; klaim "total 504,4 M" dan "OPERASIONAL_RS 43,5%" adalah artefak. Nilai pasar riil dataset = Rp99,97 M (faktur unik; kolom `jumlah` cocok 99,9%).

### 6.1 Statistik per grup pemicu (total nilai faktur unik, Rp99,97 M)

| Grup | Nilai | Produk | Faktur | Deskripsi |
|---|---|---|---|---|
| **APD_HIGIENE** | **18,3 M (18,3%)** | 97 | 1.383 | APD & higienitas rutin (handscoon, masker, penutup kepala) |
| BEDAH_OK | 14,7 M (14,7%) | 136 | 2.680 | operasi/kamar bedah (jahitan, sarung tangan steril, spinocan, mesh) |
| DBD | 12,4 M (12,4%) | 48 | 1.693 | terapi cairan IV / transfusi (cannula, infusion set) |
| SPUIT_INJEKSI | 12,3 M (12,3%) | 11 | 981 | spuit klinis non-imunisasi (3cc s.d. 50cc) |
| RESPIRATORIUM | 10,4 M (10,4%) | 71 | 1.991 | jalan napas & oksigenasi (ET tube, masker O2, suksi) |
| OPERASIONAL_RS | 7,2 M (7,2%) | 606 | 1.247 | residual generik (long tail: 80% nilainya di 74 SKU) |
| PANDEMI_COVID | 6,3 M (6,3%) | 39 | 484 | respirasi akut COVID (tes, APD darurat, high-flow) |
| DIAGNOSTIK | 4,3 M (4,3%) | 62 | 720 | ECG, USG, film rontgen, strip glukosa |
| PERAWATAN_LUKA | 3,7 M (3,7%) | 71 | 622 | kassa, plester, balutan, kapas |
| FARMASI | 2,6 M (2,6%) | 43 | 403 | antiseptik, xylocaine/cathejell, kapsul kosong |
| UROLOGI | 2,0 M (2,0%) | 29 | 706 | kateter, urine bag, hemodialisis |
| STERILISASI_CSSD | 1,6 M (1,6%) | 42 | 334 | pouches, indikator, enzimatik |
| CAPITAL_EQUIP | 1,3 M (1,3%) | 40 | 48 | alat modal (monitor, blanketrol, meja periksa) |
| IMUNISASI | 1,0 M (1,0%) | 11 | 126 | Auto Destruct Syringe, safety box, spuit 1cc |
| OBGYN | 0,8 M (0,8%) | 17 | 99 | partus, sectio, speculum |
| NEONATAL | 0,5 M (0,5%) | 18 | 79 | neopuff, infant warmer, nCPAP |
| LABORATORIUM | 0,5 M (0,5%) | 40 | 62 | tabung, pipet, reagen wadah |

### 6.2 Bukti temporal (spike empiris per event, nilai level-item)

**Delta COVID (Jul-Agu 2021) — sinyal kuat, didominasi APD:** Handscoon 4 varian (x37/x15/x9/x8, total >Rp700 jt), Masker x33,8 (253 jt), Rapit Tes x36,6 (128 jt), Tabung Oxygen x12,7, Breathing Circuit x11,3, Oxygen Mask Non-Rebreathing x8,7, MF Cath Pro x8,3 (232 jt), Coverall x6,3. Pola: APD + tes + oksigen melonjak serentak.

**Auto Destruct Syringe tidak masuk top-20 spike Delta** — bukan karena tidak naik (rasio x5,1 pada agregat seluruh varian ADS), tetapi karena pembelian AD berulang setiap pertengahan tahun 2020-2024 (Jul 2020: 153 jt; Jul-Agu 2021: 250 jt; Jun-Jul 2022: 307 jt; Jul 2024: 59 jt; total Rp1,03 M): pola siklus fiskal pengadaan imunisasi, bukan event tunggal. Konsisten dengan ITS: intervensi vaksinasi signifikan hanya pada AD syringe (D2 +37.384 unit/bulan, p=0,0045).

**Omicron (Feb-Mar 2022) — LEMAH:** hanya 3 hit kecil (Assucryl jahitan x6,3/x3,6, Mask Nebulizer x3,1). Gelombang ringan, testing menurun.

**BIAN imunisasi (Agu 2022):** Handscoon Non Steril x17,4 (396 jt) — sarung tangan non-steril melonjak saat kampanye imunisasi massal nasional. Antiseptik (Chlorhexidine/Povidine/Alkohol) TIDAK lolos ambang pada nilai riil (<50 jt/bulan; klaim lama "Chlorhexidine x54,8" artefak double counting). Sinyal BIAN = consumable bervolume besar harga murah, bukan antiseptik.

**Karhutla El Nino 2023 (Des 2022 - Okt 2023, puncak Agst-Spt 2023) — TEMUAN NEGATIF (multi-lapis):**
- **Konteks temporal:** Api dan lahan terbakar mulai muncul di beberapa provinsi (Lampung, Riau, Kalimantan, Kepri) sejak Desember 2022; karhutla kecil-menengah marak Feb 2023; El Nino menguat signifikan Agst-Spt 2023 (BNPB hotspot; The Conversation 2023).
- **Penjualan aggregate TIDAK naik:** Agst 2023 = 1.189 M IDR (215 faktur); Spt 2023 = 772 M IDR (188 faktur). Total Agst+Spt 2023 (1.961 M) justru LEBIH RENDAH dari Agst+Spt 2022 (2.016 M) dan jauh di bawah puncak COVID 2021 (4.288 M).
- **Produk respiratory TIDAK spike:** Spt 2023 respiratory = 90,8 M IDR (27 faktur) = terendah sejak Maret 2023. Faktur besar (> 50 M) di Agst-Spt 2023 hanya 1 (Palembang, JVA Tech Medical, 141,7 M). Spt 2023 = nol faktur > 50 M.
- **Pola aggregate per tahun (Agst+Spt):** 2020 1.109 M; 2021 4.288 M (COVID); 2022 2.016 M; 2023 1.961 M (El Nino); 2024 1.902 M; 2025 1.532 M. Tren menurun, tanpa spike karhutla.
- **Produk level-item:** masker TIDAK melonjak. Yang naik justru IV Catheter x21,9, Infusion Set x6,5, IV Cannula x5,9 (pola belanja umum Sep-Okt, bukan respons asap). Karhutla BUKAN pemicu signifikan penjualan di data ini.
  - **Verifikasi satuan (28 Agu 2026, `scripts/masker_karhutla_analysis.py`):** masker dijual campur satuan Box dan Pcs, dan isi box bervariasi per tipe (surgical/duckbill = 50 pcs/box, N95/KN95 = 10 pcs/box; dari user). Setelah qty dikonversi ke pcs (qty_box x isi_box + qty_pcs), temuan negatif justru makin kuat: saat karhutla (Agu-Sep 2023) qty total = 21.050 pcs vs baseline 44.907 pcs/bln (ratio 0,47); surgical ratio 0,48; N95/KN95 = 0 pcs vs 679/bln. Nilai Box = 8,0 jt vs 64,4 jt/bln (ratio 0,12). Puncak masker sebenarnya di masa COVID 2020-2021 (Sep 2020 = 290.000 pcs; Sep 2021 = 362.740 pcs), bukan karhutla.
- **Puncak sebenarnya mengikuti siklus fiskal:** Nov 2023 = 1.825 M (belanja akhir tahun); Jun 2023 = 1.400 M (akhir FY). Bukan saat karhutla.

**Musim DBD (Jan-Apr) — TERKONTAMINASI:** spike besar di semua famili (Oxygen Mask x75, Feeding Tube x75, Handscoon x69, Assucryl x43) karena Jan-Apr = musim belanja anggaran awal tahun, bukan spesifik DBD. DBD tidak terpisahkan dari seasonality umum.

### 6.3 Kesimpulan

1. **Residual operasional tinggal 7,2%** (606 SKU long tail; 80% nilai di 74 SKU). Klaim lama "43,5% operasional rutin RS" artefak double counting — dikoreksi.
2. **Struktur pasar:** mayoritas belanja = consumable klinis rutin bernilai kecil per item (APD 18,3% + bedah 14,7% + IV/DBD 12,4% + spuit 12,3% + respiratorium 10,4%). Volatilitas permintaan harian inilah tantangan utama model, bukan event besar.
3. **Event-driven kecil:** COVID 6,3%; imunisasi-murni 1,0%. Sinyal vaksinasi lebih kuat di UNIT AD syringe (ITS D2 signifikan) daripada rupiah — bundling ADS dengan vial oleh Kemenkes membuat pembelian lokal tak proporsional terhadap volume vaksinasi di beberapa kab/kota.
4. **Fitur kanal untuk model (V5):** share komposisi (V5b) satu-satunya turunan pemetaan ini yang signifikan memperbaiki model (DM vs V3_ref = -5,27, p<0,0001; R2 0,0687 vs 0,0670). Nilai kanal mentah (V5a) tidak signifikan (p=0,06).
5. **Outlier mahal (CAPITAL_EQUIP) = 1,3%, 48 faktur** — kontribusi kecil tapi spike per transaksi besar; mendukung winsorization target stage amount.
6. **Karhutla: negatif. DBD: tidak terpisah dari seasonality umum.**

---

## 7. Diagnostik Zero-Inflated Classifier Collapse (27 Agustus 2026)

**Script:** `scripts/diag_zi.py`, `scripts/exp_zi_fix_v3.py`, `scripts/exp_zi_fix_final.py`
**Hasil:** `results/exp_zi_fix_final.json`, `results/exp_zi_fix_v3.json`

### 7.1 Temuan: Classifier collapse ke Palembang

Dengan 84,85% target test bernilai nol, binary cross-entropy loss didominasi oleh kelas mayoritas. Classifier (p_clf) **tidak pernah** output p > 0,5 untuk selain Palembang di seluruh 231 hari test:

| Region | nz_true | nz_pred>0.5 | mean p_clf | max p_clf |
|---|---|---|---|---|
| Palembang | 223 | 231 | 0,969 | 0,991 |
| OKU Timur | 72 | 0 | 0,264 | 0,381 |
| OKU | 66 | 0 | 0,245 | 0,403 |
| Lahat | 57 | 0 | 0,231 | 0,461 |
| Semua lainnya | 0-28 | 0 | <0,26 | <0,32 |

**Threshold sweep 0,1-0,7 menghasilkan R2 identik = 0,156** karena regression head (p_amt) juga output near-zero untuk non-Palembang. R2 pooled 0,0523 didorong oleh prediksi zero yang benar (3.136 dari 3.696 region-day) + magnitude prediksi Palembang.

**Per-region R2 negatif untuk SEMUA region** termasuk Palembang (R2 = -0,3336). Model RMSE 8,16M **lebih buruk** dari baseline train_mean RMSE 7,66M. Model under-predict total penjualan test sebesar 78%.

### 7.2 Eksperimen perbaikan

| Config | Deskripsi | R2 (3-seed) | RMSE | vs Baseline |
|---|---|---|---|---|
| **A baseline** | pw=1, masked reg, thr=0.5 | 0,0367 | 8.413.218 | -- |
| **C1 wt+fullreg** | pw=4.1, full-data reg, thr=0.5 | 0,0494 | 8.081.407 | **+0,0127 (+35%)** |
| **C3 base+fullreg** | pw=1.0, full-data reg, thr=0.5 | 0,0494 | 8.081.412 | **+0,0127 (+35%)** |
| B1 weighted | pw=4.1, masked reg, thr=0.5 | -0,7055 | 9.009.053 | -78x worse |
| D masked-only | regression only, no ZI | -60,28 | 11.421.107 | catastrophic |
| D2 full-only | fullreg only, no ZI gate | -0,0753 | 8.073.251 | worse |

**Temuan kunci:**

1. **C1 = C3 secara identik** — classifier weight (pw=1 vs 4.1) tidak mempengaruhi hasil. Yang penting adalah **training regression pada SEMUA hari** (bukan masked ke non-zero).
2. **Fullreg meningkatkan R2 dari 0,0367 ke 0,0494** dengan membiarkan regression head mempelajari distribusi data secara keseluruhan, bukan hanya non-zero mode.
3. **Weighted BCE (pw=4.1) memperbaiki collapse classifier** (sekarang prediksi non-zero untuk Lahat 160 hari, OKU 206, OKU Timur 227) tapi over-predict → false positive → R2 lebih buruk (-0,71).
4. **Threshold tuning pada validasi** tidak membantu: threshold optimal = 0.80 (membatalkan efek weighted BCE) atau 0.15 (membuka semua prediksi, identik dengan fullreg).
5. **Regression tanpa ZI gate** (D) catastrophe: R2 = -60. ZI gating secara struktural diperlukan.

### 7.3 Implikasi

- **Nilai praktis model ada di classification stage** (kapan ada order), bukan amount stage.
- **R2 pooled didorong zero classification**, bukan magnitude forecasting per region.
- **Fullreg layak dipertimbangkan** sebagai perbaikan di manuscript (R2 0,0367 -> 0,0494).
- **Alternatif ZI untuk future work:** soft gating, mixture density network, atau threshold-free formulation.

---

## 5. Angka Model yang Tidak Berubah (Referensi)

- Hybrid base: R2 0,0523, RMSE 8.002.603 (16 region inti)
- Distribution star ablation: R2 0,0703
- Random search konfigurasi: R2 0,0582
- Ljung-Box residual: Q=21,510, p=0,0178
- Environment: venv `E:\pyvenv_geo`, RAM 31,6 GB