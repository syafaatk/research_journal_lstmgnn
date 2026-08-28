# Daftar TBD (To Be Determined) - Status per 19 Agustus 2026

Daftar ini memetakan semua `[TBD]` di `revisi.md` dan `[ISI: ...]` di `balasan_reviewer.md`. Kolom Status: `TERISI` (sudah diisi dari eksperimen ulang dengan `view_penjualan_detail data hingga oktober.xlsx`), `PERLU DATA` (butuh data/konfirmasi penulis), `PERLU EKSPERIMEN` (butuh eksperimen yang belum dijalankan).

## A. Data dan Graf

| # | Lokasi | Isi TBD | Status |
|---|---|---|---|
| A1 | revisi.md 3.4 (Coordinates) | Sumber koordinat: centroid wilayah, lokasi pelanggan, atau kantor pemerintah | TERISI - geocoding aktual 118/156 pelanggan (98,8% faktur = 10.752/10.878) via SIRS Kemenkes (21), Photon komoot (90), idalamat (3), lewatmana (2), Google Maps (1), Medicastore (1); 38 sisanya (126 faktur, 1,2%) = apotek/individu/dinas kecil tanpa alamat publik. Jarak koordinat aktual vs data asli: median 3,2 km, p90 25,6 km, max 125,7 km -> koordinat data asli = level kota. Region salah di data: Empat Lawang (~104,17 vs aktual 102,93) dan RSUD Sungai Lilin terlabel Palembang (aktual Musi Banyuasin, dikoreksi); RSUP Rivai Abdullah terlabel Banyuasin = BENAR (lokasi Mariana, Banyuasin, terverifikasi lewatmana -2,97966/104,86339). **MERGE (19 Agu 2026):** Musi Rawas Utara digabung ke Musi Rawas (data MRU hanya 2 faktur, fraksi nol 99,9%; dimekarkan dari Musi Rawas 2013) -> 16 region. File: data/customers_geocoded_final.csv, data/not_found_list.csv |
| A2 | revisi.md 3.4 (Edges) | Justifikasi derivasi angka 95,82 km dari distribusi jarak antarwilayah | TERISI - jarak 3-NN aktual (16 region, geocoded): min 22,61, median 46,68, max 104,66 km; threshold 104,66 km tidak memotong kandidat edge (0 dari 48) |
| A3 | revisi.md 3.4 (Graph properties) | Jumlah edge akhir untuk 17 node, derajat min/maks/rata-rata, konfirmasi tidak ada node terisolasi | TERISI - 29 edge undirected (16 node), derajat 3-6 (rata 3,62), tanpa node terisolasi |
| A4 | revisi.md 4.1 | Proporsi hari dengan penjualan nol, statistik deskriptif per wilayah | TERISI - fraksi nol 3,1% (Palembang) s.d. 99,9% (Musi Rawas Utara, digabung ke Musi Rawas); test 84,85% nol (16 region) |
| A5 | revisi.md 3.8 | MAPE dan jumlah titik aktual nol yang dikecualikan | TERISI - MAPE tidak dilaporkan (84,85% target test nol; persentase error tidak stabil); keputusan dicatat di naskah |
| A6 | revisi.md 3.9 | Ketersediaan data jaringan distribusi untuk ablation variant 6 | TERISI (19 Agu 2026) - penulis konfirmasi: **semua rute distribusi berpusat di Palembang** (pusat: Komplek Villa Kenten Blok G No. 2, Palembang). Varian 6 diimplementasikan sebagai star graph: Palembang terhubung ke semua 15 region lain. Hasil ada di eksperimen geocoded (ablation "Distribution network (star from Palembang)"). |

## B. Hasil Eksperimen (diisi dari eksperimen ulang dengan file Oktober)

> **Catatan (19 Agu 2026):** eksperimen geocoded di-restart pukul 19:16 karena perbaikan anti-leakage — normalisasi log1p (mu/sd) kini dihitung HANYA pada window training (sebelumnya seluruh panel termasuk test; klaim di `balasan_reviewer.md` poin 3 sudah sesuai kode). Log: `logs/zi_geocoded2.log`. **Run 16 region SELESAI 21:32**; angka B1/B3/B4/B5/B6/B7 di bawah sudah diperbarui ke hasil final.

| # | Lokasi | Isi TBD | Status |
|---|---|---|---|
| B1 | revisi.md 4.2 (Tabel 3) | Metrik skala asli (IDR) untuk 4 model + metrik klasifikasi | TERISI (16 region) - LSTM R2 0,0433; GNN 0,0059; Hybrid 0,0523; HybridTuned 0,0462; **Random search 0,0582 (terbaik)**; ACC 90,67%; AUC 0,861 |
| B2 | revisi.md 4.2 (Tabel 4) | Baseline: always-zero, naive, training-mean, ARIMA, XGBoost | TERISI (16 region) - always-zero R2 0,0059; naive -0,7978; mean -7,3242; ARIMA -0,0775; XGBoost -0,0840 |
| B3 | revisi.md 4.3 (Tabel 5) | Mean +/- std over seeds, Diebold-Mariano p-value, bootstrap 95% CI | TERISI (16 region) - DM vs Hybrid: LSTM 7,714, GNN 9,586, HybridTuned 7,818 (semua p < 0,001); CI Hybrid [6.857.854, 9.192.672] |
| B4 | revisi.md 4.4 (Tabel 6) | Hasil ablation: identity, random, distance, correlation, tanpa COVID, distribution star | TERISI (16 region) - Identity 0,0514; Random (5) 0,0476; Distance 0,0458; Correlation 0,0514; **Distribution star 0,0703 (terbaik, +34% relatif atas distance)**; Tanpa COVID 0,0470 |
| B5 | revisi.md 4.5 (Tabel 7) | RMSE, MAE, R2 per wilayah untuk 16 kab/kota | TERISI (16 region) - 2 region R2=1,0 (Empat Lawang, OKI); Palembang terburuk R2 -0,3336, RMSE 23.964.716 |
| B6 | revisi.md 4.5 | Analisis best/worst region, peta error, actual vs predicted, hubungan error dengan volume | TERISI (19 Agu 2026) - figure dibuat: `figures/fig06_error_map.jpg` (peta R2 per region), `fig05_actual_vs_predicted.jpg` (scatter pooled + time series region terpilih + total harian), `fig07_error_vs_volume.jpg` (RMSE vs volume test); analisis best/worst di naskah. **Regenerasi dengan hasil geocoded SELESAI (21:40).** |
| B7 | revisi.md 4.6 | Hasil residual: histogram, residual over time, per region, Ljung-Box | TERISI (19 Agu 2026) - figure dibuat: `figures/fig08_residuals.jpg` (histogram + residual harian over time + boxplot per region); mean 1.665.257 IDR, Ljung-Box Q=21,510, p=0,0178 (autokorelasi signifikan, diakui di Section 5). **Regenerasi dengan hasil geocoded SELESAI (21:40).** |

## C. Konfigurasi dan Lingkungan

| # | Lokasi | Isi TBD | Status |
|---|---|---|---|
| C1 | revisi.md 3.5 (Training config) | Versi library dan spesifikasi hardware | TERISI - Python 3.12.7, PyTorch 2.13.0+cpu, pandas 2.2.2, numpy 1.26.4, scikit-learn 1.5.1, scipy 1.13.1; CPU (tanpa GPU) |

## D. Referensi (butuh verifikasi metadata)

| # | Lokasi | Isi TBD | Status |
|---|---|---|---|
| D1 | revisi.md REFERENCES (Beldek) | Halaman bab dalam buku Springer | TERISI - pp. 570-579, *Industrial Engineering in the Industry 4.0 Era*, Lecture Notes in Mechanical Engineering, Springer, Cham; DOI 10.1007/978-3-030-31343-2_50 (verifikasi via Springer, First Online 25 Okt 2019) |
| D2 | revisi.md REFERENCES (Liu & Liu) | Volume, halaman, dan DOI International Core Journal of Engineering | TERISI - Vol. 10 No. 12 (2024), pp. 65-70, ISSN 2414-1895, DOI 10.6919/ICJE.202412_10(12).0008 (verifikasi via PDF asli bcpublication.org, 6 halaman mulai hal. 65) |
| D3 | revisi.md REFERENCES (Tunnicliffe Wilson) | Halaman review di Journal of Time Series Analysis 37(6) | TERISI - **koreksi: 37(5)**, pp. 709-711, Sept 2016; DOI 10.1111/jtsa.12194 (verifikasi via Wiley) |
| D4 | revisi.md REFERENCES (Wen et al.) | Halaman di Proceedings ICBBT 2024 | TERISI - pp. 108-114, *Proceedings of the 2024 16th International Conference on Bioinformatics and Biomedical Technology (ICBBT 2024)*, ACM; DOI 10.1145/3674658.3674677 (verifikasi via ACM DL) |

## E. Catatan Penting: Perubahan Protokol dan Perbedaan Data Aktual vs Klaim Tesis

### E1. Perubahan protokol (keputusan bersama)

Berdasarkan evaluasi granularitas, protokol diubah dari bulanan ke **harian zero-inflated**:

| Resolusi | R2 model terbaik | Catatan |
|---|---|---|
| Bulanan | -0,394 | 30 sampel training saja |
| Mingguan | -0,343 | 170 sampel training |
| Harian (regresi langsung) | -0,110 | 957 sampel training |
| **Harian zero-inflated** | **+0,043 s.d. +0,052** | klasifikasi 90,67% akurat, AUC 0,861 |

Fitur **COVID-19** ditambahkan sebagai kanal masukan kedua (indikator biner Juli 2020 - Des 2022, aktif 706 hari / 47,2% panel). Ablasi (16 region): dengan COVID R2 0,0523 vs tanpa 0,0470.

### E2. Perbedaan data aktual vs klaim tesis

| Aspek | Tesis | File Oktober (aktual) |
|---|---|---|
| Jumlah faktur unik | 46.822 entri | 13.661 faktur unik; 10.878 setelah pemetaan ke 16 kab/kota Sumsel |
| Rentang tanggal | Juli 2020 - Oktober 2025 | Juli 2020 - Oktober 2025 (1.495 hari) |
| Kota unik | 18 (termasuk luar Sumsel) | 34 (termasuk luar Sumsel dan nama kecamatan) |
| Kolom target | jual_total_fak | jual_total (nilai faktur) |

Implikasi:
1. Panel harian = 1.495 hari (Juli 2020 - Oktober 2025), 16 region.
2. Split 3 arah: training Juli 2020 - Des 2023 (957 sampel), validasi Jan - Des 2024 (277 sampel), testing Jan - Okt 2025 (231 sampel).
3. 34 kota dipetakan ke 16 kab/kota Sumsel (lihat `scripts/archive/experiment_zi_final.py` untuk tabel pemetaan). Kota di luar Sumsel (Bengkulu, Jambi, Metro, Jakarta Pusat, Bangka Belitung, dll.) dikeluarkan.
4. Semua angka hasil di Tabel 3-7 adalah hasil eksperimen ulang dengan data aktual dan TIDAK sama dengan angka tesis (0,0881/0,0324/8,219/0,8595). Ini wajar dan justru lebih kredibel karena dapat direplikasi.
5. Abstrak dan kesimpulan di revisi.md sudah diperbarui dengan angka aktual hasil eksperimen ulang.
6. Skrip eksperimen tersimpan di `E:\Download\Jurnal\skrip\`: `experiment_zi_final.py` (eksperimen utama), `continue_zi.py` (penyelesaian ablasi), `dm_hybrid.py` (DM test vs Hybrid), `analyze_zi.py` (rangkuman), `experiment_results_zi.json` (hasil lengkap).

### E3. Analisis parameter d_jual_qty dan d_barang_brg_id (19 Agustus 2026)

Pertanyaan: apakah parameter kuantitas dan produk bisa dipakai. Hasil eksplorasi dan uji empiris (`scripts/archive/explore_qty.py`, `scripts/archive/explore_kategori.py`, `scripts/archive/test_qty_signal.py`):

| Aspek | Temuan |
|---|---|
| d_jual_qty sebagai target | Tidak layak: katalog heterogen, harga satuan implisit median Rp207.037 - maksimum Rp500.500.000; korelasi qty vs nilai 0,29 (level item) / 0,46 (level faktur) |
| d_barang_brg_id | 2.911 produk, 35,9% muncul sekali; terlalu jarang untuk forecasting per produk |
| kategori_nama | Ternyata nama merek (375 unik setelah pemetaan), bukan kategori produk; 516 nilai mentah dengan banyak duplikat/kotor |
| Fitur qty lagged sebagai kanal input | Uji regresi linier (test_qty_signal.py): incremental R2 +0,014 (test) / +0,004 (semua data) di atas riwayat 30 hari - dapat diabaikan |
| Fitur brand/supplier/sales lagged | Uji serupa (test_attr_signal.py): incremental R2 +0,011 (test) / +0,006 (semua data) - dapat diabaikan; ketiganya berkorelasi identik (~0,41) karena proksi sinyal aktivitas yang sama |
| Atribut level item/faktur lain (barang_nama, kategori_nama, suplier_id, sales_id, diskon, pajak, stok) | Tidak bisa masuk model langsung (target agregat wilayah-hari); sales_nama adalah data pribadi (21 salesperson) dan tidak dilaporkan |
| Penggunaan yang dipilih | Analisis deskriptif di revisi.md 3.1 (item per faktur, proporsi faktur 1 item, top merek) + justifikasi empiris pemilihan target nilai dan eksklusi 32 atribut lainnya |

Keputusan: target tetap jual_total (nilai faktur); qty/produk dipakai untuk analisis deskriptif dan justifikasi, bukan sebagai target atau kanal input tambahan.

### E4. Analisis: kenapa HybridTuned lebih jelek daripada Hybrid base (19 Agustus 2026)

Hasil eksperimen geocoded 16 region (anti-leakage):

| Model | Konfigurasi | R2 (mean seeds) | RMSE | MAE | std R2 |
|---|---|---|---|---|---|
| Hybrid base | LSTM 128, GNN 128, dropout 0 | 0,0523 | 8.002.603 | 1.881.358 | 0,0040 |
| HybridTuned | LSTM 128, GNN 64, dropout 0,2 | 0,0462 | 8.165.301 | 1.924.589 | 0,0061 |

DM test (seed 42): Hybrid vs HybridTuned DM = -8,315, p < 0,001 (Hybrid lebih baik, signifikan). Bootstrap CI RMSE: Hybrid [6.724.730, 9.051.348], HybridTuned [7.325.181, 9.659.654].

Penyebab (4 faktor, urut dampak):
1. **Dropout 0,2 pada model yang tidak overfit.** R2 ~0,05 menunjukkan model underfit (training loss tidak jauh di bawah val loss). Dropout dirancang untuk mengatasi overfitting; diterapkan pada model yang sudah underfit, ia hanya membuang kapasitas efektif dan memperburuk prediksi. Ini konsisten dengan literatur: dropout membantu hanya saat training loss << validation loss.
2. **Kapasitas GNN dikurangi (64 vs 128).** Modul spasial kehilangan setengah kapasitas. Meski GNN mandiri hampir tidak berguna (R2 0,0059), lewat fusion concat [h_LSTM; h_GCN] dimensi turun dari 256 ke 192, mengurangi kapasitas lapisan fully connected.
3. **Overlap antar seeds.** HybridTuned seed 123 (0,0540) melebihi mean Hybrid (0,0523); std HybridTuned (0,0061) sebanding dengan selisih mean (0,0061). Perbedaan bukan "selalu jelek" melainkan "rata-rata lebih jelek", didukung DM test signifikan.
4. **Konfigurasi tidak transfer dari setting asli.** (128/64, dropout 0,2) dioptimalkan untuk panel bulanan tesis asli (hanya 30 sampel training, sangat rawan overfit). Pada 30 sampel dropout 0,2 krusial; pada 957 sampel harian kebutuhan dropout hilang dan kapasitas ekstra lebih bernilai.

### E5. Random search sesuai Bergstra & Bengio (2012) - rencana & status

Referensi yang membahas hyperparameter tuning di daftar pustaka: hanya **Bergstra & Bengio (2012)** (JMLR 13:281-305). Referensi lain (Kingma & Ba 2015 = optimizer Adam; Hochreiter & Schmidhuber 1997 = LSTM; Velickovic 2018 = GAT) membahas arsitektur/optimizer, bukan tuning.

Prinsip Bergstra & Bengio yang diterapkan (`scripts/random_search_bb.py`):
- Random search (bukan grid): N trial mengeksplorasi N nilai berbeda per dimensi; unggul karena low effective dimensionality.
- Distribusi sampling: hidden_lstm & hidden_gnn log-uniform [32, 256]; lr log-uniform [1e-4, 1e-2]; dropout uniform [0, 0,5]; batch uniform {32, 64, 128}.
- N = 30 trial: P(menemukan konfigurasi dalam top 10%) = 1 - 0,9^30 = 0,958.
- Seleksi: validation loss (BCE + masked MSE); test dipakai SEKALI di akhir.
- Early stopping (patience 8) untuk menghemat komputasi.

Status: **SELESAI (20 Agu 2026, ~1 jam)**. 30 trial dijalankan (log `logs/random_search_bb.log`, hasil `results/random_search_bb.json`). Konfigurasi terbaik (val loss & test R2 konsisten): trial 30, hl=219 hg=107 do=0,17 lr=1,5e-3 bs=32, val_loss=1,7314, test R2=0,0607 (seed 42). 12/30 trial mengungguli Hybrid base (0,0523); median R2 0,0481. Evaluasi ulang multi-seed (42/7/123, `scripts/eval_rs_best.py`, hasil `results/rs_best_eval.json`): mean R2 0,0582 +/- 0,0021, RMSE 7.843.703 +/- 57.252, MAE 1.846.331 +/- 12.116; ensemble R2 0,0583, RMSE 7.841.738, ACC 0,9067, AUC 0,8613; DM vs Hybrid base -6,307 (p < 0,001); CI RMSE [6.704.882, 9.028.509]. Hasil masuk ke revisi.md 3.6, Tabel 3/5, abstrak, kesimpulan, dan balasan_reviewer.md poin 10.

### E6. Eksperimen fitur kalender (20 Agustus 2026)

Lanjutan dari analisis residual (weekend kuat, libur indikasi, fase pandemi). Fitur hari target ditambahkan sebagai kanal input konstan sepanjang window 30 hari, di-tile ke semua region: (i) weekend biner, (ii) libur nasional biner (daftar resmi SKB 3 Menteri 2020-2025, `scripts/holidays.py`), (iii) fase pandemi 2-level (refocusing Jul-Des 2020, gelombang 2021-2022, endemi 2023+ baseline). Varian: V1 (weekend + holiday + fase, 6 kanal), V2 (day-of-week one-hot + holiday + fase, 12 kanal), V3 (config random search + fitur V2). Semua 3 seeds (42/7/123), config base 128/128, pipeline identik anti-leakage.

Status: **SELESAI (20 Agu 2026)**. Hasil (`scripts/exp_features.py`, `scripts/v3_rs_features.py`, hasil `results/exp_features.json` + `results/exp_features_v3.json`):
- V1: ensemble R2 0,0528, RMSE 7.990.984, DM -1,740 (p=0,082, tidak signifikan), Palembang RMSE 23.923.979.
- V2: ensemble R2 0,0612, RMSE 7.762.318, DM -6,655 (p<0,001, signifikan), Palembang RMSE 22.689.018 (-5,3%).
- V3: ensemble R2 0,0666, RMSE 7.609.122, MAE 1.774.426, DM -6,307 (p<0,001), CI [6.506.764, 8.795.473], Palembang RMSE 21.842.944 (-8,9%).
- Interpretasi: day-of-week one-hot jauh lebih informatif daripada weekend biner, konsisten dengan residual per hari (model overpredict hari kerja, Senin +22,1 jt). Masuk ke revisi.md 3.6, 4.7 (Tabel 8), 5, dan kesimpulan.

### E7. Analisis faktor eksternal Data BPS (20 Agustus 2026)

Data BPS Sumsel 2020-2025 (`data/bps/`, dinormalisasi dari `data/Data BPS/`): penduduk, kasus penyakit (DBD per 100.000), RS/puskesmas/posyandu, sarana kesehatan desa, belanja pemerintah kab/kota (2020-2023), realisasi belanja provinsi (2020-2025), jarak ibukota dari Palembang (2025), luas daerah (2025). Korelasi Spearman pooled per region-tahun terhadap penjualan (`scripts/bps_analysis.py`, hasil `results/bps_analysis.json`):

| Faktor | r | p | Kesimpulan |
|---|---|---|---|
| Belanja pemerintah kab/kota (2020-2023) | 0,401 | 0,001 (n=64) | signifikan pooled - TAPI efek skala (lihat verifikasi di bawah) |
| Penduduk | 0,164 | 0,111 | tidak signifikan |
| DBD per 100.000 | -0,051 | 0,619 | tidak signifikan |
| Jarak dari Palembang | -0,099 | 0,716 | tidak signifikan |
| Luas daerah | -0,341 | 0,196 | tidak signifikan |
| Realisasi belanja barang&jasa provinsi | -0,143 | 0,787 (n=6) | tidak signifikan |

Status: **SELESAI (20 Agu 2026)**. Temuan utama: belanja pemerintah daerah adalah satu-satunya faktor eksternal yang berkorelasi signifikan dengan penjualan alat kesehatan; demografi/epidemiologi/geografi tidak. Konsisten dengan sifat kelembagaan pengadaan alat kesehatan (anggaran dan program, bukan skala populasi). Masuk ke revisi.md Bagian 5 dan kesimpulan. Catatan: kolom belanja 2021 di file BPS dalam rupiah (bukan ribu rupiah) - dikoreksi /1000; jarak MRU digabung ke Musi Rawas memakai max.

**Verifikasi residual model (20 Agu 2026, `scripts/bps_residual_analysis.py`, hasil `results/bps_residual_analysis.json`):** korelasi residual model V3 (test 2025, per region, n=16) terhadap faktor BPS 2025: penduduk r=0,066 p=0,807; DBD r=0,274 p=0,305; RS/puskesmas r=0,478 p=0,061; sarana r=0,510 p=0,044; jarak r=-0,134 p=0,621; luas r=-0,341 p=0,196. Korelasi sarana yang tampak signifikan adalah **artefak**: file sarana_2025 hanya berisi 6 region bernilai (Palembang 25, sisanya 0-6) dan tanpa Palembang r=-0,158 p=0,800. Kesimpulan: TIDAK ADA faktor BPS yang menjelaskan residual model.

**Verifikasi within-region belanja (20 Agu 2026, temuan awal 3 tahun):** korelasi pooled belanja vs penjualan r=0,433 (p=0,002, n=48) ternyata **murni efek level antar region**. Setelah demean per region (menghilangkan efek skala), korelasi within-region r=-0,112 p=0,449 - TIDAK signifikan, bahkan negatif. Artinya variasi belanja antar tahun dalam region yang sama TIDAK memprediksi penjualan; yang berkorelasi hanya level (region besar = belanja besar = penjualan besar), dan level ini sudah ditangkap model dari data penjualan itu sendiri. **Keputusan final: eksperimen ulang dengan fitur BPS TIDAK dilakukan** - (1) korelasi belanja adalah efek skala, bukan sinyal temporal yang bisa dipelajari model; (2) faktor lain tidak signifikan terhadap penjualan maupun residual; (3) fitur tahunan/statis hampir konstan dalam window 30 hari; (4) risiko overfitting. Limitasi ini sudah tercatat di revisi.md. (Angka ini disupervisi oleh analisis gabungan 6 tahun di bawah.)

**Koreksi file belanja BPS (20 Agu 2026):** file "Belanja Pemerintah menurut kabupaten kota.csv" berisi 4 kolom (2020-2023), bukan 3 - script sebelumnya salah slice `columns[1:4]` (melewatkan 2023). Diperbaiki ke `columns[1:5]`. Format: 2020/2022/2023 = ribu rupiah, 2021 = rupiah (dikoreksi /1000). Korelasi belanja pooled 2020-2023: r=0,401 p=0,001 n=64.

**APBD Provinsi Sumsel 2025 (20 Agu 2026, `scripts/apbd_2025_analysis.py`, hasil `results/apbd_2025_analysis.json`):** user menyediakan `data/data_apbd.xls` - ternyata XML Spreadsheet 2003 (bukan .xls biner), berisi APBD **Provinsi Sumsel agregat** (bukan per kab/kota), realisasi s.d. Oktober 2025: Pendapatan Daerah 8,26 T (78,5%), Belanja Daerah 6,30 T (59,2%), Belanja Barang & Jasa 1,42 T (62,1%). Data agregat provinsi tidak bisa dipetakan ke 16 region. Catatan temporal penting: file lama (Realisasi Belanja Pemerintah.csv) berisi realisasi 2025 = 2,20 T (nilai akhir tahun), sedangkan file baru = 1,42 T (s.d. Oktober, saat test set berakhir) - file baru lebih tepat secara anti-leakage. Korelasi realisasi barang&jasa vs total penjualan dengan nilai 2025 dari file baru (n=6): r=0,314 p=0,544 - tetap TIDAK signifikan. Kesimpulan tidak berubah: APBD 2025 tidak mengubah hasil eksperimen.

**APBD 2025 per kab/kota dari portal DJPK (20 Agu 2026, `scripts/apbd_portal_fetch.py` + `scripts/apbd_2025_corr.py`, hasil `results/apbd_2025_per_pemda.json` + `results/apbd_2025_corr.json`):** nilai file user terverifikasi identik dengan portal resmi DJPK Kemenkeu (djpk.kemenkeu.go.id, periode=10, tahun=2025, provinsi=06, data SIKD per 19 Agu 2026). Diambil realisasi APBD 2025 s.d. Oktober untuk 17 pemda (16 kab/kota + MRU). Korelasi cross-sectional per region (n=16): Belanja Daerah r=0,256 p=0,338; Barang & Jasa r=0,252 p=0,347; per kapita r=0,263 p=0,324 dan r=0,487 p=0,056 - SEMUA TIDAK signifikan. Ini bukti tambahan dengan data resmi per kab/kota: belanja APBD bukan prediktor penjualan, konsisten dengan within-region 2020-2022 (r=-0,112). Efek skala yang tampak di pooled 2020-2022 (r=0,433) tidak konsisten di 2025 (penjualan 2025 anomali rendah, banyak region ~0). Keputusan final tetap: TIDAK ada eksperimen ulang dengan fitur BPS.

**APBD 2024 per kab/kota dari portal DJPK (20 Agu 2026):** diambil realisasi akhir tahun 2024 (periode=12) untuk 17 pemda (`results/apbd_2024_per_pemda.json`). Analisis gabungan 2020-2025 lengkap (`scripts/apbd_combined_analysis.py`, hasil `results/apbd_combined_analysis.json`; 2020-2023 BPS + 2024-2025 DJPK, semua dalam rupiah, n=96): pooled r=0,380 p=0,0001 (efek skala); **within-region r=0,058 p=0,576 - TIDAK signifikan** (konfirmasi kuat dengan 6 titik per region); cross-sectional tidak konsisten (2024 r=0,571 p=0,021 signifikan, 2025 r=0,256 p=0,338 tidak). Per region (n=6): semua tidak signifikan. Kesimpulan final diperkuat: dengan data belanja LENGKAP 2020-2025 sekalipun, belanja bukan prediktor temporal penjualan - keputusan TIDAK ada eksperimen ulang tetap valid, kini dengan alasan murni statistik (bukan keterbatasan data).

**Fokus Belanja Barang & Jasa (20 Agu 2026, `scripts/apbd_bj_analysis.py`, hasil `results/apbd_bj_analysis.json`):** atas permintaan user, analisis difokuskan pada Belanja Barang & Jasa (paling relevan untuk pengadaan alkes). Data BJ 2020-2025 diambil SEMUA dari portal DJPK (konsisten sumber; 2020-2024 periode=12 akhir tahun, 2025 periode=10 s.d. Oktober) - `results/apbd_2020..2025_per_pemda.json`. Hasil (n=96): pooled r=0,345 p=0,0006 (efek skala); **within-region r=0,103 p=0,318 - TIDAK signifikan**; cross-sectional TIDAK ADA tahun yang signifikan (2020 r=0,496 p=0,051 marginal, 2021 r=0,409 p=0,116, 2022 r=0,471 p=0,066 marginal, 2023 r=0,162 p=0,550, 2024 r=0,384 p=0,142, 2025 r=0,252 p=0,347); per region semua tidak signifikan. BJ lebih bersih dari belanja total: tidak ada tahun cross-sectional yang signifikan. Narasi revisi.md Bagian 5, limitasi, dan kesimpulan diperbarui dengan angka BJ sebagai fokus (belanja total sebagai konfirmasi).

### E8. Produk COVID bernilai tinggi (20 Agustus 2026)

Analisis produk mahal penanganan COVID (`scripts/covid_products_analysis.py`, hasil `results/covid_products_analysis.json`):
- **Airvo Unit Optiflow Therapy**: 762,5 jt IDR, 1 faktur, Agustus 2021 (puncak gelombang Delta), 3,2% dari total penjualan 2021.
- **Hyper-Hypothermia System Blanketrol III**: 441,5 jt IDR, 1 faktur, November 2023, 2,3% dari total 2023 (pengadaan non-pandemi).
- **Junior Optiflow Nasal Cannula** (aksesori): 10,0 jt IDR, 4 faktur, kecil.

Kesimpulan: transaksi tunggal bernilai besar adalah outlier yang berkontribusi pada volatilitas harian dan residual; Airvo Agustus 2021 = bukti dampak pandemi (gelombang Delta). Masuk ke revisi.md Bagian 5 (paragraf COVID).

### E9. Produk bernilai tinggi lain + pembelian qty besar (20 Agustus 2026)

Analisis lanjutan (`scripts/expensive_products_analysis.py`, hasil `results/expensive_products_analysis.json`):

**Produk dengan harga satuan > 50 jt (6 produk):** Blanketrol III 441,5 jt (RSUD Siti Fatimah Palembang, Nov 2023); Infant Warmer 420 jt (PT Bank Tabungan Negara, Jakarta Pusat, Agu 2021 - LUAR Sumsel); Bubble Infant nCPAP 179,4 jt (Yay. Rafflesia, Bengkulu, Agu 2021 - LUAR Sumsel); Airvo Optiflow 152,5 jt x 5 (Rumkit Bhayangkara Palembang, Agu 2021); GeNose 88,5 jt x 2 (Rumkit Bhayangkara, Jun 2021); Servis Juli 77,8 jt (jasa servis alat, Rumkit Bhayangkara Jambi, Jul 2024). Koreksi grain: harga satuan per baris item; Patient Monitor maks 40,1 jt/unit dan Neopuff 42,2 jt/unit di bawah ambang (angka lama "Patient Monitor 74,6 jt" artefak dedup).

**Produk neonatal (100 faktur, 940,5 jt = ~0,9% total):** Infant Warmer 420 jt + Bubble nCPAP 179,4 jt (keduanya pembeli di luar Sumsel); dalam Sumsel: Neonatal Nasal Ventilation Double Tube 8Fr 98,3 jt (56 faktur), 10Fr 92,0 jt (55 faktur), Neopuff 84,4 jt (Prabumulih). Pembeli: RS AR Bunda Prabumulih (118,3 jt, 17 faktur), RSUD Rabain Muara Enim (105,9 jt, 11 faktur). Puncak 2021 (612,9 jt) didominasi pembeli luar Sumsel.

**Pembelian qty besar (>= 10.000 unit/baris item, 177 faktur, 7,38 M IDR):** Disposable Syringe 6,60 jt unit / 5,78 M IDR (161 faktur), Auto Destruct Syringe 885 rb unit / 1,00 M IDR (13 faktur). Pembeli dominan: DINKES Kab. Muara Enim, RSUD Sekayu, RSUD Palembang Bari, RSUD Rabain, DINKES Musi Banyuasin - pola pengadaan massal jarum suntik oleh Dinas Kesehatan (program layanan/imunisasi).

**Keputusan data kelahiran:** produk neonatal hanya ~0,9% dari total penjualan dan pembeli terbesarnya (Infant Warmer, nCPAP) berada di luar Sumsel; data kelahiran per kab/kota (BPS Lahir Hidup) TIDAK wajib ditambahkan - dampaknya marginal terhadap model. Faktor anggaran yang sudah diuji (belanja APBD pooled r=0,345-0,380, efek skala bukan prediktor temporal) tetap yang paling relevan secara kontekstual. Jika ingin lengkap, data kelahiran BPS bisa dicari, tapi tidak mengubah kesimpulan.

### E10. Kandidat faktor X: struktur fasilitas kesehatan - indikator kinerja RS 2024 (20 Agustus 2026)

User menyediakan `data/Indikator kinerja pelayanan di RS Provinsi Sumatera Selatan.json` (88 RS, tahun 2024): jumlah tempat tidur, pasien keluar, hari perawatan, BOR, BTO, TOI, ALOS. Ini menguji kandidat RS yang muncul dari analisis BPS (RS_umum pooled r=0,298, RS_khusus pooled r=0,422, residual RSK r=0,572).

**Script:** `scripts/rs_kinerja_2024.py` | **Hasil:** `results/rs_kinerja_2024.json`

**Pemetaan 88 RS ke 16 region:** 8 nama RS ambigu diverifikasi via web - Karunia Indah Medika → Muara Enim (Muara Lawai), Trijaya Medical Center → Muara Enim (Tanjung Enim), RSIA Adellia Graha Medika → Lahat, RSUD Sukajadi → Banyuasin (Talang Kelapa), RS Musi Medika Cendikia → Palembang (Ilir Barat I), RSUD Petanang → Lubuk Linggau (Petanang Ilir), RS Mahyuzahra → Ogan Ilir (Indralaya), RS Ar-Royyan → Ogan Ilir (Indralaya Utara). Keputusan: Hermina OPI Jakabaring dan Bunda Medika Jakabaring → Palembang (operasional di Jakabaring; alamat administratif tercatat Banyuasin di direktori Sun Life). Rupit (Musi Rawas Utara) → Musi Rawas (konsisten dengan merge region penjualan). Total 88 RS, 10.161 TT (direktori Kemenkes mencatat 89 RS / 10.491 TT; JSON tidak mencakup 1 RS baru).

**Hasil korelasi (n=16):**

| Faktor | vs penjualan 2024 | vs residual V3 2025 |
|---|---|---|
| **Jumlah RS** | **r=+0,669 p=0,005** | **r=+0,697 p=0,003** |
| Tempat tidur | r=+0,577 p=0,019 | r=+0,615 p=0,011 |
| Pasien keluar | r=+0,522 p=0,038 | r=+0,643 p=0,007 |
| Hari perawatan | r=+0,574 p=0,020 | r=+0,618 p=0,011 |
| BOR | r=+0,478 p=0,061 (marginal) | r=+0,458 p=0,075 (marginal) |
| BTO | r=+0,396 p=0,129 | r=+0,514 p=0,042 |
| TOI / ALOS | tidak signifikan | tidak signifikan |

**Sensitivitas:** tanpa Palembang (n=15) jumlah RS tetap signifikan (sales2024 r=+0,596 p=0,019; resid2025 r=+0,631 p=0,012); kontrol penduduk (parsial, n=10) jumlah RS r=+0,648 p=0,043 dan tempat tidur r=+0,648 p=0,043 - signifikan setelah mengontrol populasi.

**Interpretasi:** struktur fasilitas kesehatan (jumlah RS, kapasitas tempat tidur, volume pasien) adalah kandidat faktor X TERKUAT yang pernah diuji - konsisten lintas lapis (cross-sectional, residual, tanpa Palembang, kontrol penduduk) dan lintas sumber (BPS rs_2020-2025 pooled RS umum r=0,298 / RS khusus r=0,422; residual BPS 2025 RSK r=0,572 / desa_RS r=0,510). Mekanisme: RS = pembeli alkes terbesar. Efisiensi operasional (TOI/ALOS) tidak relevan. Memperkuat kesimpulan naskah: penjualan didorong struktur layanan kesehatan, bukan anggaran umum.

**Keterbatasan:** data 1 tahun (2024) → bukti struktural cross-sectional, bukan temporal; jumlah RS hampir statis antar tahun; n=16 kecil + banyak faktor diuji (risiko false positive). TIDAK dipakai sebagai fitur model (konsisten dengan keputusan E7: fitur statis hampir konstan dalam window 30 hari). Masuk ke temuan.md 2.6 dan revisi.md Bagian 5.

### E11. Distribusi temporal pengadaan syringe + uji DBD/COVID (20 Agustus 2026)

Lanjutan E9 (pembelian qty besar). Analisis temporal pengadaan syringe (`scripts/syringe_temporal.py`, hasil `results/syringe_temporal.json`) dan uji korelasi dengan DBD/COVID. Skrip uji lama (`syringe_dbd.py`, `syringe_covid.py`) terhapus saat rapikan folder dan memakai qty hasil dedup per faktur (basi); direkonstruksi 21 Agustus 2026 sebagai `scripts/syringe_wabah.py` dengan grain baris-item (hasil `results/syringe_wabah.json`). Angka di bawah sudah dikoreksi.

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

**Uji DBD (qty syringe vs kasus DBD per 100.000, region-tahun, n=77):** pooled r=+0,048 p=0,677; within-region r=+0,072 p=0,536 — TIDAK ada hubungan. Tahun DBD tertinggi (2024: 70,4/100k menurut baris provinsi BPS) bukan tahun syringe tertinggi; 2021 syringe tertinggi justru DBD terendah (13,4/100k).

**Uji COVID (qty syringe per bulan vs fase pandemi):** pandemi (Jul 2020-Des 2022) mean 220.398/bulan vs endemi (2023+) 169.418/bulan — Mann-Whitney p=0,614 (tidak berbeda); Spearman qty vs pandemi r=+0,064 p=0,613. Puncak tertinggi justru Jul 2022 (709.779); Omicron (Jan-Feb 2022) syringe terendah (37.659 dan 57.854).

**Konteks Muara Enim (pembeli syringe terbesar):** kasus kesehatan 2023 didominasi ISPA, TB Paru, dan PD3I — 6 bayi pertusis dan 13 campak akibat penurunan cakupan imunisasi dasar. Pengadaan syringe DINKES terkait **program imunisasi rutin/kejar imunisasi**, bukan respons wabah DBD/COVID.

**Kesimpulan:** pengadaan massal syringe = kebutuhan rutin fasilitas kesehatan (imunisasi, pelayanan), didorong siklus anggaran, bukan wabah. Konsisten dengan E10: struktur fasilitas kesehatan adalah kandidat faktor X terkuat.

### E12. Anggaran barang & jasa Dinkes Provinsi Sumsel (20 Agustus 2026)

User menyediakan 3 file anggaran: `data/alokasi anggaran kesehatan provinsi sumsel 2022.json`, `data/realisasi anggaran kesehatan provinsi sumsel 2023.json`, `data/alokasi anggaran kesehatan provinsi sumsel 2024.json` (semua tingkat Dinkes Provinsi). Verifikasi silang dengan Laporan Monev TW IV Dinkes 2022 (`data/monev_dinkes_2022.pdf`, pdf2.sumselgo.id), LKjIP Dinkes 2023 (`data/lakip_dinkes_2023.pdf`), LKjIP Dinkes 2024 (`data/lakip_dinkes_2024.pdf`), dan satudata1.sumselprov.go.id (realisasi belanja provinsi 2022). Script: `scripts/anggaran_compare.py`, log `logs/anggaran_compare.log`, `logs/anggaran_bj.log`.

| Tahun | Belanja barang & jasa Dinkes | Penjualan (PT Parit Panjang) |
|---|---|---|
| 2022 | **Belanja Barang dan Jasa Rp36,39 M** (DPPA TA 2022 setelah perubahan; sebelum perubahan Rp44,37 M); belanja pegawai Rp181,42 M pagu / Rp171,73 M realisasi; belanja hibah Rp5,11 M; belanja modal Rp23,60 M pagu / Rp22,31 M realisasi | Rp6,96 M |
| 2023 | **Program Sediaan Farmasi, Alkes & Makanan Rp36,14 M alokasi / Rp28,12 M realisasi (77,81%)** — komponen paling relevan alkes; kegiatan pengadaan obat/vaksin/alkes Rp15,83 M realisasi; total Rp304,38 M / Rp253,26 M (83,20%) | Rp8,02 M |
| 2024 | **Belanja Barang dan Jasa Rp283,04 M** (eksplisit, 58% dari total Rp486,52 M; mencakup program sediaan farmasi/alkes: BHP medis, pemeliharaan alkes, kalibrasi rutin); pegawai Rp164,52 M; **belanja modal Rp16,19 M = pengadaan aset alkes** (faskes rujukan provinsi, UPTD/BLKP/RS pemprov); hibah Rp4,08 M; DAK BOK Rp11,59 M | Rp5,82 M |

**Breakdown program 2023 (alokasi setelah perubahan / realisasi):** Penunjang Urusan Pemerintahan (gaji, operasional, BJ kantor) Rp94,13 M / Rp82,52 M (87,66%); Pemenuhan Upaya Kesehatan (layanan masyarakat & rujukan, termasuk JKN Rp122,47 M) Rp158,46 M / Rp129,54 M (81,75%); Pencegahan & Pengendalian Penyakit Rp10,75 M / Rp8,87 M (82,56%); Peningkatan Kapasitas SDM Kesehatan Rp4,89 M / Rp4,20 M (85,84%); **Sediaan Farmasi, Alkes & Makanan Rp36,14 M / Rp28,12 M (77,81%)** — komponen paling relevan alkes, di dalamnya kegiatan pengadaan obat/vaksin/alkes Rp15,83 M realisasi. Mayoritas anggaran = transfer JKN + gaji, bukan pengadaan alkes langsung.

**Mekanisme pengadaan 2024 (konteks dari user):** Dinkes Provinsi membeli BMHP, obat program (TBC, malaria, HIV), dan alkes tertentu secara TERPUSAT di tingkat provinsi (dari BJ Rp283,04 M), lalu mendistribusikan fisik ke Gudang Farmasi Kabupaten/Kota. Dalam laporan keuangan provinsi, nilai ini dicatat sebagai satu kesatuan BJ operasional provinsi, bukan di-breakdown per kabupaten. Implikasi: (1) penjualan distributor ke kab/kota/RS tidak tercermin langsung di belanja provinsi — korelasi penjualan dengan anggaran provinsi lemah karena jalur pengadaannya berbeda; (2) permintaan riil tetap ditentukan fasilitas kesehatan di kab/kota (struktur RS), konsisten dengan 2.6.

**Temuan:** arah BERLAWANAN dengan penjualan — total belanja barang & jasa naik 7,78x (Rp36,39 M → Rp283,04 M, 2022→2024), penjualan turun 16% (Rp6,96 M → Rp5,82 M). CATATAN PENTING: Rp283,04 M = TOTAL BJ Dinkes (operasional, jasa, perjalanan dinas, dll), BUKAN khusus alkes. Komponen alkes 2024 yang pasti hanya belanja modal aset alkes Rp16,19 M; program sediaan farmasi/alkes (BHP, pemeliharaan, kalibrasi) ada di dalam BJ tapi tidak eksplisit. Pembanding: program sediaan farmasi/alkes/makanan 2023 = Rp36,14 M alokasi / Rp28,12 M realisasi. Dengan komponen alkes yang lebih akurat (2023: Rp28,12 M realisasi program alkes; 2024: Rp16,19 M modal alkes + bagian BJ), arah tetap tidak sejalan dengan penjualan (2023 Rp8,02 M → 2024 Rp5,82 M). Belanja barang & jasa 2024 (Rp283 M) = 49x penjualan 1 distributor (wajar: mencakup jasa, pemeliharaan, perjalanan dinas, bukan hanya alkes; Dinkes belanja dari banyak distributor). Data satudata (realisasi belanja provinsi 2022: barang & jasa Rp2,04 triliun) adalah seluruh APBD provinsi (semua OPD), terlalu makro untuk dikorelasikan dengan penjualan alkes.

**Keterbatasan:** belanja barang & jasa 2023 total tidak eksplisit (hanya per program; program sediaan farmasi/alkes/makanan Rp36,14 M adalah bagian darinya); komponen alkes 2024 tidak eksplisit (hanya belanja modal aset alkes Rp16,19 M yang pasti; program sediaan farmasi/alkes ada di dalam BJ Rp283,04 M tanpa rincian); data tingkat provinsi (1 nilai/tahun, n=3) → hanya deskriptif, tidak bisa korelasi statistik; penjualan = 1 distributor.

**Kesimpulan:** komponen belanja yang relevan (barang & jasa, pengadaan obat/alkes) juga tidak sejalan dengan penjualan — memperkuat posisi struktur fasilitas kesehatan (E10) sebagai kandidat faktor X terkuat. Masuk ke temuan.md 2.7.

### E13. Konteks epidemiologi & bencana Sumsel 2020-2026 (20 Agustus 2026)

Data konteks dari user (Dinkes Sumsel, BPBD Sumsel) — agregat/kualitatif, belum diuji korelasi statistik per region.

**Wabah & penyakit menular:**
- **COVID-19 (2020-2023):** pandemi, puncak 2020 & pertengahan 2021 (Delta), BOR naik di Palembang, endemi pertengahan 2023. Konsisten dengan E11 (uji COVID vs syringe tidak signifikan) dan temuan Airvo Agustus 2021 (event tunggal Delta).
- **DBD (2024-2025):** lonjakan signifikan; 2025 tercatat 4.130-6.265 kasus, tren naik September-November, Palembang tertinggi. Konsisten dengan BPS (2024: 70,4/100k, tertinggi sepanjang 2020-2025); uji DBD vs syringe tidak signifikan (E11).
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
- Volume penyakit tinggi & konsisten (ISPA 390 ribu, DBD, TBC 24 ribu, diare) = volume pasien tinggi = permintaan alkes stabil melalui fasilitas kesehatan — konsisten dengan E10 (pasien keluar r=+0,522).
- Wabah spesifik (COVID, DBD) tidak berkorelasi dengan penjualan syringe (E11); hanya event tunggal COVID (Airvo, Delta 2021) yang terlihat di data.
- Bencana (banjir, karhutla) menambah kebutuhan logistik/alkes tapi tidak mengubah struktur permintaan.
- Konteks Muara Enim (pembeli syringe terbesar): ISPA, TBC, PD3I tinggi — pengadaan terkait program imunisasi & layanan rutin, bukan wabah.

**Keterbatasan:** data agregat kualitatif tanpa rincian temporal per region yang bisa diuji korelasi; TIDAK dipakai sebagai fitur model. Masuk ke temuan.md 2.8.
## F. Audit Naskah & Perbaikan (26 Agustus 2026)

### F1. Graf korelasi di Tabel 6 DEGENERATE (bug ditemukan)

Verifikasi: korelasi |Pearson| antar region pada panel penuh MAKSIMAL 0,332 (120 pasangan), sehingga threshold 0,5 menghasilkan NOL edge. Graf kosong setelah normalisasi = adjacency identitas -> hasil bit-identik dengan baris Identity (RMSE 8.028.042, R2 0,0514 sama persis). Baris "Correlation graph" lama TIDAK menguji graf korelasi sungguhan.

Perbaikan: rerun semua sel ablasi dalam satu proses dengan init terpasang (`scripts/ablation_rerun_paired.py`, hasil `results/ablation_rerun_paired.json`). Varian korelasi baru: k=3 tetangga terkorelasi, korelasi dari periode training saja (bebas kebocoran); threshold = min korelasi tangga-3 antar node (0,0126; 67 edge).

### F2. Ketidakcocokan figure vs naskah (diperbaiki)

`plot_all_figures.py` semula memakai ensemble HybridTuned seed 42 untuk fig05-fig09, padahal Tabel 7 dan residual Bagian 4.6 naskah memakai Hybrid base ensemble (per_region_hybrid/residual_hybrid). Bukti: Palembang Tabel 7 (RMSE 23.964.716, R2 -0,3336) cocok dengan Hybrid base, bukan Tuned (26.337.036 / -0,6106). Diperbaiki: semua figure model diregenerasi dari `models.Hybrid.preds_mean` + `per_region_hybrid`; judul internal diganti "Hybrid LSTM-GNN"; fig10 ablation caption "tuned configuration, seed 42"; angka edge fig01 dibuat dinamis.

### F3. Klaim COVID lintas-konfigurasi (tidak valid) -> eksperimen pengganti

Klaim lama "removing it decreases the R-squared from 0.0523 to 0.0470" membandingkan Hybrid base (128/128/do0) vs HybridTuned (128/64/do0.2) sekaligus - dua variabel berubah. Eksperimen same-config pertama (`scripts/covid_ablation_base.py`, hasil `results/covid_ablation_base.json`): Hybrid base tanpa-COVID 3 seed -> mean R2 0,0484 (42: 0,0502; 7: 0,0479; 123: 0,0471) vs dengan-COVID 0,0523 (referensi). Arah klaim terkonfirmasi (+0,004). Catatan: sanity retrain with-COVID seed 42 memberi 0,0601 vs referensi 0,0572 karena bobot awal bergantung state RNG global saat pembuatan model -> perbandingan antar-proses tidak terpasang. Solusi final: rerun paired-init satu proses (F1); klaim COVID akan diambil dari sel distance-dgn-COVID vs tanpa-COVID pada tabel yang sama. DM test versi covid_ablation_base NaN (variansi negatif setelah koreksi ACF) dan bootstrap CI salah sumbu ([0; 24,9 jt]) - dua bug script, tidak dipakai; DM versi sederhana (konsisten metodologi naskah) dipakai di rerun paired.

### F4. Penomoran tabel & referensi gambar (diperbaiki)

Tabel: naskah melompat Table 1 -> Table 3; tabel arsitektur di 3.5 kini diberi caption **Table 2**. Gambar: fig01-fig04, fig09, fig10 semula tidak dirujuk teks; kini Fig 2-3 dirujuk di 3.4, Fig 1 + Table 2 di 3.5, Fig 4 di 4.1, Fig 10 di 4.4, Fig 9 di 4.5. Seluruh 10 gambar kini dirujuk.

### F5. Klarifikasi cakupan fraksi nol

Abstrak: "81.54% of daily region-level targets are zero" -> ditambah kualifikasi panel penuh + angka test 84,85%. balasan_reviewer.md poin 3 disesuaikan. Angka "+0,102" stale di balasan_reviewer.md (catatan protokol) dikoreksi menjadi "+0,043 s.d. +0,052" sesuai E1.

### F6. Eksperimen fitur rolling + atribut node (SELESAI)

Hasil (`results/exp_rolling_nodefeat.json`, 26 Agu 2026):
- **W0 (V2 replikasi, acuan):** ensemble R2 0.0592, RMSE 7.815.386, Palembang 22.978.653
- **W1 (rolling mean/std 7 & 30 hari):** R2 0.0591, DM p = 0.763 -> tidak signifikan. LSTM sudah menangkap smoothing secara internal.
- **W2 (+ jumlah RS sebagai atribut node GCN):** R2 0.0609, DM = -2.861, p = 0.004 -> signifikan. Palembang RMSE turun 1,1% (22.729k vs 22.979k). Bukti mekanistis: infrastruktur kesehatan masuk via jalur GCN.
- **W3 (+ log luas daerah):** R2 0.0609 = W2, tidak menambah. Konsisten dengan korelasi BPS (r = -0,341, p = 0,196).

Catatan: W0 seed-42 (0.0661) tidak identik dengan V2 lama (0.0614) karena bobot awal model bergantung state RNG saat pembuatan; selisih dalam rentang variansi seed. Gain W2 (+0.0017) nyata tapi W2 (0.0609) masih di bawah V3 (0.0666). Narasi masuk ke Discussion revisi.md (paragraf sebelum "At the provincial level").

### F7. Koreksi `holidays.py` + panel gap pada hari kerja kosong (selesai)

Daftar `scripts/holidays.py` sebelumnya kehilangan 33 tanggal cuti bersama/libur nasional (Idulfitri CB 2021-2025, Imlek/Nyepi/Waisak/Iduladha/Natal CB, Pemilu 14 Feb 2024, Pilkada 27 Nov 2024) dan memuat 2 tanggal salah (TB Islam 1443 H geser 10->11 Agu 2021; Maulid Nabi 1443 H geser 19->20 Okt 2021). Dikoreksi penuh (26 Agustus 2026).

Hasil: dari 436 hari kosong, hanya **21 hari kerja murni** (sebelumnya terhitung 54 karena libur yang terlewat). Forensik D-3 s.d. D+3 menunjukkan pola: (a) jembatan long-weekend tidak resmi; (b) PPKM/PSBB transisi Palembang Agustus 2020; (c) pull-forward akhir bulan; (d) fluktuasi B2B tanpa pola khusus. Detail: `docs/temuan.md` 3.6.

Eksperimen `scripts/exp_libur.py` membuktikan hari-hari ini **tidak terjangkau arsitektur LSTM-GNN saat ini**: panel hanya memuat hari transaksi (hasil pivot table); hari tanpa penjualan bukan sampel, sehingga fitur apapun bernilai nol di semua split. Flag jendela Idulfitri/Natal justru membuat model lebih buruk signifikan (R2 0,0661 vs 0,0670 baseline; DM = +4,107, p < 0,0001). Insight: hari kerja kosong adalah persoalan cakupan kalender, bukan fitur prediksi dalam desain sekarang.

Catatan: `results/V3_ref` berisi cache yang sekarang **stale** untuk eksperimen kalender berikutnya karena `is_hol` di panel sudah berubah.

### F8. Rerun ablasi paired-init (SELESAI, 26 Agu 2026)

Masalah: eksperimen ablasi lama menghasilkan graf korelasi degenerate (0 edge, identik dengan identitas), dan angka COVID lintas-konfigurasi (0.0523 vs 0.0470). Solusi: `scripts/ablation_rerun_paired.py` melatih ulang semua varian dalam satu proses dengan init identik (HybridTuned: LSTM 128, GCN 64, dropout 0.2; seed 42). Hasil (`results/ablation_rerun_paired.json`):

| Varian | RMSE | MAE | R2 | DM vs Distance | p |
|---|---|---|---|---|---|
| Identity | 7.964.173 | 1.871.263 | 0.0538 | -8.041 | < 0.001 |
| Random (mean 5) | 8.101.548 | 1.906.271 | 0.0486 | — | — |
| Distance k=3 | 8.235.956 | 1.941.944 | 0.0436 | (acuan) | — |
| **Correlation k=3 (training-only)** | 8.063.890 | 1.895.385 | **0.0501** | -8.180 | < 0.001 |
| Star Palembang | 7.504.506 | 1.806.491 | 0.0703 | -6.353 | < 0.001 |
| Tanpa COVID | 8.262.716 | 1.949.778 | 0.0425 | +6.278 | < 0.001 |

Korelasi k=3 baru: threshold 0.0126 (training-only), 67 edge (bukan 0). R2 0.0501 mengungguli distance tapi di bawah identitas -- sinyal spasial memang tipis. COVID same-config: 0.0436 vs 0.0425 (delta +0.0011, DM p < 0.001). Semua angka Tabel 6, 4.4, dan Discussion di `revisi.md` sudah diperbarui.

### F9. Atribut node GCN pada HybridTuned (selesai, 27 Agu 2026)

Eksperimen: apakah jumlah RS sebagai atribut node GCN memperbaiki HybridTuned (128/64/do0.2) — konfigurasi yang dipakai di Tabel 7 naskah? Protocol: paired init, seeds 42/7/123, distance graph k=3, pipeline identik.

Hasil (`results/exp_nodeattr_tuned.json`):

| Varian | R2 ens. | RMSE ens. | Palembang RMSE | DM vs V2_ref | p |
|---|---|---|---|---|---|
| **V2_ref** (acuan) | 0.0508 | 8.044.312 | 24.209.473 | — | — |
| **T1_hosp** (+jumlah RS node) | **0.0528** | **7.990.621** | **23.923.516** | **-6.037** | **< 0,0001** |

T2 (rolling + RS) dan T3 (rolling + RS + area) tidak selesai karena timeout — training dengan 16 fitur input terlalu lambat. Namun T2/T3 tidak kritis: rolling sudah terbukti tidak membantu di base config (exp_rolling_nodefeat W1 DM p=0,763). Temuan utama sudah final: **jumlah RS secara signifikan memperbaiki HybridTuned (p < 0,0001)**.

Interpretasi: Palembang memiliki 33 RS (44% total provinsi), kabupaten lain 1-8 RS. Jumlah RS berfungsi sebagai proxy kapasitas kesehatan regional yang mempengaruhi pola pengadaan alat kesehatan. Informasi ini masuk via jalur GCN sehingga embedding node mencerminkan struktur infrastruktur kesehatan, bukan hanya kedekatan geografis.

### F10. Temuan domain: lag pembayaran (informatif)

Berdasarkan informasi dari user (27 Agu 2026): dataset mencatat tanggal faktur (order/pembelian), bukan tanggal pembayaran. Pembayaran dilakukan pelanggan dengan mekanisme **tempo 30 hari / tanggal yang sama bulan berikutnya** (misalnya faktur 3 Maret 2024, pembayaran 3 April 2024).

Implikasi: model memprediksi pola order, bukan cash flow. Ini lebih relevan untuk supply chain planning. Fluktuasi akhir tahun (push order sebelum tutup buku) tercatat di data sebagai order Desember, meskipun payment masuk Januari. Hari kerja kosong mencerminkan tidak adanya order baru di hari tersebut, bukan penundaan pembayaran.

### F11. Analisis karhutla El Nino 2023 (selesai, 27 Agu 2026)

**Script:** `cek_karhutla.py`, `cek_karhutla2.py` (adhoc analysis)
**Sumber konteks:** BNPB hotspot data; The Conversation (2023) — El Nino memicu karhutla di Sumatra sejak Desember 2022, puncak Agst-Spt 2023.

Temuan:
- **Penjualan aggregate Agst-Spt tidak naik saat karhutla:** 2023 Agst+Spt = 1.961 M IDR (403 faktur), justru turun dari 2022 (2.016 M, 391 faktur), jauh di bawah puncak COVID 2021 (4.288 M).
- **Agst 2023** = 1.189 M (215 faktur); **Spt 2023** = 772 M (188 faktur) = salah satu bulan terendah 2023. Puncak sebenarnya: Nov 2023 = 1.825 M (belanja akhir tahun), Jun 2023 = 1.400 M (akhir FY).
- **Produk respiratory tidak spike:** Spt 2023 respiratory = 90,8 M (27 faktur), terendah sejak Maret 2023. Faktur > 50 M di Agst-Spt 2023 hanya 1 (Palembang, JVA Tech, 141,7 M).
- **Produk level-item:** masker TIDAK melonjak. Yang naik IV Catheter x21,9, Infusion Set x6,5 (pola belanja umum, bukan respons asap).
- **Aggregate per tahun (Agst+Spt):** 2020 1.109 M; 2021 4.288 M (COVID); 2022 2.016 M; 2023 1.961 M (El Nino); 2024 1.902 M; 2025 1.532 M. Tren menurun, tanpa spike karhutla.

**Implikasi untuk naskah:** Karhutla 2023 ditambahkan ke paragraf Discussion (Section 5) sebagai bukti empiris bahwa event darurat alam TIDAK drive permintaan alkes di level distributor. Mengkuatkan narasi bahwa infrastruktur kesehatan (jumlah RS) adalah driver utama, bukan event eksternal.
