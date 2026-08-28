# Outline Paper Lainnya (Selain #4)

Dokumen ini berisi kerangka naskah untuk usulan judul #1, #2, #3, #5, #6, #7, #8, dan #9.
Outline #4 ada terpisah di `docs/outline_paper_4_infrastruktur.md`.
Semua angka terverifikasi dari `temuan.md`, `usulan_judul_jurnal.md`, `balasan_reviewer.md`, dan `daftar_tbd.md`.
Status: outline draf (28 Agu 2026). Belum ditulis naskah penuh.

---

# Outline #1 — Topologi Jaringan Distribusi vs Kedekatan Geografis

**Judul kerja:** *When Geography Lies: Distribution Network Topology Outperforms Spatial Proximity in Graph-Based Regional Demand Forecasting*

**Sasaran:** jurnal logistics/supply chain analytics atau applied AI.

## Ringkasan satu paragraf (untuk abstrak)

Menggunakan data transaksi distributor alat kesehatan di Sumatera Selatan (16 region, 2020-2025), kami menguji bagaimana pilihan konstruksi graf mempengaruhi kinerja forecasting berbasis GNN. Graf bintang distribusi (Palembang sebagai pusat ke semua region) mencapai R2 0,0703 — terbaik di semua varian ablasi, naik +61% relatif atas graf jarak (0,0436). Graf jarak justru lebih buruk daripada adjacency identitas (0,0538) dan graf acak (0,0486). Korelasi penjualan antar region maksimal 0,332, menunjukkan struktur spasial "alamiah" tipis. Kesimpulan: untuk GNN forecasting pada bisnis distribusi, tanyakan dulu bagaimana barang benar-benar bergerak, bukan di mana lokasinya.

## 1. Pendahuluan
- Latar: GNN forecasting spatiotemporal umumnya membangun graf dari kedekatan geografis.
- Kesenjangan: asumsi "dekat secara geografis = saling mempengaruhi" sering tidak berlaku pada jaringan distribusi yang terpusat.
- Pertanyaan riset: apakah topologi jaringan distribusi aktual mengungguli kedekatan spasial dalam forecasting permintaan regional?
- Kontribusi: panduan praktis pemilihan graf untuk GNN forecasting pada bisnis distribusi.

## 2. Data dan Metode
### 2.1 Data
- Transaksi distributor alkes Sumsel 2020-2025, 16 region, target nilai faktur per region per hari.
- Konfirmasi rute distribusi: semua berpusat di Palembang (Komplek Villa Kenten, Palembang).

### 2.2 Konstruksi graf (varian ablasi)
- Jarak (Haversine k=3), korelasi penjualan, acak (5 permutasi), identitas, bintang (star dari Palembang), jaringan aktual.

### 2.3 Evaluasi
- Model LSTM-GNN zero-inflated, split 3 arah, ensemble 3 seeds, uji Diebold-Mariano.

## 3. Hasil
- Star graph R2 0,0703 (terbaik, +61% atas graf jarak 0,0436).
- Graf jarak (0,0436) < identitas (0,0538) < acak (0,0486).
- Korelasi penjualan antar region maksimal 0,332 (struktur spasial tipis).

## 4. Diskusi
- Interpretasi: distribusi terpusat membuat topologi jaringan lebih informatif daripada geografi.
- Implikasi praktis: desain graf harus mengikuti aliran barang aktual.
- Keterbatasan: satu domain aplikasi; perlu generalisasi ke dataset publik.

## 5. Kesimpulan
- Pilihan graf adalah keputusan desain yang menentukan; tanya bagaimana barang bergerak.

## Catatan pengembangan
- Perlu generalisasi ke dataset publik (transportasi/retail) untuk memperkuat klaim.
- Angka kunci: R2 0,0703 / 0,0436 / 0,0538 / 0,0486; korelasi max 0,332.

---

# Outline #2 — Granularitas Agregasi dan Struktur Nol

**Judul kerja:** *Aggregation Destroys the Signal: Zero-Inflated Daily Forecasting of Sparse Regional Demand*

**Sasaran:** jurnal forecasting (International Journal of Forecasting, applied track).

## Ringkasan satu paragraf (untuk abstrak)

Kami menguji pada resolusi waktu apa permintaan regional yang sparse dapat diprediksi. Panel bulanan menghasilkan R2 -0,394 dan mingguan -0,343, sementara harian zero-inflated mencapai +0,043 s.d. +0,052. Fraksi nol 81,54% (panel penuh) dan 84,85% (test). ARIMA dan XGBoost tidak mengungguli baseline always-zero. Temuan: pilihan resolusi waktu dan formulasi zero-inflated menentukan naik-tidaknya model, lebih besar daripada perbedaan antar arsitektur (rentang gain arsitektur hanya 0,005-0,01).

## 1. Pendahuluan
- Latar: forecasting permintaan sparse sering diagregasi ke bulanan untuk menghindari nol.
- Kesenjangan: agregasi dapat menghancurkan sinyal temporal.
- Pertanyaan riset: pada resolusi apa permintaan ini bisa diprediksi?
- Kontribusi: bukti empiris bahwa resolusi + formulasi ZI lebih menentukan daripada arsitektur.

## 2. Data dan Metode
### 2.1 Data
- Transaksi alkes Sumsel, 16 region, 2020-2025.

### 2.2 Eksperimen granularitas
- Bulanan / mingguan / harian x beberapa keluarga model (LSTM-GNN, ARIMA, XGBoost, always-zero).

### 2.3 Dekomposisi
- Kontribusi klasifikasi vs regresi jumlah.

## 3. Hasil
- Bulanan R2 -0,394; mingguan -0,343; harian ZI +0,043 s.d. +0,052.
- Fraksi nol 81,54% / 84,85%.
- ARIMA & XGBoost tidak mengungguli always-zero.

## 4. Diskusi
- Interpretasi: agregasi menghancurkan sinyal harian.
- Implikasi: formulasi ZI wajib untuk data sparse.
- Keterbatasan: satu dataset.

## 5. Kesimpulan
- Pertanyaannya "pada resolusi apa bisa diprediksi", bukan "model apa terbaik".

## Catatan pengembangan
- Angka kunci: R2 -0,394 / -0,343 / +0,043-0,052; fraksi nol 81,54% / 84,85%.

---

# Outline #3 — Ritme Institusional Pengadaan (Efek Hari)

**Judul kerja:** *The Institutional Rhythm of Healthcare Procurement: Day-of-Week Patterns in Medical Device Ordering*

**Sasaran:** jurnal healthcare operations / health services management.

## Ringkasan satu paragraf (untuk abstrak)

Kami mengkarakterisasi perilaku pemesanan instansi kesehatan melalui pola hari-dalam-minggu. Encoding day-of-week one-hot + libur + fase pandemi menaikkan ensemble R2 dari 0,0523 ke 0,0612 (DM p < 0,001). Residual menunjukkan over-prediksi hari kerja (Senin +22,1 jt) dan under-prediksi akhir pekan. Encoding weekend biner tidak signifikan (p = 0,082), menegaskan perlunya encoding penuh hari. Implikasi praktis untuk penjadwalan pengiriman dan staf gudang.

## 1. Pendahuluan
- Latar: perilaku pemesanan institusi kesehatan mengikuti ritme operasional.
- Pertanyaan riset: bagaimana pola hari-dalam-minggu dalam pengadaan alkes?
- Kontribusi: karakterisasi perilaku + implikasi operasional.

## 2. Data dan Metode
### 2.1 Data
- Level faktur (unit analisis faktur, bukan panel harian agregat).

### 2.2 Metode
- Regresi efek tetap hari/libur pada level faktur dan region (bisa tanpa deep learning).
- Robust, mudah direplikasi.

## 3. Hasil
- Day-of-week one-hot: R2 0,0523 -> 0,0612 (DM p < 0,001).
- Residual Senin +22,1 jt (over-prediksi hari kerja); under-prediksi akhir pekan.
- Weekend biner tidak signifikan (p = 0,082).

## 4. Diskusi
- Interpretasi: ritme institusional pengadaan.
- Implikasi praktis: penjadwalan pengiriman dan staf gudang.

## 5. Kesimpulan
- Encoding penuh hari diperlukan; perilaku pemesanan institusi terpola.

## Catatan pengembangan
- Unit analisis faktur (bukan panel harian agregat) — beda dari naskah utama.
- Angka kunci: R2 0,0523 -> 0,0612; DM p < 0,001; residual Senin +22,1 jt; weekend p = 0,082.

---

# Outline #5 — Permintaan Event-Driven Masa Pandemi

**Judul kerja:** *Event-Driven Demand Spikes in Healthcare Supply Chains: Evidence from Indonesia's COVID-19 Emergency Procurement*

**Sasaran:** jurnal disaster/public health preparedness atau supply chain risk management.

## Ringkasan satu paragraf (untuk abstrak)

Kami menganalisis anatomi lonjakan pengadaan darurat alat kesehatan selama pandemi COVID-19 di Indonesia. Grup produk COVID bernilai 99,3 M IDR (19,7% total). Gelombang Delta Jul-Agu 2021 memicu lonjakan x5,85 median bulanan (handscoon x43,9, rapid test x25,4, auto-destruct syringe x10,0); satu invoice Airvo 762,5 jt. Sebaliknya, karhutla El Nino 2023 TIDAK memicu lonjakan masker (diverifikasi dengan normalisasi satuan Box/Pcs ke pcs). Kesimpulan: spike pengadaan darurat terkonsentrasi waktu dan produk, dan sinyalnya tak tertangkap model berbasis riwayat.

## 1. Pendahuluan
- Latar: pengadaan darurat alkes saat pandemi.
- Pertanyaan riset: apa yang melonjak bersamaan, seberapa terkonsentrasi, mengapa tak tertangkap model riwayat?

## 2. Data dan Metode
### 2.1 Data
- Level produk, periode 2020-2022 (periode terbatas).

### 2.2 Metode
- Analisis temporal produk-level (spike factor vs median).
- Pemetaan produk-pemicu.
- Triangulasi kebijakan (BTT, status tanggap darurat).

## 3. Hasil
- Grup COVID 99,3 M (19,7% total).
- Spike Delta x5,85; handscoon x43,9; rapid test x25,4; ADS x10,0.
- Airvo 762,5 jt (1 invoice).
- Karhutla 2023 NEGATIF (masker tidak melonjak; ratio 0,47 vs baseline).

## 4. Diskusi
- Anatomi spike pengadaan darurat.
- Mengapa model riwayat gagal menangkap event.

## 5. Kesimpulan
- Spike darurat terkonsentrasi waktu & produk; perlu penanganan event-driven.

## Catatan pengembangan
- Level produk & event, bukan panel region-hari.
- Angka kunci: 99,3 M / 19,7%; x5,85 / x43,9 / x25,4 / x10,0; Airvo 762,5 jt; karhutla ratio 0,47.

---

# Outline #6 — Siklus Fiskal Pengadaan, Bukan Wabah

**Judul kerja:** *Procurement Cycles, Not Outbreaks: Temporal Patterns of Syringe Purchasing in South Sumatra*

**Sasaran:** jurnal public health (research note) atau vaccine/immunization program bulletin.

## Ringkasan satu paragraf (untuk abstrak)

Kami menguji asumsi umum "wabah = lonjakan alkes" pada pengadaan syringe di Sumatera Selatan. Puncak syringe berulang Juli dan November (pola siklus fiskal); November 2023 (masa endemi) justru tertinggi 242.303 unit. Uji DBD pooled r = 0,058 (p = 0,639) dan within r = 0,001; uji COVID Mann-Whitney p = 0,824. Konteks Muara Enim (pembeli terbesar) = program imunisasi rutin (pertusis/campak akibat cakupan imunisasi turun). Kesimpulan: kalender fiskal lebih menentukan daripada epidemiologi.

## 1. Pendahuluan
- Latar: asumsi wabah mendorong pengadaan alkes.
- Pertanyaan riset: apakah pengadaan syringe mengikuti wabah atau siklus fiskal?

## 2. Data dan Metode
### 2.1 Data
- Deret waktu bulanan produk tunggal (syringe), qty dinormalisasi satuan (spuit Box -> pcs, isi 100/box; produk khusus di-exclude).

### 2.2 Metode
- Uji korelasi epidemiologi (DBD, COVID).
- Ringkas dan tajam, cocok short communication.

## 3. Hasil
- Puncak berulang Juli & November (siklus fiskal).
- Nov 2023 (endemi) tertinggi 242.303.
- DBD pooled r = 0,058 p = 0,639; within r = 0,001.
- COVID Mann-Whitney p = 0,824.
- Konteks Muara Enim: imunisasi rutin.

## 4. Diskusi
- Sanggah asumsi "wabah = lonjakan".
- Kalender fiskal lebih menentukan daripada epidemiologi.

## 5. Kesimpulan
- Pengadaan syringe didorong siklus anggaran, bukan wabah.

## Catatan pengembangan
- Satu kelompok produk, pertanyaan spesifik, tulisan pendek.
- Angka kunci: Nov 2023 242.303; DBD r = 0,058/0,001; COVID p = 0,824.

---

# Outline #7 — Geocoding Pelanggan untuk Analitik Permintaan Daerah

**Judul kerja:** *Invoice-Weighted Centroids: A Practical Geocoding Protocol for Regional Demand Analytics in Address-Poor Markets*

**Sasaran:** data paper / jurnal GIS applied (atau bagian metodologi tesis).

## Ringkasan satu paragraf (untuk abstrak)

Kami menyajikan protokol geocoding replikabel untuk data master pelanggan yang buruk — masalah nyata di pasar berkembang. Dari 156 pelanggan, 118 (98,8% nilai faktur) tergeolokasi dari enam sumber publik (SIRS Kemenkes, Photon, idalamat, lewatmana, Google Maps, Medicastore); sisanya (apotek/praktiker tanpa alamat publik) dipetakan ke pusat kota. Centroid dibobot jumlah faktur. Dampak terukur pada konstruksi graf dan hasil model (coords_old vs geocoded tersedia untuk perbandingan).

## 1. Pendahuluan
- Latar: data master pelanggan buruk di pasar berkembang.
- Pertanyaan riset: bagaimana geocoding yang praktis dan replikabel untuk analitik permintaan daerah?

## 2. Data dan Metode
### 2.1 Data
- 156 pelanggan, 6 sumber publik.

### 2.2 Protokol
- Geocoding aktual 118/156 (98,8% faktur = 10.752/10.878).
- 38 sisanya (126 faktur, 1,2%) = pusat kota.
- Centroid dibobot jumlah faktur.
- Dokumentasi keputusan merge wilayah (MRU -> Musi Rawas).

## 3. Hasil
- 118/156 tergeolokasi (98,8% nilai).
- Jarak koordinat aktual vs data asli: median 3,2 km, p90 25,6 km, max 125,7 km.
- Koreksi region salah (Empat Lawang, RSUD Sungai Lilin).
- Dampak pada graf & hasil model (coords_old vs geocoded).

## 4. Diskusi
- Protokol replikabel untuk data master buruk.
- Implikasi untuk analitik permintaan daerah.

## 5. Kesimpulan
- Protokol geocoding praktis meningkatkan kualitas input analitik.

## Catatan pengembangan
- Fokus kualitas data masukan, bukan model.
- Angka kunci: 118/156 (98,8%); median 3,2 km; p90 25,6 km; max 125,7 km.

---

# Outline #8 — Collapse Classifier pada Permintaan Zero-Inflated

**Judul kerja:** *The Zero-Inflated Classifier Collapse: When Majority-Class Dominance Silences Sparse Regions in Demand Forecasting*

**Sasaran:** jurnal forecasting / applied ML (International Journal of Forecasting applied track, Expert Systems with Applications) atau bagian revisi naskah utama.

## Ringkasan satu paragraf (untuk abstrak)

Dengan 84,85% target test bernilai nol, binary cross-entropy didominasi kelas mayoritas sehingga classifier tidak pernah output p > 0,5 untuk 15/16 region di seluruh 231 hari test (hanya Palembang yang ter-gate ke non-zero). Per-region R2 negatif untuk SEMUA region termasuk Palembang (-0,3336); model under-predict total penjualan test sebesar 78%; RMSE 8,16M lebih buruk dari baseline train_mean 7,66M. R2 pooled 0,0523 didorong prediksi nol yang benar (3.136 dari 3.696 region-day). Weighted BCE memperbaiki collapse tapi over-predict (R2 -0,71). Kesimpulan: keunggulan pooled R2 bisa menutupi kegagalan per-region; klasifikasi "apakah ada order" adalah sinyal yang lebih andal daripada jumlah.

## 1. Pendahuluan
- Latar: forecasting permintaan sparse dengan ZI.
- Kesenjangan: laporan R2 pooled tanpa dekomposisi per-unit.
- Pertanyaan riset: bagaimana class imbalance mempengaruhi classifier ZI?

## 2. Data dan Metode
### 2.1 Data
- 16 region, 231 hari test, 84,85% nol.

### 2.2 Metode
- Diagnostik dekomposisi kontribusi (klasifikasi vs regresi).
- Per-region R2, threshold sweep.
- Eksperimen mitigasi (weighted BCE, fullreg, tanpa gate).

## 3. Hasil
- Classifier collapse ke Palembang (15/16 region di-gate ke nol).
- Per-region R2 negatif semua region (Palembang -0,3336).
- Under-predict 78%; RMSE 8,16M vs baseline 7,66M.
- R2 pooled 0,0523 didorong zero classification (3.136/3.696).
- Weighted BCE (pw 4,1) memperbaiki collapse tapi over-predict (R2 -0,71).
- Threshold tuning tidak membantu; regression tanpa ZI gate catastrophe (R2 -60).

## 4. Diskusi
- Peringatan metodologis jebakan class imbalance.
- Klasifikasi "apakah ada order" lebih andal daripada jumlah.
- Temuan negatif fullreg (kontradiktif di seluruh pipeline).

## 5. Kesimpulan
- R2 pooled bisa menutupi kegagalan per-region; perlu dekomposisi per-unit.

## Catatan pengembangan
- Objek kajian: perilaku model di bawah class imbalance (diagnostik + mitigasi), bukan klaim performa.
- Melengkapi #2 dari sudut klasifikasi.
- Angka kunci: 84,85%; 15/16 region; -0,3336; 78%; 8,16M vs 7,66M; 3.136/3.696; -0,71; -60.

---

# Outline #9 — Injeksi Atribut Node Statis ke GNN untuk Data Sparse

**Judul kerja:** *Injecting Static Regional Attributes into Graph Neural Networks for Sparse Zero-Inflated Spatiotemporal Demand Forecasting*

**Sasaran:** jurnal IT/applied ML — IEEE Access, Applied Intelligence, Expert Systems with Applications, Neurocomputing, atau jurnal informatika nasional terakreditasi (SINTA 1-2).

## Ringkasan satu paragraf (untuk abstrak)

Kami mengkaji bagaimana menginjeksi atribut node statis ke GNN untuk forecasting spatiotemporal data sparse. Menambahkan jumlah RS per kab/kota sebagai atribut node GCN memperbaiki model secara signifikan pada kedua konfigurasi: base R2 0,0609 vs 0,0592 (DM = -2,861, p = 0,004); tuned R2 0,0528 vs 0,0508 (DM = -6,037, p < 0,0001). Luas daerah tidak menambah; rolling mean/std tidak membantu (DM p = 0,763). Atribut statis tidak bisa masuk sebagai kanal input temporal biasa (konstan dalam window), tapi efektif sebagai atribut node GCN.

## 1. Pendahuluan
- Latar: GNN forecasting spatiotemporal; atribut node statis.
- Kesenjangan: bagaimana menginjeksi atribut hampir-konstan tanpa overfitting.
- Pertanyaan riset: apakah atribut node statis memperbaiki forecasting data sparse?

## 2. Data dan Metode
### 2.1 Data
- 16 region Sumsel, 2020-2025, target nilai faktur per region per hari.

### 2.2 Metode
- Studi ablasi sistematis injeksi atribut node (jumlah RS, luas, rolling stats).
- Konfigurasi base vs tuned, uji Diebold-Mariano, multi-seed.
- Bisa diperluas ke dataset publik sparse untuk generalisasi.

## 3. Hasil
- Base: R2 0,0609 vs 0,0592; DM = -2,861, p = 0,004.
- Tuned: R2 0,0528 vs 0,0508; DM = -6,037, p < 0,0001.
- Luas daerah tidak menambah (R2 0,0609 = W2).
- Rolling mean/std tidak membantu (DM p = 0,763).
- Mekanisme: jumlah RS = proxy kapasitas kesehatan (Palembang 33 RS = 44%).

## 4. Diskusi
- Interpretasi: atribut statis efektif sebagai atribut node GCN, bukan kanal temporal.
- Implikasi arsitektur GNN untuk data sparse.

## 5. Kesimpulan
- Injeksi atribut node statis memperbaiki forecasting data sparse secara signifikan.

## Catatan pengembangan
- Pemisah dari #4: #4 tanpa ML (kebijakan); #9 metodologi GNN (computer science).
- Pemisah dari naskah utama: #9 menjadikan injeksi atribut node sebagai objek kajian.
- Angka kunci: base DM p = 0,004; tuned DM p < 0,0001; luas tidak menambah; rolling p = 0,763.
