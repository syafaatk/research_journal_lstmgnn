# Usulan Judul Jurnal Baru (dari Temuan Analisis)

Dibuat: 26 Agustus 2026. Diperbarui: 28 Agustus 2026 (menambahkan temuan diagnostik ZI collapse, atribut node RS, mekanisme pembayaran lag 30 hari, dan catatan hasil fullreg). Semua angka merujuk pada `temuan.md`, `daftar_tbd.md`, dan hasil eksperimen terverifikasi. Tujuan: memetakan temuan yang cukup kuat menjadi paper lanjutan, dengan posisi yang jelas terhadap naskah utama (LSTM-GNN Sumsel) agar tidak termasuk salami slicing.

---

## 1. Topologi Jaringan Distribusi vs Kedekatan Geografis

**Judul kerja:** *When Geography Lies: Distribution Network Topology Outperforms Spatial Proximity in Graph-Based Regional Demand Forecasting*

- **Temuan inti:** graf bintang distribusi (Palembang ke semua region) R2 0,0703 = terbaik di semua varian ablasi (+61% relatif atas graf jarak 0,0436); graf jarak lebih buruk dari adjacency identitas (0,0538) dan graf acak (0,0486). Korelasi penjualan antar region maksimal 0,332 -> struktur spasial "alamiah" tipis.
- **Kontribusi:** panduan praktis pemilihan graf untuk GNN forecasting pada bisnis distribusi: tanya dulu bagaimana barang benar-benar bergerak, bukan di mana lokasinya.
- **Metode:** studi ablasi sistematis lintas konstruksi graf (jarak, korelasi, acak, identitas, bintang, jaringan aktual) pada satu domain aplikasi + satu domain publik untuk generalisasi (mis. dataset transportasi/publik).
- **Sasaran:** jurnal logistics/supply chain analytics atau applied AI.
- **Pemisah dari naskah utama:** naskah utama melaporkan model; paper ini menjadikan PILIHAN GRAF sebagai objek kajian, dengan eksperimen yang dirancang untuk itu.

## 2. Granularitas Agregasi dan Struktur Nol

**Judul kerja:** *Aggregation Destroys the Signal: Zero-Inflated Daily Forecasting of Sparse Regional Demand*

- **Temuan inti:** panel bulanan R2 -0,394; mingguan -0,343; harian zero-inflated +0,043 s.d. +0,052. Fraksi nol 81,54% (panel penuh), 84,85% (test). ARIMA dan XGBoost tidak mengungguli always-zero.
- **Kontribusi:** bukti empiris bahwa pilihan resolusi waktu + formulasi zero-inflated menentukan naik-tidaknya model, lebih besar daripada perbedaan antar arsitektur (rentang gain arsitektur hanya 0,005-0,01).
- **Metode:** eksperimen granularitas terkontrol (bulanan/mingguan/harian) x beberapa keluarga model, dekomposisi kontribusi klasifikasi vs regresi jumlah.
- **Sasaran:** jurnal forecasting (mis. International Journal of Forecasting, applied track).
- **Pemisah:** pertanyaannya "pada resolusi apa permintaan ini bisa diprediksi", bukan "model apa terbaik".

## 3. Ritme Institusional Pengadaan (Efek Hari)

**Judul kerja:** *The Institutional Rhythm of Healthcare Procurement: Day-of-Week Patterns in Medical Device Ordering*

- **Temuan inti:** day-of-week one-hot + libur + fase pandemi menaikkan ensemble R2 dari 0,0523 ke 0,0612 (DM p < 0,001); residual Senin +22,1 jt (over-prediksi hari kerja), under-prediksi akhir pekan; V1 (weekend biner) tidak signifikan (p = 0,082) -> encoding penuh hari diperlukan.
- **Kontribusi:** karakterisasi perilaku pemesanan instansi kesehatan; implikasi praktis penjadwalan pengiriman dan staf gudang.
- **Metode:** bisa tanpa deep learning - regresi dengan efek tetap hari/libur pada level faktur dan region; robust, mudah direplikasi, menarik untuk pembaca operasional/manajemen.
- **Sasaran:** jurnal healthcare operations / health services management.
- **Pemisah:** unit analisis faktur (bukan panel harian agregat); pertanyaan perilaku, bukan prediksi.

## 4. Infrastruktur, Bukan Anggaran

**Judul kerja:** *Infrastructure, Not Budgets: Cross-Sectional and Temporal Determinants of Regional Medical Device Demand*

- **Temuan inti (empat lapis bukti):** (1) jumlah RS berkorelasi dengan penjualan 2024 (r = 0,669, p = 0,005) dan dengan residual model 2025 (r = 0,697, p = 0,003), bertahan tanpa Palembang (r = 0,596) dan mengontrol penduduk (partial r = 0,648); (2) tempat tidur (r = 0,577) dan pasien keluar (r = 0,522) senada; (3) **atribut node GCN jumlah RS memperbaiki model secara signifikan** pada kedua konfigurasi (base: R2 0,0609 vs 0,0592, DM p = 0,004; tuned: R2 0,0528 vs 0,0508, DM p < 0,0001) — bukti prediktif, bukan hanya korelasional; (4) sebaliknya belanja barang/jasa hanya efek skala antar-region (pooled r = 0,433 signifikan; within-region r = -0,112/0,103 tidak signifikan; APBD 2025 DJPK r = 0,256 p = 0,338). Anggaran Dinkes provinsi berlawanan arah dengan penjualan 2022-2024 karena mekanisme pencatatan pusat.
- **Kontribusi:** bagi kebijakan desentralisasi: alokasi alkes mengikuti kapasitas fasilitas, bukan siklus anggaran tahunan; peringatan metodologis tentang jebakan korelasi pooled lintas region.
- **Metode:** ekonometrika terapan (korelasi pooled vs within, partial correlation, verifikasi APBD per pemda).
- **Sasaran:** jurnal health policy/economics (Health Policy and Planning, BMC Health Services Research).
- **Pemisah:** tanpa machine learning; nilai analisis kebijakan berdiri sendiri.

## 5. Permintaan Event-Driven Masa Pandemi

**Judul kerka:** *Event-Driven Demand Spikes in Healthcare Supply Chains: Evidence from Indonesia's COVID-19 Emergency Procurement*

- **Temuan inti:** grup produk COVID 99,3 M IDR (19,7% total); spike Delta Jul-Agu 2021 x5,85 median bulanan (handscoon x43,9, rapit tes x25,4, auto-destruct syringe x10,0); satu invoice Airvo 762,5 jt; GeNose/APD; 2021 = puncak COVID + puncak BTT + puncak penjualan Rp13,03 M. Karhutla El Nino 2023 NEGATIF (masker tidak melonjak — diverifikasi dengan normalisasi satuan Box/Pcs ke pcs, lihat `temuan.md` 6.2).
- **Kontribusi:** anatomi spike pengadaan darurat: apa yang melonjak bersamaan, seberapa terkonsentrasi waktunya, dan mengapa sinyal ini tak tertangkap model berbasis riwayat.
- **Metode:** analisis temporal produk-level (spike factor vs median), pemetaan produk-pemicu, triangulasi kebijakan (BTT, status tanggap darurat).
- **Sasaran:** jurnal disaster/public health preparedness atau supply chain risk management.
- **Pemisah:** level produk dan event, bukan panel region-hari; periode terbatas 2020-2022.

## 6. Siklus Fiskal Pengadaan, Bukan Wabah

**Judul kerja:** *Procurement Cycles, Not Outbreaks: Temporal Patterns of Syringe Purchasing in South Sumatra*

- **Temuan inti:** puncak syringe berulang Juli dan November (siklus fiskal); Nov 2023 (endemi) justru tertinggi 242.303; uji DBD pooled r = 0,058 p = 0,639, within r = 0,001; uji COVID Mann-Whitney p = 0,824; konteks Muara Enim = program imunisasi rutin (pertusis/campak akibat cakupan imunisasi turun). Qty syringe sudah dinormalisasi satuan (spuit Box dikonversi ke pcs, isi 100 pcs/box; produk khusus pre-filled/insulin di-exclude — lihat `temuan.md` 6.4), sehingga angka qty valid.
- **Kontribusi:** sanggah asumsi umum "wabah = lonjakan alkes"; bukti bahwa kalender fiskal lebih menentukan daripada epidemiologi.
- **Metode:** deret waktu bulanan produk tunggal + uji korelasi epidemiologi; ringkas dan tajam, cocok short communication/research note.
- **Sasaran:** jurnal public health (research note) atau vaccine/immunization program bulletin.
- **Pemisah:** satu kelompok produk, pertanyaan spesifik, bisa jadi tulisan pendek.

## 7. Geocoding Pelanggan untuk Analitik Permintaan Daerah

**Judul kerja:** *Invoice-Weighted Centroids: A Practical Geocoding Protocol for Regional Demand Analytics in Address-Poor Markets*

- **Temuan inti:** 118/156 pelanggan (98,8% nilai faktur) tergeolokasi dari enam sumber publik; sisanya apotek/praktiker tanpa alamat publik dipetakan ke pusat kota; centroid dibobot jumlah faktur; dampaknya terukur pada konstruksi graf dan hasil model (coords_old vs geocoded tersedia untuk perbandingan).
- **Kontribusi:** protokol replikabel untuk data master pelanggan yang buruk - masalah nyata di pasar berkembang; plus dokumentasi keputusan merge wilayah administratif (MRU).
- **Metode:** deskriptif-metodologis + sensitivitas hasil model terhadap strategi geocoding.
- **Sasaran:** data paper / jurnal GIS applied (atau bagian metodologi tesis).
- **Pemisah:** fokus kualitas data masukan, bukan model.

---

## 8. Collapse Classifier pada Permintaan Zero-Inflated

**Judul kerja:** *The Zero-Inflated Classifier Collapse: When Majority-Class Dominance Silences Sparse Regions in Demand Forecasting*

- **Temuan inti:** dengan 84,85% target test bernilai nol, binary cross-entropy didominasi kelas mayoritas sehingga classifier **tidak pernah** output p > 0,5 untuk 15/16 region di seluruh 231 hari test (hanya Palembang yang ter-gate ke non-zero). Per-region R2 negatif untuk SEMUA region termasuk Palembang (-0,3336); model under-predict total penjualan test sebesar 78%; RMSE 8,16M lebih buruk dari baseline train_mean 7,66M. R2 pooled 0,0523 didorong prediksi nol yang benar (3.136 dari 3.696 region-day). Weighted BCE (pos_weight 4,1) memperbaiki collapse tapi over-predict (R2 -0,71); threshold tuning tidak membantu; regression tanpa ZI gate catastrophe (R2 -60).
- **Kontribusi:** peringatan metodologis yang jujur tentang jebakan class imbalance pada forecasting permintaan sparse: keunggulan pooled R2 bisa menutupi kegagalan per-region; klasifikasi "apakah ada order" adalah sinyal yang lebih andal daripada jumlah. Relevan untuk komunitas forecasting & applied ML yang sering melaporkan R2 pooled tanpa dekomposisi per-unit.
- **Metode:** diagnostik dekomposisi kontribusi (klasifikasi vs regresi), per-region R2, threshold sweep, eksperimen mitigasi (weighted BCE, fullreg, tanpa gate). Bisa diperluas dengan simulasi/benchmark pada dataset publik sparse.
- **Sasaran:** jurnal forecasting / applied ML (International Journal of Forecasting applied track, Expert Systems with Applications) atau sebagai bagian revisi naskah utama.
- **Pemisah:** objek kajiannya adalah **perilaku model di bawah class imbalance** (diagnostik + mitigasi), bukan klaim performa model baru; melengkapi #2 (granularitas & struktur nol) dari sudut klasifikasi.
- **Catatan (28 Agu 2026):** eksperimen fullreg (train pada semua hari) yang awalnya tampak menjanjikan (R2 0,0367 -> 0,0494) ternyata **kontradiktif di seluruh pipeline** (Hybrid base turun, star graph terbalik, V3 kolaps) sehingga di-revert ke masked regression. Ini memperkuat bahwa perbaikan ZI bukan sekadar mengganti loss, dan layak dibahas sebagai temuan negatif yang jujur.

---

## 9. Injeksi Atribut Node Statis ke GNN untuk Data Sparse (Sasaran Jurnal IT)

**Judul kerja:** *Injecting Static Regional Attributes into Graph Neural Networks for Sparse Zero-Inflated Spatiotemporal Demand Forecasting*

- **Temuan inti:** menambahkan atribut node statis (jumlah RS per kab/kota) ke jalur GCN memperbaiki model secara signifikan pada kedua konfigurasi: base R2 0,0609 vs 0,0592 (DM = -2,861, p = 0,004); tuned R2 0,0528 vs 0,0508 (DM = -6,037, p < 0,0001). Luas daerah tidak menambah (R2 0,0609 = W2); rolling mean/std 7 & 30 hari tidak membantu (DM p = 0,763). Mekanisme: jumlah RS = proxy kapasitas kesehatan regional (Palembang 33 RS = 44% provinsi, lainnya 1-8) yang mempengaruhi pola pengadaan; masuk via embedding node GCN.
- **Kontribusi (metodologis, untuk audiens IT):** bagaimana menginjeksi atribut statis/hampir-konstan (yang tidak berubah dalam window 30 hari) ke model temporal-spasial tanpa overfitting — atribut statis tidak bisa masuk sebagai kanal input temporal biasa (konstan dalam window), tapi efektif sebagai atribut node pada jalur GCN. Ini pertanyaan arsitektur GNN yang relevan untuk forecasting spatiotemporal data sparse.
- **Metode:** studi ablasi sistematis injeksi atribut node (jumlah RS, luas, rolling stats) pada konfigurasi base vs tuned, dengan uji Diebold-Mariano dan multi-seed; bisa diperluas ke dataset publik sparse untuk generalisasi.
- **Sasaran:** jurnal IT/applied ML — IEEE Access, Applied Intelligence, Expert Systems with Applications, Neurocomputing, atau jurnal informatika nasional terakreditasi (SINTA 1-2) untuk konteks Indonesia.
- **Pemisah dari #4:** #4 adalah analisis kebijakan TANPA machine learning (korelasi, partial correlation, verifikasi APBD) untuk audiens health policy. #9 adalah paper METODOLOGI GNN (dengan model, ablasi, uji statistik) untuk audiens computer science. Keduanya memakai temuan jumlah RS yang sama tapi menanyakan hal berbeda: #4 "apakah infrastruktur menentukan permintaan" (kebijakan), #9 "bagaimana menginjeksi atribut statis ke GNN untuk data sparse" (metode).
- **Pemisah dari naskah utama:** naskah utama melaporkan model LSTM-GNN zero-inflated secara keseluruhan; #9 menjadikan INJEKSI ATRIBUT NODE sebagai objek kajian metodologis tersendiri, dengan eksperimen yang dirancang untuk itu (bukan sekadar melaporkan satu hasil).
- **Catatan (28 Agu 2026):** ini mengisi celah sasaran jurnal IT yang belum terwakili — #1 (applied AI) dan #8 (ESWA) sudah IT, tapi #9 menawarkan sudut arsitektur GNN yang berbeda dari keduanya. Opsi tambahan yang lebih tipis untuk IT (data engineering): normalisasi satuan katalog produk alkes (Box/Pcs/Roll/Pack ambigu, kadang isi box sama kadang produk berbeda) — relevan untuk jurnal data quality, tapi perlu diperluas agar cukup kuat sebagai paper tersendiri.

---

## Catatan Strategis

1. **Prioritas saya:** #4 (bukti terkuat, kini empat lapis termasuk bukti prediktif atribut node, kebijakan relevan, tanpa ML sehingga beda audiens) > #1 (temuan paling orisinal secara metodologis) > #8 (temuan baru, jujur, relevan untuk komunitas forecasting) > #9 (metodologi GNN untuk jurnal IT, bukti prediktif kuat) > #3 (mudah dieksekusi, cepat selesai).
2. **Hindari salami slicing:** setiap paper harus punya pertanyaan, unit analisis, dan metode berbeda dari naskah utama; jangan menduplikasi Tabel 3-8 sebagai inti klaim. #8 beririsan dengan #2 dan naskah utama, jadi harus dibingkai sebagai diagnostik/mitigasi, bukan klaim performa.
3. **Hasil negatif bernilai:** karhutla/DBD tidak berkorelasi, graf jarak kalah dari identitas, dan collapse classifier ZI adalah temuan yang layak dilaporkan secara jujur - framing "what does NOT drive demand" dan "kapan model gagal" adalah sudut yang jarang dipakai.
4. **Periksa kebijakan jurnal** soal co-submission dan penggunaan dataset yang sama; cantumkan cross-reference antar paper.
5. Eksperimen atribut node GCN (jumlah RS) SELESAI dan signifikan (base DM p = 0,004; tuned DM p < 0,0001) - memperkuat #4 sebagai bukti prediktif. Rolling window node feature tidak membantu (DM p = 0,763).
6. **Normalisasi satuan produk kunci (28 Agu 2026) memperkuat validitas analisis produk** pada #5 dan #6: qty masker/spuit sudah dikonversi ke satuan dasar (pcs), dan kassa/plester dianalisis per produk karena Roll/Box/Pack adalah produk berbeda. Ini bukan objek kajian paper tersendiri (terlalu tipis), tapi menaikkan keandalan klaim berbasis qty produk. Detail di `temuan.md` 6.4 dan `checklist_anomali_data.md`.
7. **Sasaran jurnal IT:** #1 (applied AI), #8 (ESWA), dan #9 (IEEE Access/Applied Intelligence/Neurocomputing/SINTA) adalah kandidat untuk jurnal informatika. #9 sengaja dibingkai sebagai metodologi GNN (bukan kebijakan) agar berbeda dari #4 dan mengisi celah audiens computer science. Untuk konteks Indonesia, #9 dan #8 juga cocok untuk jurnal nasional terakreditasi SINTA 1-2.
