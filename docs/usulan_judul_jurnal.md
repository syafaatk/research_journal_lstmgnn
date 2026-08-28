# Usulan Judul Jurnal Baru (dari Temuan Analisis)

Dibuat: 26 Agustus 2026. Semua angka merujuk pada `temuan.md`, `daftar_tbd.md`, dan hasil eksperimen terverifikasi. Tujuan: memetakan temuan yang cukup kuat menjadi paper lanjutan, dengan posisi yang jelas terhadap naskah utama (LSTM-GNN Sumsel) agar tidak termasuk salami slicing.

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

- **Temuan inti (tiga lapis bukti):** jumlah RS berkorelasi dengan penjualan 2024 (r = 0,669, p = 0,005) dan dengan residual model 2025 (r = 0,697, p = 0,003), bertahan tanpa Palembang (r = 0,596) dan mengontrol penduduk (partial r = 0,648); tempat tidur (r = 0,577) dan pasien keluar (r = 0,522) senada. Sebaliknya belanja barang/jasa hanya efek skala antar-region (pooled r = 0,433 signifikan; within-region r = -0,112/0,103 tidak signifikan; APBD 2025 DJPK r = 0,256 p = 0,338). Anggaran Dinkes provinsi berlawanan arah dengan penjualan 2022-2024 karena mekanisme pencatatan pusat.
- **Kontribusi:** bagi kebijakan desentralisasi: alokasi alkes mengikuti kapasitas fasilitas, bukan siklus anggaran tahunan; peringatan metodologis tentang jebakan korelasi pooled lintas region.
- **Metode:** ekonometrika terapan (korelasi pooled vs within, partial correlation, verifikasi APBD per pemda).
- **Sasaran:** jurnal health policy/economics (Health Policy and Planning, BMC Health Services Research).
- **Pemisah:** tanpa machine learning; nilai analisis kebijakan berdiri sendiri.

## 5. Permintaan Event-Driven Masa Pandemi

**Judul kerka:** *Event-Driven Demand Spikes in Healthcare Supply Chains: Evidence from Indonesia's COVID-19 Emergency Procurement*

- **Temuan inti:** grup produk COVID 99,3 M IDR (19,7% total); spike Delta Jul-Agu 2021 x5,85 median bulanan (handscoon x43,9, rapit tes x25,4, auto-destruct syringe x10,0); satu invoice Airvo 762,5 jt; GeNose/APD; 2021 = puncak COVID + puncak BTT + puncak penjualan Rp13,03 M. Karhutla El Nino 2023 NEGATIF (masker tidak melonjak).
- **Kontribusi:** anatomi spike pengadaan darurat: apa yang melonjak bersamaan, seberapa terkonsentrasi waktunya, dan mengapa sinyal ini tak tertangkap model berbasis riwayat.
- **Metode:** analisis temporal produk-level (spike factor vs median), pemetaan produk-pemicu, triangulasi kebijakan (BTT, status tanggap darurat).
- **Sasaran:** jurnal disaster/public health preparedness atau supply chain risk management.
- **Pemisah:** level produk dan event, bukan panel region-hari; periode terbatas 2020-2022.

## 6. Siklus Fiskal Pengadaan, Bukan Wabah

**Judul kerja:** *Procurement Cycles, Not Outbreaks: Temporal Patterns of Syringe Purchasing in South Sumatra*

- **Temuan inti:** puncak syringe berulang Juli dan November (siklus fiskal); Nov 2023 (endemi) justru tertinggi 242.303; uji DBD pooled r = 0,058 p = 0,639, within r = 0,001; uji COVID Mann-Whitney p = 0,824; konteks Muara Enim = program imunisasi rutin (pertusis/campak akibat cakupan imunisasi turun).
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

## Catatan Strategis

1. **Prioritas saya:** #4 (bukti terkuat, tiga lapis, kebijakan relevan, tanpa ML sehingga beda audiens) > #1 (temuan paling orisinal secara metodologis) > #3 (mudah dieksekusi, cepat selesai).
2. **Hindari salami slicing:** setiap paper harus punya pertanyaan, unit analisis, dan metode berbeda dari naskah utama; jangan menduplikasi Tabel 3-8 sebagai inti klaim.
3. **Hasil negatif bernilai:** karhutla/DBD tidak berkorelasi dan graf jarak kalah dari identitas adalah temuan yang layak dilaporkan secara jujur - framing "what does NOT drive demand" adalah sudut yang jarang dipakai.
4. **Periksa kebijakan jurnal** soal co-submission dan penggunaan dataset yang sama; cantumkan cross-reference antar paper.
5. Eksperimen rolling + atribut node yang sedang berjalan (F6 daftar_tbd) jika berhasil menaikkan R2 > 0,01 bisa menjadi dasar tambahan untuk #1 atau masuk revisi naskah utama.
