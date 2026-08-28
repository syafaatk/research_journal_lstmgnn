# Outline Paper #4 — Infrastructure, Not Budgets

**Judul kerja:** *Infrastructure, Not Budgets: Cross-Sectional and Temporal Determinants of Regional Medical Device Demand*

**Sasaran:** Health Policy and Planning / BMC Health Services Research (health policy & economics)

**Status:** Outline draf (28 Agu 2026). Semua angka terverifikasi dari `temuan.md` 2.4-2.6, 3.7 dan `balasan_reviewer.md`. Belum ditulis naskah penuh.

---

## Ringkasan satu paragraf (untuk abstrak)

Menggunakan data transaksi satu distributor alat kesehatan di Sumatera Selatan (2020-2025, 13.661 faktur, 16 kab/kota), kami menguji apakah permintaan alat kesehatan regional ditentukan oleh infrastruktur layanan kesehatan atau oleh anggaran daerah. Empat lapis bukti konsisten: (1) jumlah rumah sakit berkorelasi dengan penjualan 2024 (r = 0,669, p = 0,005) dan dengan residual model prediktif 2025 (r = 0,697, p = 0,003), bertahan tanpa Palembang (r = 0,596) dan mengontrol populasi (partial r = 0,648); (2) kapasitas tempat tidur (r = 0,577) dan volume pasien (r = 0,522) senada; (3) memasukkan jumlah RS sebagai atribut node pada model graf temporal memperbaiki prediksi secara signifikan (DM p < 0,0001) — bukti prediktif, bukan hanya korelasional; (4) sebaliknya, belanja barang/jasa daerah hanya efek skala antar-region (pooled r = 0,433 signifikan; within-region tidak signifikan; verifikasi APBD resmi per pemda r = 0,256, p = 0,338). Kesimpulan: alokasi alat kesehatan mengikuti kapasitas fasilitas, bukan siklus anggaran tahunan — dengan peringatan metodologis tentang jebakan korelasi pooled lintas region.

---

## 1. Pendahuluan

- **Latar:** desentralisasi kesehatan Indonesia; pengadaan alkes dilakukan oleh RS dan Dinkes; pertanyaan siapa yang menentukan volume pengadaan.
- **Kesenjangan:** literatur umumnya mengaitkan belanja kesehatan dengan anggaran daerah, tapi jarang menguji dengan data transaksi aktual level distributor.
- **Pertanyaan riset:** apakah permintaan alkes regional ditentukan oleh (a) infrastruktur layanan kesehatan (jumlah RS, tempat tidur, pasien) atau (b) anggaran umum daerah (belanja barang/jasa)?
- **Kontribusi:** (1) bukti empiris pertama dari data transaksi distributor; (2) peringatan metodologis jebakan korelasi pooled; (3) implikasi kebijakan desentralisasi.

## 2. Data dan Metode

### 2.1 Data
- Transaksi distributor alkes Sumsel 2020-2025: 13.661 faktur unik, 16 kab/kota, 230 pelanggan (70,8% RS).
- Direktori RS Kemenkes 2024: 88 RS, 10.161 tempat tidur, dipetakan ke 16 region (8 nama ambigu diverifikasi).
- APBD per pemda 2020-2025 (BPS + portal DJPK Kemenkeu, fokus Belanja Barang & Jasa).

### 2.2 Variabel
- **Outcome:** penjualan tahunan per region (2024) dan residual model prediktif (2025).
- **Prediktor infrastruktur:** jumlah RS, total tempat tidur, total pasien keluar, hari perawatan.
- **Prediktor anggaran:** belanja barang/jasa per pemda (total dan per kapita).

### 2.3 Analisis
- Korelasi cross-sectional (Spearman/Pearson) per region.
- Korelasi pooled vs within-region (untuk memisahkan efek skala dari efek temporal).
- Korelasi parsial mengontrol populasi.
- Sensitivitas: tanpa Palembang (n=15).
- **Bukti prediktif:** injeksi jumlah RS sebagai atribut node GCN pada model LSTM-GNN zero-inflated; uji Diebold-Mariano vs baseline tanpa atribut (base DM p = 0,004; tuned DM p < 0,0001).
- **Koreksi multiple testing** karena banyak faktor diuji (n=16 kecil).

## 3. Hasil

### 3.1 Infrastruktur berkorelasi dengan penjualan
- Jumlah RS vs penjualan 2024: r = 0,669, p = 0,005.
- Tempat tidur r = 0,577; pasien keluar r = 0,522.
- Bertahan tanpa Palembang (r = 0,596, p = 0,019) dan mengontrol populasi (partial r = 0,648, p = 0,043).

### 3.2 Infrastruktur berkorelasi dengan residual model
- Jumlah RS vs residual 2025: r = 0,697, p = 0,003; tanpa Palembang r = 0,631, p = 0,012.
- Menunjukkan infrastruktur menjelaskan variasi yang TIDAK ditangkap model berbasis riwayat.

### 3.3 Bukti prediktif: atribut node GCN
- Base: R2 0,0609 vs 0,0592; DM = -2,861, p = 0,004.
- Tuned: R2 0,0528 vs 0,0508; DM = -6,037, p < 0,0001.
- Palembang RMSE turun 1,1% (base).
- Mekanisme: Palembang 33 RS (44% provinsi), lainnya 1-8 RS; jumlah RS = proxy kapasitas kesehatan.

### 3.4 Anggaran BUKAN prediktor
- Pooled r = 0,433 signifikan (efek skala antar-region).
- Within-region r = -0,112 / 0,103 TIDAK signifikan.
- APBD 2025 DJPK per pemda r = 0,256, p = 0,338; per kapita r = 0,487, p = 0,056.
- Anggaran Dinkes provinsi berlawanan arah dengan penjualan 2022-2024 (mekanisme pencatatan pusat).

## 4. Diskusi

- **Interpretasi:** RS = pembeli alkes terbesar; lebih banyak RS/pasien = lebih banyak pengadaan. Efisiensi operasional (TOI/ALOS) tidak relevan; yang penting keberadaan dan skala layanan.
- **Implikasi kebijakan:** alokasi alkes mengikuti kapasitas fasilitas, bukan siklus anggaran tahunan; desentralisasi pengadaan sebaiknya berbasis kebutuhan infrastruktur.
- **Peringatan metodologis:** korelasi pooled lintas region menyesatkan (efek skala); analisis harus memisahkan within vs between.
- **Keterbatasan:** data 1 tahun (2024) untuk bukti struktural; jumlah RS hampir statis (tidak menjelaskan variasi harian); n=16 kecil, risiko false positive; data satu distributor.

## 5. Kesimpulan

- Permintaan alkes regional didorong struktur layanan kesehatan, bukan anggaran umum daerah.
- Implikasi untuk perencanaan pengadaan dan evaluasi desentralisasi.

---

## Catatan pengembangan

- **Perlu ditambahkan sebelum submit:** (1) tabel ringkasan statistik deskriptif per region; (2) plot korelasi (scatter jumlah RS vs penjualan, dengan label region); (3) tabel hasil DM untuk atribut node; (4) penjelasan mekanisme pencatatan anggaran Dinkes (belanja modal vs barang/jasa).
- **Angka kunci untuk verifikasi ulang:** r = 0,669 / 0,697 / 0,596 / 0,648 / 0,577 / 0,522; DM p = 0,004 / <0,0001; pooled r = 0,433; within-region r = -0,112 / 0,103; APBD 2025 r = 0,256.
- **Pemisah dari #9:** #4 tanpa ML (kebijakan); #9 metodologi GNN (computer science). Jangan mencampur keduanya dalam satu naskah.
