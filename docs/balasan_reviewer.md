# Balasan untuk Editor / Reviewer

> **PENTING SEBELUM DIKIRIM:** Surat ini adalah template final. Seluruh angka di dalamnya berasal dari eksperimen ulang pada `view_penjualan_detail data hingga oktober.xlsx` (protokol harian zero-inflated dengan fitur COVID-19). Bagian yang masih bertanda `[ISI: ...]` hanya memerlukan konfirmasi penulis (bukan eksperimen). Jangan kirim surat ini sebelum seluruh `[ISI: ...]` terisi dan sebelum naskah revisi final selesai dilengkapi. Setiap klaim di surat ini harus sesuai dengan isi naskah yang dikirim.

---

Kepada Yth. Editor dan Reviewer

Perihal: Respons atas komentar reviewer terhadap naskah berjudul "Spatiotemporal Prediction of Medical Device Sales Using Hybrid LSTM-GNN in South Sumatra"

Dengan hormat,

Kami mengucapkan terima kasih atas review yang mendalam dan konstruktif. Seluruh komentar telah kami baca dengan saksama dan menjadi dasar revisi menyeluruh pada naskah. Berikut ringkasan perubahan utama:

1. Seluruh eksperimen dijalankan ulang dari data aktual dengan protokol yang dapat direplikasi. Semua angka metrik yang tidak konsisten secara matematis dikoreksi, dan seluruh metrik dihitung pada skala asli (IDR) setelah transformasi balik.
2. Protokol metodologi diperluas: definisi target, satuan waktu (harian), pembagian data, pengendalian kebocoran data (data leakage), konstruksi graf, arsitektur model dua tahap zero-inflated, model pembanding (baseline), evaluasi statistik, studi ablasi, dan analisis residual kini dijelaskan secara eksplisit.
3. Berdasarkan evaluasi granularitas, protokol diubah dari agregasi bulanan menjadi harian dengan formulasi zero-inflated (klasifikasi ada/tidak penjualan + regresi jumlah), karena 81,54% seluruh target harian pada panel penuh bernilai nol (84,85% pada periode pengujian). Fitur indikator COVID-19 ditambahkan sebagai kanal masukan eksogen.
4. Klaim yang sebelumnya terlalu kuat telah diturunkan dan diberi syarat bukti, termasuk klaim kontribusi spasial, simetri residual, dan implikasi praktis.
5. Daftar referensi telah diperbaiki dan dilengkapi.

Berikut respons kami poin per poin. Penomoran mengikuti komentar asli reviewer.

---

**1. Inkonsistensi MAE dan RMSE**

Komentar reviewer: Nilai RMSE 0,1252 < MAE 0,1273 pada Hybrid LSTM-GNN baseline (dan Test 1 pada Tabel 3) tidak mungkin secara matematis karena RMSE >= MAE selalu berlaku untuk himpunan galat yang sama.

Respons: Kami setuju. Seluruh eksperimen dijalankan ulang dari data aktual dan semua metrik dihitung pada skala asli. Hasil akhir: Hybrid LSTM-GNN memperoleh RMSE 8.002.603 IDR dan MAE 1.881.358 IDR (MAE <= RMSE terpenuhi di semua tabel). Nilai lama pada skala transformasi (RMSE 0,0881, MAE 0,0324, MAPE 8,219%, R2 0,8595) dihapus dan diganti dengan hasil eksperimen ulang pada skala asli. Persentase penurunan galat dihitung ulang dari angka aktual dan disajikan di abstrak, bagian hasil, dan kesimpulan.

---

**3. Pembagian data tidak dapat direplikasi**

Komentar reviewer: Naskah hanya menyebut pembagian kronologis (chronological split) tanpa rasio, tanggal batas, jumlah sampel, dan detail pengendalian kebocoran data.

Respons: Bagian 3.2 kini menjelaskan secara eksplisit pembagian tiga arah (training, validasi, testing) dengan tanggal batas: data pelatihan Juli 2020 sampai Desember 2023 (957 sampel jendela), data validasi Januari sampai Desember 2024 (277 sampel), dan data pengujian Januari sampai Oktober 2025 (231 sampel). Pembagian yang sama diterapkan untuk seluruh 16 wilayah. Jendela geser dibangun atas seluruh panel kemudian dibagi secara kronologis berdasarkan tanggal target; karena pembagian bersifat kronologis, target pengujian dan validasi tidak pernah menjadi masukan pada jendela pelatihan. Transformasi log dan normalisasi hanya dilatih menggunakan set pelatihan (statistik mu dan sd dihitung pada window training saja), seed acak 42 digunakan untuk seluruh operasi stokastik, dan hari tanpa penjualan diisi nol tanpa imputasi target. Kami juga menjelaskan bahwa sekuens pengujian dan validasi boleh memakai riwayat sebelum periode masing-masing, tetapi target pengujian dan validasi tidak pernah masuk ke set pelatihan.

---

**4. Satuan waktu tidak dijelaskan**

Komentar reviewer: Jendela 60 time steps tanpa penjelasan satu time step setara dengan apa.

Respons: Bagian 3.1 kini menetapkan satu time step = satu hari. Resolusi harian dipilih setelah evaluasi granularitas: panel bulanan hanya menyediakan 64 bulan (30 sampel training) dan menghasilkan R2 negatif (-0,394), sedangkan panel harian menyediakan 1.495 hari (957 sampel training) dan memungkinkan pelatihan model sekuens yang dalam. Data transaksi diagregasi harian, hari tanpa penjualan diberi nilai nol, seluruh wilayah menggunakan kerangka waktu (timestamp) yang sama, dan dimensi masukan dijelaskan (masukan X berukuran 30 x 16 x 2, target y berukuran 16). Jendela 30 langkah mencakup satu bulan riwayat.

---

**5. Target prediksi belum tegas**

Komentar reviewer: Naskah menggunakan istilah penjualan (sales) dan permintaan (demand) secara bergantian tanpa menetapkan target secara tegas.

Respons: Target kini ditetapkan secara eksplisit sebagai total nilai faktur (jual_total_fak) per wilayah per hari. Kuantitas terjual (d_jual_qty) tidak digunakan sebagai target karena katalog produk bersifat sangat heterogen (misalnya jarum suntik dan ventilator tidak sebanding dalam satuan unit), sehingga agregasi unit lintas kategori produk tidak bermakna. Heterogenitas ini terkonfirmasi secara empiris: harga satuan implisit berkisar dari median Rp207.037 hingga maksimum Rp500.500.000, dan korelasi kuantitas terhadap nilai faktur hanya 0,29 pada level item dan 0,46 pada level faktur. Uji tambahan menunjukkan fitur kuantitas berlag (lagged) hanya menambah R2 sebesar +0,014 pada tahap jumlah di set pengujian, di atas riwayat penjualan 30 hari, sehingga tidak layak ditambahkan sebagai kanal masukan. Uji serupa terhadap agregat berlag atribut lain (jumlah merek, supplier, dan salesperson per wilayah-hari) juga hanya menambah R2 sebesar +0,011. Atribut level item/faktur lainnya (nama produk, merek, supplier, salesperson, diskon, pajak, stok, harga satuan) tidak dapat masuk model secara langsung karena target adalah agregat wilayah-hari; identitas salesperson juga merupakan data pribadi dan tidak dilaporkan. Istilah "demand" didefinisikan ulang sebagai total nilai faktur harian per wilayah.

---

**6. Evaluasi dilakukan pada skala apa**

Komentar reviewer: Tidak jelas apakah metrik dihitung pada skala hasil transformasi atau skala asli.

Respons: Bagian 3.8 kini menetapkan bahwa seluruh prediksi dikembalikan ke skala asli (IDR) sebelum metrik dihitung. MAPE tidak dilaporkan karena 84,85% target pengujian bernilai nol, sehingga persentase galat menjadi tidak stabil dan tidak informatif; keputusan ini dijelaskan di naskah. Seluruh tabel hasil (Tabel 3-7) memuat metrik skala asli.

---

**7. Klaim kontribusi spasial belum terisolasi**

Komentar reviewer: Perbandingan model hibrida vs LSTM mandiri belum membuktikan bahwa peningkatan berasal dari struktur geografis.

Respons: Bagian 3.9 menambahkan studi ablasi dengan enam varian: adjacency identitas, graf acak (5 permutasi, rata-rata dilaporkan), graf jarak (varian yang diusulkan), graf korelasi k=3 berdasarkan korelasi penjualan training-only (threshold 0,0126, 67 edge), tanpa fitur COVID-19, dan graf jaringan distribusi. Kriteria interpretasi ditulis eksplisit: klaim spasial hanya dipertahankan jika graf jarak mengungguli varian identitas dan acak. Hasil aktual (Tabel 6, HybridTuned config seed 42): graf jarak R2 0,0436, graf acak R2 0,0486, adjacency identitas R2 0,0538, graf korelasi R2 0,0501. Karena graf jarak tidak mengungguli varian identitas dan acak, kami secara jujur menurunkan klaim kontribusi spasial dari jarak geografis: peningkatan atas LSTM mandiri terutama berasal dari kapasitas model dan formulasi dua tahap, bukan dari struktur geografis. Graf korelasi yang kini tidak degenerate (67 edge, bukan 0 edge seperti konstruksi sebelumnya) juga tidak mengungguli identitas, mengonfirmasi bahwa sinyal spasial memang tipis pada data ini. Varian 6 (graf jaringan distribusi) diimplementasikan sebagai star graph: perusahaan mengonfirmasi bahwa seluruh rute distribusi berpusat di Palembang (pusat distribusi: Komplek Villa Kenten Blok G No. 2), sehingga Palembang terhubung ke semua 15 wilayah lain. Hasil varian ini (R2 0,0703, terbaik di antara semua varian, peningkatan relatif 61% atas graf jarak) dilaporkan pada Tabel 6 dan menunjukkan bahwa topologi distribusi aktual, bukan kedekatan geografis, yang membawa sinyal prediktif.

---

**8. Konstruksi graf belum cukup kuat**

Komentar reviewer: Asal angka 95,82 km, jenis koordinat, sifat graf, dan penggunaan jarak Euclidean tidak jelas.

Respons: Bagian 3.4 kini menjelaskan: jarak dihitung dengan formula Haversine (jarak lingkaran besar) dalam kilometer, bukan Euclidean pada derajat lintang/bujur; koordinat menggunakan centroid tertimbang faktur dari koordinat aktual pelanggan hasil geocoding (118 dari 156 pelanggan, mencakup 98,8% faktur; sumber: SIRS Kemenkes, Photon, idalamat, lewatmana, Google Maps, Medicastore; 38 pelanggan sisanya adalah apotek/individu/dinas kecil tanpa alamat publik dan ditetapkan ke pusat kotanya); node dibatasi ke 16 kabupaten/kota di Provinsi Sumatera Selatan (Musi Rawas Utara digabung ke Musi Rawas karena hanya mencatat 2 faktur dalam periode studi), sedangkan wilayah di luar provinsi yang muncul pada data mentah (Bangka Belitung, Bengkulu, Jakarta Pusat, Jambi, Lebong, Metro, Pinang, Rejang Lebong, Sungailiat) dikeluarkan; graf bersifat tak berarah (undirected), tanpa self-loop, statis, dengan bobot edge biner. Angka aktual: graf akhir memiliki 29 sisi tak berarah, derajat node 3-6 (rata-rata 3,62), dan tidak ada node terisolasi. Angka 104,66 km diturunkan dari distribusi jarak 3 tetangga terdekat (minimum 22,61 km, median 46,68 km, maksimum 104,66 km): threshold dipilih sama dengan jarak maksimum teramati sehingga bertindak sebagai pengaman terhadap koneksi yang tidak realistis; seluruh 48 kandidat sisi berada di bawah threshold.

---

**9. Arsitektur hibrida belum dapat direplikasi**

Komentar reviewer: Detail teknis (jumlah lapisan, kepala atensi, dimensi, fungsi aktivasi, dropout, konfigurasi pelatihan) tidak tersedia dan persamaan LSTM tidak lengkap.

Respons: Bagian 3.5 kini memuat arsitektur dua tahap zero-inflated: (1) klasifikasi biner ada/tidak penjualan dengan binary cross-entropy, dan (2) regresi jumlah pada hari non-nol dengan mean squared error yang diberi mask (masked MSE) pada target log(1+y). Prediksi akhir = keputusan klasifikasi (threshold 0,5) x jumlah. Persamaan LSTM lengkap (gerbang lupa, gerbang masukan, kandidat status sel, pembaruan status sel, gerbang keluaran, pembaruan status tersembunyi), persamaan propagasi GCN (Laplacian ternormalisasi), persamaan penggabungan (concatenation), tabel arsitektur lengkap (LSTM 1 lapisan 128 unit, GCNConv 1 lapisan 128 unit dengan ReLU, lapisan fully connected, masukan dua kanal), dan konfigurasi pelatihan (Adam, laju pembelajaran 0,001, ukuran batch 64, epoch maksimum 60, penghentian dini dengan patience 8, pustaka PyTorch, eksekusi CPU). Versi lingkungan: Python 3.12.7, PyTorch 2.13.0+cpu, pandas 2.2.2, numpy 1.26.4, scikit-learn 1.5.1, scipy 1.13.1; seluruh eksperimen dijalankan pada CPU (tanpa GPU).

---

**10. Optimasi hiperparameter belum benar-benar jelas**

Komentar reviewer: Pencarian acak (random search) tanpa ruang pencarian, jumlah percobaan, dan kriteria pemilihan; kolom "Test" mengindikasikan kebocoran set uji.

Respons: Random search dipertahankan sebagai bagian metodologi dan kini dijelaskan secara eksplisit di Bagian 3.6 sesuai prosedur Bergstra & Bengio (2012): ruang pencarian (unit LSTM dan unit GNN dari distribusi log-uniform [32, 256], laju pembelajaran log-uniform [1e-4, 1e-2], dropout uniform [0, 0,5], ukuran batch uniform {32, 64, 128}), jumlah percobaan (30 trial, seed 42; P(menemukan konfigurasi dalam 10% terbaik) = 1 - 0,9^30 = 0,958), dan kriteria seleksi (loss validasi total: BCE klasifikasi + masked MSE jumlah; penghentian dini patience 8). Set uji hanya digunakan satu kali setelah konfigurasi terbaik dipilih, sehingga tidak ada kebocoran set uji. Hasil aktual: konfigurasi terpilih (LSTM 219, GNN 107, dropout 0,17, lr 1,5e-3, batch 32) dievaluasi ulang dengan 3 seed dan mencapai R2 0,0582 +/- 0,0021 (ensemble 0,0583), terbaik di antara semua model, dan mengungguli konfigurasi dasar secara signifikan (Diebold-Mariano DM = -6,307, p < 0,001). Konfigurasi dasar tetap mengungguli varian tuned studi asli (R2 0,0523 vs 0,0462), sehingga tuning asli tidak bertransfer ke setting harian zero-inflated; hal ini dilaporkan secara jujur di naskah.

---

**11. Model pembanding belum cukup untuk peramalan**

Komentar reviewer: Tidak ada naive, seasonal naive, ARIMA, dan pembanding lain; GNN mandiri bukan pembanding temporal yang adil.

Respons: Bagian 3.7 menambahkan model pembanding yang relevan dengan struktur data zero-inflated: always-zero (prediksi nol untuk semua wilayah dan hari, informatif karena 84,85% target bernilai nol), naive (persistensi), rata-rata training per wilayah, ARIMA per wilayah (order dipilih via AIC pada grid p,q dalam {0,1,2} dan d dalam {0,1}, transformasi log1p, peramalan 231 langkah), XGBoost (regresi pohon dengan fitur riwayat 30 hari dan indikator COVID-19, penghentian dini pada validasi), LSTM mandiri, dan GNN mandiri. Hasil aktual (Tabel 4): always-zero R2 0,0059, naive R2 -0,7978, rata-rata training R2 -7,3242, ARIMA R2 -0,0775, XGBoost R2 -0,0840; model yang diusulkan mengungguli seluruh pembanding. Menariknya, ARIMA dan XGBoost tidak mengungguli prediksi selalu-nol, menegaskan bahwa struktur zero-inflated mendominasi data dan bahwa model yang diusulkan menangkap sinyal di luar persistensi dan model statistik sederhana. Kami setuju bahwa GNN mandiri bukan pembanding temporal yang adil karena tidak memiliki komponen waktu; hal ini dinyatakan eksplisit di naskah dan GNN dipertahankan hanya untuk kelengkapan.

---

**12. Tidak ada ketidakpastian dan pengujian statistik**

Komentar reviewer: Kata "significantly" tanpa uji statistik, eksperimen berulang, atau interval kepercayaan.

Respons: Bagian 3.8 menambahkan protokol: varian tuned dilatih dengan 5 seed acak, model yang diusulkan dan model pembanding dengan 3 seed, hasil dilaporkan sebagai rata-rata +/- standar deviasi, perbandingan berpasangan menggunakan uji Diebold-Mariano, dan interval kepercayaan bootstrap 95% disajikan. Hasil aktual (Tabel 5): uji Diebold-Mariano menolak hipotesis nol kesetaraan akurasi untuk semua perbandingan pada p < 0,001 (LSTM: DM 7,714; GNN: DM 9,586; varian tuned: DM 7,818; semuanya relatif terhadap model yang diusulkan). Interval kepercayaan bootstrap RMSE model yang diusulkan: [6.857.854, 9.192.672] IDR, terendah di antara semua model. Kata "significantly" hanya digunakan untuk hasil yang didukung uji statistik.

---

**13. Klaim residual tidak mempunyai bukti visual**

Komentar reviewer: Klaim simetri residual tanpa plot, histogram, plot Q-Q, atau uji autokorelasi.

Respons: Bagian 3.10 dan 4.6 menambahkan protokol analisis residual: histogram residual, residual per waktu, residual per wilayah, nilai rata-rata residual, dan uji Ljung-Box untuk autokorelasi. Hasil aktual: rata-rata residual 1.665.257 IDR (over-prediksi kecil secara sistematis), uji Ljung-Box Q = 21,510 dengan p = 0,0178, sehingga hipotesis nol tanpa autokorelasi ditolak pada taraf 5%; autokorelasi residual ini diakui sebagai keterbatasan di Section 5 dan menjadi arah perbaikan (misalnya galat autoregresif atau jendela masukan lebih panjang). Klaim simetri hanya dipertahankan jika bukti visual mendukung. Plot residual (histogram, residual per waktu, residual per wilayah) disajikan pada Gambar 8 (`figures/fig08_residuals.jpg`).

---

**14. Tidak ada hasil per wilayah**

Komentar reviewer: Semua hasil digabung menjadi satu nilai global padahal penelitian mengklaim prediksi spatiotemporal untuk 18 wilayah.

Respons: Tabel 7 menambahkan RMSE, MAE, dan R-kuadrat per wilayah untuk 16 kabupaten/kota di Sumatera Selatan. Dua wilayah (Empat Lawang dan Ogan Komering Ilir) mencapai R2 = 1,0 karena nyaris tidak memiliki penjualan pada periode pengujian (fraksi nol di atas 97%), sehingga prediksi nol tepat. Palembang merupakan wilayah dengan kinerja terburuk (R2 -0,3336, RMSE 23.964.716 IDR) karena mencatat penjualan pada hampir setiap hari dengan jumlah harian yang sangat volatil. Analisis peta distribusi galat (Gambar 6, `figures/fig06_error_map.jpg`), plot aktual versus prediksi (Gambar 5, `figures/fig05_actual_vs_predicted.jpg`), dan hubungan galat dengan volume transaksi (Gambar 7, `figures/fig07_error_vs_volume.jpg`) disajikan pada gambar.

---

**15. Implikasi praktis masih terlalu jauh**

Komentar reviewer: Kesimpulan mengklaim pemerataan distribusi, pengurangan kekurangan stok, dan optimasi rute tanpa mengukurnya.

Respons: Kami setuju. Klaim praktis diturunkan: model mendukung peramalan penjualan regional dan perencanaan distribusi. Naskah kini menyatakan secara eksplisit bahwa pemerataan distribusi, kekurangan stok (stockout), dan optimasi rute tidak diukur dalam penelitian ini dan memerlukan data operasional tambahan (tingkat persediaan, waktu tunggu (lead time), biaya distribusi).

---

**16. Referensi**

Komentar reviewer: Ejaan jurnal salah, entri tidak lengkap, DOI salah, dan terdapat referensi yang tidak relevan (WHO Model List of Essential Medicines, Pratama et al.).

Respons: Seluruh daftar referensi telah diperbaiki: ejaan "Journal of Advanced Transportation" dikoreksi; entri Sonani et al. dan Longa et al. dilengkapi; DOI Springer yang salah pada Graph Attention Networks dihapus dan diganti dengan sitasi ICLR 2018 yang benar; referensi WHO Model List of Essential Medicines diganti dengan WHO Global Atlas of Medical Devices 2022 yang relevan dengan alat kesehatan; referensi Pratama et al. (identifikasi varietas kelengkeng) dihapus dan diganti dengan literatur asli Adam (Kingma & Ba, 2015) dan pencarian acak (Bergstra & Bengio, 2012); entri Santos et al., Yang Y. et al., dan Luo et al. dilengkapi dengan volume dan DOI; referensi Lambert (1992) ditambahkan untuk mendukung formulasi zero-inflated. Metadata yang tersisa telah diverifikasi dari sumber primer: Beldek et al. (2019) pp. 570-579, Springer, DOI 10.1007/978-3-030-31343-2_50; Liu & Liu (2024) Vol. 10 No. 12, pp. 65-70, DOI 10.6919/ICJE.202412_10(12).0008; Tunnicliffe Wilson (2016) JTSA 37(5), pp. 709-711 (koreksi dari 37(6)), DOI 10.1111/jtsa.12194; Wen et al. (2024) pp. 108-114, ACM ICBBT 2024, DOI 10.1145/3674658.3674677.

---

**Catatan tambahan tentang novelty (komentar 16 pada daftar asli)**

Komentar reviewer: Penggunaan LSTM-GNN pada penjualan alat kesehatan bersifat kontekstual, bukan novelty algoritmis.

Respons: Kami setuju. Pernyataan novelty di Pendahuluan diturunkan menjadi kontribusi kontekstual: penerapan zero-inflated hybrid LSTM-GNN pada penjualan alat kesehatan regional di Indonesia dengan protokol evaluasi yang dapat direplikasi. Naskah tidak lagi mengklaim novelty algoritmis.

---

**Catatan tambahan: perubahan protokol eksperimen**

Sebagai bagian dari revisi, kami mengevaluasi sensitivitas terhadap resolusi agregasi. Hasilnya: panel bulanan menghasilkan R2 -0,394, mingguan -0,343, dan harian +0,043 s.d. +0,052 (rata-rata tiga seed dengan formulasi zero-inflated). Berdasarkan hasil ini, protokol diubah ke harian zero-inflated, dan fitur indikator COVID-19 (biner, Juli 2020 sampai Desember 2022, aktif 706 hari atau 47,2% panel) ditambahkan sebagai kanal masukan kedua. Perubahan ini dijelaskan di naskah dan menjadi dasar seluruh angka yang dilaporkan.

---

Demikian respons ini kami sampaikan. Kami berterima kasih atas masukan yang sangat membantu meningkatkan kualitas naskah. Seluruh komentar telah kami tindak lanjuti dan kami siap memberikan klarifikasi tambahan jika diperlukan.

Hormat kami,

Khoirusy Syafaat
Herri Setiawan
Faculty of Computer and Natural Sciences, Indo Global Mandiri University, Palembang 30129, Indonesia
khoirusysyafaat@students.uigm.ac.id
herri@uigm.ac.id

---

## Daftar periksa sebelum mengirim

- [x] Seluruh `[ISI: ...]` sudah terisi (termasuk keputusan random search dan penambahan baseline ARIMA/XGBoost).
- [x] Angka model (LSTM, GNN, Hybrid, HybridTuned) di Tabel 3-7 dan abstrak diperbarui dengan hasil eksperimen 16 region (selesai 19 Agu 2026, 21:32); baseline sudah final untuk 16 region.
- [x] Konfirmasi penulis: detail ruang pencarian random search (rentang nilai dan jumlah percobaan) - sudah dicantumkan eksplisit di naskah (Bagian 3.6) dan direspons pada poin 10.
- [x] Semua `[TBD]` pada revisi.md sudah terisi.
- [x] Gambar diregenerasi dengan data terkoreksi (16 wilayah, agregasi harian, window 30, dua kanal masukan) - fig01-10 selesai (19 Agu 2026, 21:40).
- [x] Eksperimen fitur kalender (day-of-week one-hot + libur nasional + fase pandemi) selesai (20 Agu 2026): V2 R2 0,0612 (DM -6,655, p<0,001), V3 (config random search + fitur) R2 0,0666, RMSE 7.609.122, Palembang RMSE turun 8,9% ke 21.842.944. Masuk ke Bagian 3.6, 4.7 (Tabel 8), dan 5.
- [x] Analisis faktor eksternal Data BPS selesai (20 Agu 2026): Belanja Barang & Jasa per kab/kota pooled 2020-2025 r=0,345 p<0,001 (efek skala antar region), within-region r=0,103 p=0,318 (tidak signifikan), cross-sectional tidak ada tahun signifikan - belanja bukan prediktor temporal; penduduk (r=0,164, p=0,111), DBD (r=-0,051, p=0,619), jarak dari Palembang (r=-0,099, p=0,716), dan luas (r=-0,341, p=0,196) tidak signifikan. Masuk ke Bagian 5.
- [x] Struktur folder dirapikan (20 Agu 2026): data/, scripts/, figures/, docs/, manuscript/, logs/, results/; log eksekusi otomatis via `scripts/run.ps1`.
- [x] Verifikasi APBD 2020-2025 (20 Agu 2026): data resmi DJPK Kemenkeu per kab/kota diambil untuk SEMUA tahun (2020-2024 periode=12, 2025 periode=10). Fokus Belanja Barang & Jasa (relevan untuk pengadaan alkes): pooled r=0,345 p=0,0006 (efek skala), within-region r=0,103 p=0,318 (tidak signifikan), cross-sectional tidak ada tahun yang signifikan. Belanja APBD bukan prediktor temporal penjualan. Narasi Bagian 5, limitasi, dan kesimpulan diperbarui.
- [x] Analisis produk bernilai tinggi (20 Agu 2026; angka dikoreksi 21 Agu setelah perbaikan grain baris-item): produk neonatal 100 faktur 940,5 jt (~0,9% total, pembeli terbesar di luar Sumsel); pembelian qty besar didominasi jarum suntik oleh DINKES (Disposable Syringe 6,60 jt unit / 5,78 M IDR, 161 faktur). Masuk ke Bagian 5.
- [x] Penomoran tabel/gambar di naskah final disesuaikan (Table 2 caption ditambahkan di Bagian 3.5; Fig 1-4, 9, 10 direferensikan di naskah).
- [ ] Naskah revisi final dikirim bersama surat balasan ini.