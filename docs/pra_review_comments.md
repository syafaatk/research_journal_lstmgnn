Ada angka MAE dan RMSE yang secara matematis tidak mungkin
Pada Hybrid LSTM-GNN baseline dilaporkan:

RMSE = 0,1252;
MAE = 0,1273.
Ini tidak mungkin jika kedua metrik dihitung dari himpunan error yang sama.

Secara matematis berlaku:

RMSE≥MAE

Tetapi pada paper:

0,1252<0,1273

Inkonsistensi yang sama muncul pada Test 1 di Table 3.

Ini bukan sekadar typo kecil karena angka tersebut digunakan untuk:

membandingkan model;
menghitung persentase peningkatan;
menyatakan keberhasilan tuning;
mendukung kesimpulan utama.
Jika salah satu angka keliru, maka klaim penurunan RMSE atau MAE juga harus dihitung ulang. Sebelum diperbaiki, integritas tabel hasil belum sepenuhnya dapat dipercaya.

Ini adalah kelemahan paling serius dalam paper.

3. Pembagian data belum dapat direplikasi
Paper hanya menyebut:

divided into training, validation, and testing sets following a chronological split

Tetapi tidak menjelaskan:

rasio training, validation, dan testing;
tanggal batas masing-masing subset;
jumlah sampel pada setiap subset;
apakah setiap wilayah mempunyai periode pembagian yang sama;
apakah sliding window dibuat sebelum atau sesudah pembagian data;
apakah Power Transformer hanya dilatih menggunakan training set;
random seed;
bagaimana data kosong pada tanggal tertentu ditangani.
Pada forecasting, chronological split saja belum cukup. Jika Power Transformer dilatih menggunakan keseluruhan data sebelum split, informasi distribusi test set dapat masuk ke preprocessing.

Sliding window juga harus dibuat secara hati-hati. Sequence untuk testing boleh memakai riwayat sebelum test period, tetapi target test tidak boleh masuk ke training atau validation.

Tanpa tanggal pemisahannya, pembaca tidak dapat memverifikasi bahwa leakage benar-benar dihindari.

4. Unit waktu tidak dijelaskan
Paper menggunakan window sebesar 60 time steps, tetapi tidak menerangkan satu time step berarti:

satu hari;
satu minggu;
satu bulan;
satu transaksi;
atau satu periode agregasi tertentu.
Data awal berupa transaksi, tetapi model membutuhkan panel time series yang teratur untuk 18 wilayah.

Belum dijelaskan:

apakah data diagregasi harian atau bulanan;
apakah hari tanpa penjualan diberi nilai nol;
apakah semua wilayah mempunyai timestamp yang sama;
bagaimana missing dates ditangani;
berapa jumlah sequence akhir setelah windowing;
berapa dimensi input model.
Tanpa informasi ini, desain LSTM dan GNN tidak dapat direplikasi.

5. Target prediksi belum tegas
Dataset memiliki atribut:

quantity sold;
total sales value;
product information;
customer location.
Namun paper berulang kali menggunakan istilah “sales” dan “demand” tanpa menetapkan secara tegas apakah targetnya:

jumlah unit alat kesehatan;
nilai penjualan rupiah;
jumlah transaksi;
atau agregasi seluruh produk.
“Sales value” tidak selalu sama dengan demand. Nilai penjualan dapat berubah karena:

harga produk;
inflasi;
diskon;
perbedaan jenis alat kesehatan;
perubahan komposisi produk.
Jika targetnya total nilai penjualan, kesimpulan tentang “equitable distribution of medical devices” terlalu jauh. Nilai penjualan tinggi tidak selalu berarti kebutuhan alat kesehatan tinggi.

6. Evaluasi dilakukan pada skala apa?
Paper menggunakan Power Transformer terhadap target. Namun tidak dijelaskan apakah prediksi:

dikembalikan dahulu ke skala asli;
atau metrik dihitung pada skala hasil transformasi.
Nilai RMSE 0,0881 dan MAE 0,0324 tidak memiliki satuan yang dapat diinterpretasikan. Jika dihitung pada transformed scale, angka tersebut tidak dapat langsung digunakan untuk menjelaskan kesalahan penjualan atau permintaan sebenarnya.

MAPE pada nilai hasil Power Transformer juga berpotensi menyesatkan, terutama jika hasil transformasi mengandung nilai nol atau negatif.

Paper seharusnya melaporkan metrik pada skala asli, misalnya:

unit produk;
jumlah transaksi;
atau nilai rupiah.
7. Klaim kontribusi spasial belum terisolasi
Paper menyimpulkan bahwa spatial dependency meningkatkan prediksi karena hybrid model mengungguli standalone LSTM.

Namun perbandingan tersebut belum cukup untuk membuktikan bahwa peningkatan berasal dari struktur geografis.

Diperlukan ablation seperti:

LSTM tanpa graph;
LSTM dengan graph sebenarnya;
LSTM dengan graph acak;
LSTM dengan adjacency identity;
graph berbasis jarak;
graph berbasis korelasi penjualan;
graph berbasis jaringan distribusi;
beberapa nilai k dan distance threshold.
Tanpa shuffled-graph atau no-edge baseline, peningkatan bisa berasal dari:

kapasitas parameter yang lebih besar;
attention fusion;
dense layer tambahan;
perbedaan proses tuning;
bukan dari hubungan geografis itu sendiri.
Karena itu, klaim bahwa spatial relationships “significantly improve” prediksi masih terlalu kuat.

8. Konstruksi graph belum cukup kuat
Paper menyatakan menggunakan tiga tetangga terdekat dan threshold 95,82 km. Namun belum dijelaskan:

dari mana angka 95,82 km diperoleh;
apakah koordinat merupakan centroid wilayah, lokasi pelanggan, atau kantor pemerintah;
bagaimana wilayah yang tidak memiliki tiga tetangga dalam radius tersebut diperlakukan;
apakah graph directed atau undirected;
apakah self-loop digunakan;
apakah edge mempunyai bobot;
berapa jumlah edge akhirnya;
apakah graph tetap atau berubah terhadap waktu.
Penggunaan “Euclidean distance” juga perlu diperhatikan. Jika koordinat masih berupa latitude dan longitude, jarak Euclidean dalam derajat tidak sama dengan jarak kilometer. Seharusnya menggunakan Haversine atau metode geodesic, kecuali koordinat sudah ditransformasikan ke sistem proyeksi metrik.

Figure 1 menunjukkan jaringan wilayah, tetapi tidak cukup untuk menjelaskan seluruh konstruksi adjacency matrix.

9. Arsitektur hybrid belum dapat direplikasi
Paper menyebut:

LSTM;
GAT;
attention-based fusion;
fully connected layer.
Namun detail teknis penting tidak tersedia:

jumlah GAT layers;
jumlah attention heads;
dimensi node embedding;
fungsi aktivasi;
dropout;
jumlah layer LSTM;
hidden state dimension untuk seluruh konfigurasi;
bentuk tensor input dan output;
mekanisme fusion;
persamaan attention fusion;
loss pada validation;
early stopping;
jumlah epoch;
patience;
weight decay;
inisialisasi bobot;
library dan versinya.
Bahkan bagian persamaan LSTM hanya menampilkan forget gate dan input gate. Tidak ada penjelasan lengkap tentang:

candidate cell state;
cell-state update;
output gate;
hidden-state update.
Untuk paper deep learning, metode saat ini belum cukup reproducible.

10. Optimasi hyperparameter belum benar-benar jelas
Paper menyatakan menggunakan random search, tetapi Table 3 hanya menunjukkan lima konfigurasi yang tersusun secara teratur.

Belum dijelaskan:

search space;
distribusi sampling;
jumlah total trial;
kriteria pemilihan model;
apakah pemilihan menggunakan validation loss;
apakah angka di Table 3 merupakan hasil validation atau test;
apakah test set dipakai berulang kali selama tuning.
Kolom pertama bahkan diberi nama “Test”. Jika lima konfigurasi dibandingkan menggunakan test set, kemudian konfigurasi terbaik dipilih berdasarkan hasil test, maka terjadi test-set leakage.

Test set seharusnya hanya digunakan satu kali setelah konfigurasi terbaik dipilih berdasarkan validation set.

11. Baseline belum cukup untuk forecasting
Paper hanya menggunakan:

standalone LSTM;
standalone GNN;
hybrid LSTM-GNN;
tuned hybrid.
Belum ada baseline forecasting yang sederhana tetapi penting, seperti:

persistence/naive forecast;
seasonal naive;
moving average;
ARIMA/SARIMA;
Prophet;
XGBoost atau LightGBM;
GRU;
temporal convolutional network.
Tanpa naive dan seasonal baseline, belum diketahui apakah deep learning benar-benar diperlukan.

Standalone GNN juga bukan baseline temporal yang adil karena GNN memang tidak dirancang menangani sequence tanpa komponen waktu. Kinerjanya yang rendah sudah dapat diperkirakan sejak awal.

12. Tidak ada ketidakpastian dan pengujian statistik
Paper menggunakan kata:

significantly improves

Tetapi tidak ada:

uji statistik;
repeated experiment;
standard deviation;
confidence interval;
hasil beberapa random seed;
Diebold–Mariano test;
bootstrap interval;
fold berbasis waktu.
Deep learning dapat menghasilkan skor berbeda berdasarkan inisialisasi bobot. Satu nilai RMSE untuk setiap model belum cukup untuk menyebut peningkatan “significant”.

Kata “significantly” seharusnya diganti menjadi “substantially” atau dibuktikan dengan pengujian statistik.

13. Klaim residual tidak mempunyai bukti visual
Discussion menyatakan:

residual analysis ... shows that prediction errors are symmetrically distributed around zero

Namun paper tidak menyajikan:

residual plot;
histogram residual;
Q–Q plot;
residual per waktu;
residual per wilayah;
uji autokorelasi;
nilai rata-rata residual.
Jadi klaim tersebut tidak memiliki bukti yang dapat diperiksa.

Figure 3 hanya membandingkan metrik dan Figure 4 hanya menunjukkan perubahan hasil tuning.

14. Tidak ada hasil per wilayah
Penelitian mengklaim sebagai spatiotemporal prediction untuk 18 wilayah, tetapi semua hasil digabung menjadi satu nilai global.

Tidak tersedia:

RMSE setiap wilayah;
MAE setiap wilayah;
MAPE setiap wilayah;
wilayah dengan prediksi terbaik dan terburuk;
peta distribusi error;
plot actual versus predicted;
analisis wilayah terpencil;
hubungan error dengan jumlah transaksi.
Tanpa evaluasi per node/wilayah, kontribusi spasial justru belum terlihat pada bagian hasil.

15. Implikasi praktis masih terlalu jauh
Kesimpulan menyatakan model dapat mendukung:

pemerataan distribusi;
pengurangan stock shortage;
optimasi rute;
perencanaan alat kesehatan.
Tetapi penelitian hanya memprediksi penjualan. Paper tidak mengukur:

stockout;
inventory level;
lead time;
kebutuhan fasilitas kesehatan;
equity;
distribution cost;
route optimization.
Karena itu, hasilnya lebih tepat disebut membantu forecasting penjualan regional, belum membuktikan pemerataan distribusi alat kesehatan.

16. Referensi
Referensinya relatif mutakhir dan mendukung konteks LSTM-GNN. Namun kualitas dan konsistensinya masih campuran:

beberapa referensi berupa prosiding;
terdapat referensi arXiv;
terdapat working paper/SSRN;
beberapa bibliografi tidak lengkap;
“Journal of Advanced Transporation” salah eja;
referensi Sonani et al. belum memiliki informasi jurnal/penerbit yang lengkap;
referensi Longa et al. tidak lengkap;
DOI yang dicantumkan untuk Graph Attention Networks perlu diverifikasi;
WHO Model List of Essential Medicines membahas obat esensial, bukan secara langsung alat kesehatan;
Pratama et al. tentang identifikasi varietas kelengkeng tidak tepat sebagai rujukan utama random search dan Adam optimizer;
sumber metode seharusnya merujuk langsung pada literatur asli Adam dan random search.
Kajian pustaka juga terlalu bergantung pada perbedaan domain sebagai novelty. Penggunaan LSTM-GNN pada medical-device sales memang kontekstual, tetapi bukan novelty algoritmis.