# Spatiotemporal Prediction of Medical Device Sales Using Hybrid LSTM-GNN in South Sumatra

Zero-inflated hybrid Long Short-Term Memory + Graph Neural Network (LSTM-GNN) untuk prediksi spatiotemporal penjualan alat kesehatan (alkes) regional di Sumatera Selatan, Indonesia.

Model memprediksi penjualan harian per kabupaten/kota (16 region) dari data faktur distributor alkes. Pendekatan **zero-inflated dua tahap**: (1) klasifikasi biner apakah ada penjualan pada hari itu, (2) regresi jumlah penjualan pada hari non-nol. Karena 84,85% target test bernilai nol, tahap klasifikasi menyumbang sinyal prediktif dominan.

---

## Hasil Utama

| Model | R2 | RMSE (IDR) | Catatan |
|---|---|---|---|
| Hybrid LSTM-GNN (base) | 0,0523 | 8.002.603 | model utama |
| HybridTuned (random search) | 0,0462 | - | 30 konfigurasi |
| LSTM standalone | 0,0433 | - | pembanding |
| GNN standalone | 0,0059 | - | pembanding |
| + Fitur kalender V2 | 0,0612 | 7.762.318 | day-of-week one-hot + libur + fase pandemi |
| + V3 (rs config + V2) | **0,0666** | **7.609.122** | model terbaik |
| Ablasi graf bintang distribusi | 0,0703 | - | terbaik di semua varian graf |

Temuan kunci:

- **Graf bintang distribusi** (Palembang ke semua region) mengungguli graf jarak geografis (R2 0,0703 vs 0,0436, +61% relatif). Topologi distribusi aktual lebih informatif daripada kedekatan geografis.
- **Fitur kalender** (day-of-week one-hot, libur nasional, fase pandemi) signifikan (DM p < 0,001). Encoding penuh hari lebih baik daripada biner weekend.
- **Klasifikasi zero-inflated** adalah sumber sinyal utama: dengan 84,85% target nol, model lebih baik memprediksi "apakah ada order" daripada jumlahnya. Per-region R2 negatif untuk semua region; keunggulan pooled didorong prediksi nol yang benar.
- **Faktor eksternal**: jumlah rumah sakit berkorelasi dengan penjualan (r = 0,669, p = 0,005) dan residual model (r = 0,697, p = 0,003); belanja pemerintah hanya efek skala antar-region (within-region tidak signifikan).

---

## Struktur Repo

```
.
├── data/                 # Data eksternal publik (BPS, geojson, APBD, imunisasi)
├── docs/                 # Dokumentasi penelitian (lihat indeks di bawah)
├── figures/              # Gambar hasil (fig01-fig10)
├── manuscript/           # Naskah paper (.docx)
├── results/              # Hasil eksperimen (.json) dan cache
├── scripts/              # Skrip eksperimen (aktif)
│   └── archive/          # Skrip eksperimen historis
├── .gitignore
└── .gitattributes
```

### Indeks Dokumentasi (`docs/`)

| File | Isi |
|---|---|
| `revisi.md` | Log revisi paper, status eksperimen, keputusan desain |
| `temuan.md` | Semua temuan analisis terverifikasi + jejak audit (script & file hasil) |
| `usulan_judul_jurnal.md` | 7 usulan paper lanjutan dari temuan |
| `daftar_tbd.md` | Daftar pekerjaan yang belum selesai / TBD |
| `balasan_reviewer.md` | Balasan komentar reviewer |
| `pra_review_comments.md` | Komentar pra-review |

---

## Menjalankan

Skrip berjalan dengan Python di `E:\pyvenv_geo` (venv). Runner standar menulis log otomatis ke `logs/`:

```powershell
.\scripts\run.ps1 experiment_zi_geocoded.py
```

Skrip utama:

- `scripts/experiment_zi_geocoded.py` — eksperimen model utama (zero-inflated LSTM-GNN, 16 region geocoded)
- `scripts/random_search_bb.py` — random search 30 konfigurasi
- `scripts/ablation_rerun_paired.py` — ablasi konstruksi graf
- `scripts/exp_features.py`, `scripts/v3_rs_features.py` — fitur kalender (V1/V2/V3)
- `scripts/plot_all_figures.py` — regenerasi semua gambar
- `scripts/diag_zi.py` — diagnostik collapse classifier zero-inflated

### Konfigurasi model

- Panel: 16 region x 1495 hari; 13.661 faktur unik
- Split: train <= 2023-12-31, validasi 2024, test 2025
- Window 30, horizon 1, transformasi log1p, normalisasi anti-leakage
- Graf: Haversine k=3 (jarak geografis)
- Zero-inflated dua tahap: BCE (klasifikasi) + masked-MSE (regresi non-nol)
- Optimizer Adam, lr=1e-3, batch 64, 60 epoch, patience 8
- Ensemble 3 seeds (42, 7, 123)

---

## Catatan Data

- **Data mentah penjualan tidak di-commit** ke repo publik ini karena berisi nama pelanggan, nama sales, dan detail transaksi (data komersial sensitif). File yang di-exclude: `view_penjualan_detail*.xlsx`, `df.xlsx`, `customers_geocoded*.csv`, `sirs_sumsel_geocoded*.csv`, dan PDF laporan.
- `data/` berisi hanya data eksternal publik (BPS, geojson, APBD, imunisasi).
- Cache hasil (`results/_cache_*.npz`, `results/*.npy`) dan `logs/` di-exclude karena dapat diregenerasi.

---

## Lisensi

Data dan kode untuk keperluan penelitian. Hubungi pemilik untuk penggunaan komersial atau akses data mentah.
