# Penyesuaian Format Paper #1 untuk Jurnal Target

Dokumen ini merangkum kebijakan format jurnal target untuk paper #1 (*When Geography Lies*) dan penyesuaian yang perlu dilakukan sebelum submit. Informasi kebijakan diverifikasi dari situs resmi jurnal (28 Agu 2026).

**Sasaran utama:** Expert Systems with Applications (ESWA), Elsevier — jurnal applied AI/intelligent systems.

---

## 1. Ringkasan Kebijakan ESWA yang Relevan

| Aspek | Kebijakan | Status paper #1 |
|---|---|---|
| File format | Word (.docx) single-column ATAU LaTeX (.tex); PDF bukan source file | Perlu konversi dari .md |
| Abstract | 200-250 kata, berdiri sendiri, tanpa referensi | Abstract saat ini ~230 kata — OK, perlu dicek |
| Keywords | AI/ML keywords mencerminkan domain aplikasi | Ada, perlu disesuaikan |
| Highlights | 3-5 poin, masing-masing maks 85 karakter | **BELUM ADA — perlu ditambahkan** |
| Double anonymized review | Title page terpisah (tanpa nama/afiliasi di naskah) | **Perlu title page terpisah** |
| Figures | 300 dpi minimum; PNG/TIFF/JPG untuk halftone, EPS/PDF untuk vector | Figure sudah 300 dpi PNG — OK |
| CRediT author statement | Wajib, spesifik per peran | **BELUM ADA — perlu ditambahkan** |
| Data statement | Wajib (Option C: deposit data + cite; jika tidak bisa, nyatakan alasan) | **BELUM ADA — perlu ditambahkan** |
| Funding statement | Wajib disclosure | **BELUM ADA — perlu ditambahkan** |
| Conflicts of interest | Wajib untuk semua penulis | **BELUM ADA — perlu ditambahkan** |
| ORCID | Wajib untuk corresponding author | **Perlu ditambahkan** |
| Related manuscripts | Wajib disclosure co-submission/related | **Perlu deklarasi terkait naskah utama** |
| References | Elsevier style (author-year atau numbered) | Perlu disesuaikan |

---

## 2. Elemen yang Perlu Ditambahkan ke Paper #1

### 2.1 Highlights (3-5 poin, masing-masing maks 85 karakter)

Draf highlights untuk paper #1:

1. Distribution-network star graph beats geographic distance in GNN forecasting
2. Distance graph performs worse than identity and random graphs
3. Geographic proximity adds no signal in sparse regional demand data
4. Graph should encode how goods move, not where facilities are located
5. Reproducible ablation protocol for graph construction selection

### 2.2 CRediT Author Statement

```
Khoirusy Syafaat: Conceptualization, Methodology, Software, Formal analysis,
Investigation, Data curation, Writing - original draft, Visualization.
Herri Setiawan: Supervision, Writing - review and editing, Project administration.
```

### 2.3 Data Statement

```
The data that support the findings of this study are available from PT Parit Panjang
but restrictions apply to the availability of these data, which were used under
license for the current study and so are not publicly available. Data are however
available from the authors upon reasonable request and with permission of PT Parit
Panjang. The derived regional panel and the ablation results are available in the
project repository.
```

### 2.4 Funding Statement

```
This research received no specific grant from any funding agency in the public,
commercial, or not-for-profit sectors.
```

### 2.5 Conflicts of Interest

```
The authors declare that they have no known competing financial interests or
personal relationships that could have appeared to influence the work reported
in this paper.
```

### 2.6 Declaration of Related Manuscripts

```
This manuscript is a companion to a related study (Syafaat & Setiawan, 2025) that
develops and evaluates the zero-inflated hybrid LSTM-GNN model. The two manuscripts
share the same dataset and model framework but ask different questions. This
manuscript isolates the question of graph construction; the companion manuscript
reports the overall model and its baselines. The authors declare this relationship
to the editor.
```

### 2.7 Title Page (terpisah untuk double anonymized review)

```
Title: When Geography Lies: Distribution Network Topology Outperforms Spatial
Proximity in Graph-Based Regional Demand Forecasting

Authors:
Khoirusy Syafaat (khoirusysyafaat@students.uigm.ac.id)
Herri Setiawan (herri@uigm.ac.id)

Affiliations:
Faculty of Computer and Natural Sciences, Indo Global Mandiri University,
Palembang 30129, Indonesia

Corresponding author: Khoirusy Syafaat

ORCID: [perlu diisi]
```

---

## 3. Penyesuaian Format Lainnya

### 3.1 Abstract
- Target 200-250 kata. Abstract saat ini perlu dihitung dan dipangkas jika melebihi 250 kata.
- Pastikan tidak ada referensi di abstract.

### 3.2 Keywords
- Sesuaikan dengan domain AI/ML: Graph Neural Networks; Demand Forecasting; Graph Construction; Distribution Network; Zero-Inflated Data; Spatiotemporal Prediction.
- Tambahkan kata kunci domain aplikasi: Medical Device Supply Chain; Regional Demand.

### 3.3 Figures
- Figure sudah dibuat pada 300 dpi PNG (`figures/paper1_ablation_bar.png`, `figures/paper1_star_graph.png`).
- Untuk ESWA, vector (EPS/PDF) lebih disukai untuk line drawings. Perlu konversi ke PDF/EPS jika memungkinkan.
- Caption harus terpisah dari gambar (tidak di dalam gambar).

### 3.4 References
- ESWA menggunakan Elsevier style. Perlu memastikan format referensi konsisten (author-year atau numbered).
- Referensi yang belum lengkap (mis. "et al." tanpa tahun lengkap) perlu dilengkapi.

### 3.5 Struktur
- ESWA tidak mensyaratkan struktur IMRaD kaku, tapi umumnya: Introduction, Related Work, Method, Results, Discussion, Conclusion.
- Paper #1 sudah sesuai.

---

## 4. Catatan Kesesuaian dengan Scope ESWA

ESWA fokus pada intelligent systems applied ke real-world problems. Paper #1 cocok karena:
- Punya aplikasi nyata (distribusi alkes di Sumatera Selatan).
- Bukan sekadar benchmark comparison; ada temuan metodologis (konstruksi graf).
- Domain forecasting/logistics relevan.

**Peringatan dari panduan ESWA:** "ESWA wants a real application with a real system. A benchmark comparison is the usual mismatch." Paper #1 harus menekankan aplikasi nyata dan implikasi praktis, bukan hanya perbandingan metode. Section 5.2 (Implications for Practice) sudah menekankan ini.

**Alternatif jurnal jika ESWA kurang cocok:**
- International Journal of Forecasting (applied track) — lebih fokus forecasting.
- Applied Intelligence (Springer) — applied AI.
- IEEE Access — open access, format IEEE.
- Jurnal nasional SINTA 1-2 (mis. Jurnal RESTI, Jurnal Informatika) — konteks Indonesia.

---

## 5. Checklist Sebelum Submit

- [ ] Konversi naskah dari .md ke .docx (Word) atau .tex (LaTeX)
- [ ] Tambahkan highlights (3-5 poin)
- [ ] Pisahkan title page (double anonymized)
- [ ] Tambahkan CRediT statement
- [ ] Tambahkan data statement
- [ ] Tambahkan funding statement
- [ ] Tambahkan conflicts of interest
- [ ] Tambahkan deklarasi related manuscripts
- [ ] Tambahkan ORCID
- [ ] Hitung dan sesuaikan panjang abstract (200-250 kata)
- [ ] Sesuaikan keywords
- [ ] Konversi figure ke vector (PDF/EPS) jika memungkinkan
- [ ] Lengkapi referensi (Elsevier style)
- [ ] Verifikasi semua angka final
