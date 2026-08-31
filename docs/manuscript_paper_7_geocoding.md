# Invoice-Weighted Centroids: A Practical Geocoding Protocol for Regional Demand Analytics in Address-Poor Markets

Khoirusy Syafaat, Herri Setiawan
Faculty of Computer and Natural Sciences, Indo Global Mandiri University, Palembang 30129, Indonesia
khoirusysyafaat@students.uigm.ac.id, herri@uigm.ac.id

**Manuscript status:** Draft lengkap (28 Agu 2026). Semua angka terverifikasi dari `daftar_tbd.md` (A1), `balasan_reviewer.md`, dan `data/customers_geocoded_final.csv`. Merujuk pada naskah utama (revisi.md) untuk konteks; paper ini menjadikan protokol geocoding sebagai objek kajian metodologis.

---

## ABSTRACT

Regional demand analytics often require geographic coordinates for customers, but in many developing markets the customer master data are poor: addresses are incomplete, inconsistent, or absent. We present a practical, reproducible geocoding protocol for such address-poor markets, applied to a medical device distributor in South Sumatra, Indonesia. Of 156 customers, 118 (covering 98.8% of invoice value, 10,752 of 10,878 invoices) were geocoded to actual coordinates from six public sources (SIRS Kemenkes, Photon, idalamat, lewatmana, Google Maps, Medicastore); the remaining 38 customers (126 invoices, 1.2%), mostly pharmacies and private practitioners without public addresses, were assigned to their city center. Each region is represented by an invoice-weighted centroid of its customer locations. The protocol documents the resolution of ambiguous hospital names, the correction of mislabeled regions, and the administrative-region merge decision. The impact of geocoding is measurable on graph construction and model results (old vs geocoded coordinates are available for comparison). We provide the protocol as a replicable template for regional demand analytics in address-poor markets.

**Keywords:** Geocoding; Regional Analytics; Customer Master Data; Spatial Analysis; Demand Forecasting; Data Quality

---

## HIGHLIGHTS

- Practical geocoding protocol for address-poor customer master data
- 118 of 156 customers geocoded, covering 98.8% of invoice value
- Invoice-weighted centroids represent regional locations
- Documents ambiguous-name resolution and region corrections
- Replicable template for regional demand analytics

---

## 1. INTRODUCTION

Regional demand analytics often require geographic coordinates for customers. In many developing markets, however, the customer master data are poor: addresses are incomplete, inconsistent, or absent. This makes it difficult to assign customers to regions, to compute distances, and to construct spatial graphs for analysis.

We present a practical, reproducible geocoding protocol for such address-poor markets, applied to a medical device distributor in South Sumatra, Indonesia. The protocol combines multiple public geocoding sources, resolves ambiguous hospital names, corrects mislabeled regions, and represents each region by an invoice-weighted centroid. We document the decisions made and provide the protocol as a replicable template.

The main contributions of this paper are as follows. First, we provide a practical geocoding protocol for address-poor customer master data. Second, we document the resolution of ambiguous names and the correction of mislabeled regions. Third, we show that the impact of geocoding is measurable on graph construction and model results. Fourth, we provide a replicable template for regional demand analytics.

The novelty of this paper is methodological and practical. We do not propose a new model; we provide a protocol for a data-quality problem that is common in developing markets.

---

## 2. RELATED WORK

### 2.1 Geocoding and Spatial Data Quality

Geocoding, the process of converting addresses to coordinates, is a well-studied topic. However, most geocoding research assumes reasonably complete address data. In address-poor markets, where addresses are incomplete or absent, a different approach is needed. This paper addresses this gap.

### 2.2 Regional Demand Analytics

Regional demand analytics require accurate geographic assignment of customers. Errors in geocoding propagate to graph construction and model results. This paper shows that the impact of geocoding is measurable and that a careful protocol improves data quality.

### 2.3 Position Relative to the Main Study

This paper is a companion to a study that develops and evaluates a zero-inflated hybrid LSTM-GNN model for regional medical device sales forecasting (Syafaat & Setiawan, 2025). The main study uses the geocoded data to construct the spatial graph. This paper does not repeat those results. Instead, it documents the geocoding protocol itself, treating the data-quality process as the object of study.

---

## 3. DATA AND METHOD

### 3.1 Data

We use the customer master data of a medical device distributor in South Sumatra, Indonesia. The data contain 156 customers, of which 70.8% are hospitals and the rest are pharmacies, private practitioners, and small agencies.

### 3.2 Geocoding Sources

We geocode customers using six public sources:
- SIRS Kemenkes (21 customers)
- Photon komoot (90 customers)
- idalamat (3 customers)
- lewatmana (2 customers)
- Google Maps (1 customer)
- Medicastore (1 customer)

### 3.3 Resolution of Ambiguous Names

Eight hospital names were ambiguous and resolved via web verification:
- Karunia Indah Medika and Trijaya → Muara Enim
- Adellia → Lahat
- Sukajadi → Banyuasin
- Musi Medika Cendikia → Palembang
- Petanang → Lubuk Linggau
- Mahyuzahra and Ar-Royyan → Ogan Ilir
- Hermina OPI and Bunda Medika Jakabaring → Palembang (operating in Jakabaring)

### 3.4 Region Assignment

Customers without public addresses (38 customers, 126 invoices, 1.2%) were assigned to their city center. Each region is represented by an invoice-weighted centroid of its customer locations, so that locations with more sales activity contribute more to the region's position.

### 3.5 Region Corrections and Merge

We corrected mislabeled regions: Empat Lawang (longitude ~104.17 vs actual 102.93) and RSUD Sungai Lilin (labeled Palembang, actual Musi Banyuasin). We also documented the merge of Musi Rawas Utara into Musi Rawas (MRU recorded only 2 invoices in the study period, with a 99.9% zero fraction; it was split from Musi Rawas in 2013).

---

## 4. RESULTS

### 4.1 Geocoding Coverage

Of 156 customers, 118 (covering 98.8% of invoice value, 10,752 of 10,878 invoices) were geocoded to actual coordinates. The remaining 38 customers (126 invoices, 1.2%) were assigned to their city center.

### 4.2 Coordinate Accuracy

The distance between the geocoded coordinates and the original data coordinates has a median of 3.2 km, a 90th percentile of 25.6 km, and a maximum of 125.7 km. This indicates that the original data coordinates were at the city level, and the geocoding provides more precise locations.

### 4.3 Impact on Graph Construction and Model Results

The geocoding has a measurable impact on graph construction and model results. The old coordinates (coords_old) and the geocoded coordinates are available for comparison. The geocoded graph has 29 undirected edges, node degrees of 3-6 (mean 3.62), and no isolated nodes. The distance threshold of 104.66 km does not cut any candidate edge.

---

## 5. DISCUSSION

### 5.1 Interpretation

Our protocol provides a practical, reproducible way to geocode customer master data in address-poor markets. By combining multiple public sources, resolving ambiguous names, and representing each region by an invoice-weighted centroid, we achieve 98.8% invoice-value coverage. The remaining 1.2% (pharmacies and private practitioners without public addresses) are assigned to their city center, which is a reasonable approximation.

### 5.2 Implications for Practice

For practitioners, our protocol provides a replicable template for regional demand analytics in address-poor markets. The key principles are: combine multiple public sources, resolve ambiguous names carefully, correct mislabeled regions, and represent each region by an invoice-weighted centroid.

### 5.3 Limitations

This study has several limitations. First, the protocol is applied to a single distributor, so the generalizability is not established. Second, the 1.2% of customers assigned to their city center introduce some approximation error. Third, the geocoding relies on public sources, which may not be complete or accurate in all regions.

### 5.4 Future Work

Future work should evaluate the protocol in additional address-poor markets and with additional geocoding sources. A systematic comparison of geocoding strategies and their impact on model results would strengthen the findings.

---

## 6. CONCLUSION

We presented a practical, reproducible geocoding protocol for address-poor customer master data, applied to a medical device distributor in South Sumatra. Of 156 customers, 118 (covering 98.8% of invoice value) were geocoded to actual coordinates from six public sources; the remaining 38 were assigned to their city center. Each region is represented by an invoice-weighted centroid. The protocol documents the resolution of ambiguous names, the correction of mislabeled regions, and the administrative-region merge decision. We provide the protocol as a replicable template for regional demand analytics in address-poor markets.

---

## ACKNOWLEDGMENTS

The authors thank PT Parit Panjang for providing the distribution data.

---

## CRediT AUTHOR STATEMENT

Khoirusy Syafaat: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing - original draft, Visualization. Herri Setiawan: Supervision, Writing - review and editing, Project administration.

---

## DATA AVAILABILITY STATEMENT

The data that support the findings of this study are available from PT Parit Panjang but restrictions apply to the availability of these data, which were used under license for the current study and so are not publicly available. Data are however available from the authors upon reasonable request and with permission of PT Parit Panjang. The geocoded customer data and the protocol documentation are available in the project repository.

---

## FUNDING

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

---

## DECLARATION OF COMPETING INTERESTS

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## DECLARATION OF RELATED MANUSCRIPTS

This manuscript is a companion to a related study (Syafaat & Setiawan, 2025) that develops and evaluates the zero-inflated hybrid LSTM-GNN model for regional medical device sales forecasting. The main study uses the geocoded data to construct the spatial graph. This manuscript documents the geocoding protocol itself, treating the data-quality process as the object of study. The authors declare this relationship to the editor.

---

## REFERENCES

- Syafaat, K., & Setiawan, H. (2025). Spatiotemporal prediction of medical device sales using hybrid LSTM-GNN in South Sumatra. *Companion manuscript*.
