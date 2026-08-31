# Procurement Cycles, Not Outbreaks: Temporal Patterns of Syringe Purchasing in South Sumatra

Khoirusy Syafaat, Herri Setiawan
Faculty of Computer and Natural Sciences, Indo Global Mandiri University, Palembang 30129, Indonesia
khoirusysyafaat@students.uigm.ac.id, herri@uigm.ac.id

**Manuscript status:** Draft lengkap (28 Agu 2026). Semua angka terverifikasi dari `temuan.md` (Section 3.5, distribusi temporal syringe + uji wabah) dan `results/syringe_temporal.json`, `results/syringe_wabah.json`. Merujuk pada naskah utama (revisi.md) untuk konteks; paper ini menjadikan pengadaan syringe sebagai objek kajian, cocok untuk short communication/research note.

---

## ABSTRACT

A common assumption in healthcare supply chains is that disease outbreaks drive medical device demand. We test this assumption for syringe purchasing in South Sumatra, Indonesia, using real distribution data (1,756 invoices, 12,372,163 units, IDR 12.34M). The syringe purchase peaks recur in July and November, following the fiscal cycle, not the epidemiological calendar. The highest peak (July 2022, 709,779 units) occurred in the mid-year fiscal cycle, not during the Omicron wave (January-February 2022, which had the lowest quantities); November 2023, a post-pandemic (endemic) month, reached 460,191 units, again not during an outbreak. The dengue test is not significant (pooled r = 0.048, p = 0.677; within-region r = 0.072, p = 0.536), and the COVID test is not significant (Mann-Whitney p = 0.614). The context of Muara Enim, the largest syringe buyer, is a routine immunization program (pertussis and measles cases due to declining immunization coverage). We conclude that the fiscal calendar determines syringe purchasing more than epidemiology, challenging the assumption that outbreaks drive medical device demand.

**Keywords:** Syringe Procurement; Fiscal Cycle; Immunization; Outbreak; Medical Device Demand; Public Health

---

## HIGHLIGHTS

- Syringe purchase peaks recur in July and November (fiscal cycle)
- The highest peak (July 2022, 709,779 units) is not tied to an outbreak
- Dengue and COVID tests are not significant
- Fiscal calendar determines syringe demand more than epidemiology
- Challenges the assumption that outbreaks drive medical device demand

---

## 1. INTRODUCTION

A common assumption in healthcare supply chains is that disease outbreaks drive medical device demand. When an outbreak occurs, the reasoning goes, demand for related medical devices surges. We test this assumption for syringe purchasing in South Sumatra, Indonesia.

Syringes are a useful case because they are used in both routine services (immunization, injection) and outbreak response (vaccination campaigns, dengue treatment). If outbreaks drive demand, syringe purchases should spike during outbreak periods. If the fiscal calendar drives demand, syringe purchases should follow the budget cycle.

We use real distribution data from a medical device distributor in South Sumatra. We show that syringe purchase peaks recur in July and November, following the fiscal cycle, not the epidemiological calendar. The highest peak (July 2022, 709,779 units) occurred in the mid-year fiscal cycle, not during the Omicron wave (January-February 2022, which had the lowest quantities); November 2023, a post-pandemic (endemic) month, reached 460,191 units, again not during an outbreak. The dengue and COVID tests are not significant. We conclude that the fiscal calendar determines syringe purchasing more than epidemiology.

The main contributions of this paper are as follows. First, we challenge the assumption that outbreaks drive medical device demand. Second, we provide evidence that the fiscal calendar determines syringe purchasing. Third, we provide a short, focused analysis suitable for a research note.

The novelty of this paper is empirical and focused. We do not propose a new model; we provide evidence about the drivers of syringe purchasing.

---

## 2. RELATED WORK

### 2.1 Outbreak-Driven Demand

The assumption that outbreaks drive medical device demand is common but rarely tested with real distribution data. The COVID-19 pandemic highlighted the importance of understanding demand drivers for supply chain planning. This paper provides a focused test of the outbreak-driven demand assumption for syringes.

### 2.2 Fiscal Cycles in Public Procurement

Public procurement follows fiscal cycles. In Indonesia, the fiscal year runs from January to December, and procurement is concentrated in the middle and end of the year (around July and November-December). This pattern is well known in public procurement but is rarely connected to medical device demand.

### 2.3 Position Relative to the Main Study

This paper is a companion to a study that develops and evaluates a zero-inflated hybrid LSTM-GNN model for regional medical device sales forecasting (Syafaat & Setiawan, 2025). The main study reports the overall model and its baselines. This paper does not repeat those results. Instead, it isolates the syringe product group and tests the outbreak-driven demand assumption. The unit of analysis is the syringe product group over time.

---

## 3. DATA AND METHOD

### 3.1 Data

We use medical device sales data from PT Parit Panjang, a distribution company in South Sumatra, Indonesia, covering July 2020 to October 2025. The syringe product group comprises 1,756 invoices, 12,372,163 units, and IDR 12.34M in value.

### 3.2 Unit Normalization

Syringe quantities are normalized to pieces (pcs). The spuit (syringe) is sold in both Pcs and Box units, with a box containing 100 pcs. Special products (pre-filled, insulin, tuberculin) are excluded because their box contents differ. This ensures that the quantity comparison is valid.

### 3.3 Outbreak Tests

We test the outbreak-driven demand assumption using two epidemiological indicators:
1. **Dengue.** Syringe quantity vs dengue cases per 100,000 (region-year, n=77), using pooled and within-region correlation.
2. **COVID.** Syringe quantity per month vs pandemic phase (July 2020-December 2022 vs 2023+), using the Mann-Whitney test and Spearman correlation.

### 3.4 Temporal Analysis

We analyze the monthly distribution of syringe purchases to identify recurring peaks and their timing relative to the fiscal calendar.

---

## 4. RESULTS

### 4.1 Recurring Peaks in July and November

Syringe purchases show recurring peaks in July and November, following the fiscal cycle. The mass procurement (quantity >= 10,000 per line item) is spread across 2020-2025 with consistent mid-year peaks (June-July, the start of the fiscal year) plus a December surge (end of the fiscal year). This is a government procurement pattern.

### 4.2 The Highest Peak Is Not Tied to an Outbreak

The highest peak (July 2022, 709,779 units) occurred in the mid-year fiscal cycle, not during the Omicron wave (January-February 2022), which had the lowest syringe quantities (37,659 and 57,854 units). November 2023, a post-pandemic (endemic) month, reached 460,191 units, again not during an outbreak. This is inconsistent with the outbreak-driven demand assumption: if outbreaks drove demand, the peaks should occur during outbreaks, not during non-outbreak months.

### 4.3 Dengue Test Is Not Significant

The dengue test is not significant: pooled r = 0.048 (p = 0.677) and within-region r = 0.072 (p = 0.536). The highest dengue year (2024, 70.4/100k) is not the highest syringe year; the highest syringe year (2021) had the lowest dengue (13.4/100k).

### 4.4 COVID Test Is Not Significant

The COVID test is not significant: the pandemic phase (July 2020-December 2022) has a mean of 220,398 units/month vs 169,418 units/month in the endemic phase, but the Mann-Whitney test gives p = 0.614 and the Spearman correlation is r = 0.064 (p = 0.613). The highest peak (July 2022, 709,779 units) is not during the Omicron wave (January-February 2022), which had the lowest syringe quantities (37,659 and 57,854 units).

### 4.5 The Muara Enim Context

Muara Enim, the largest syringe buyer, provides the context. Its 2023 health cases were dominated by ISPA, pulmonary TB, and PD3I, with 6 pertussis and 13 measles cases due to declining basic immunization coverage. The syringe procurement by the health office is related to routine/kejar immunization programs, not to outbreak response.

---

## 5. DISCUSSION

### 5.1 Interpretation

Our results challenge the assumption that outbreaks drive medical device demand. Syringe purchases follow the fiscal calendar, with recurring peaks in July and November, and the highest peak (July 2022, 709,779 units) occurred in the mid-year fiscal cycle rather than during the Omicron wave; November 2023 (an endemic month) also reached 460,191 units without an outbreak. The dengue and COVID tests are not significant.

The mechanism is clear: syringe procurement is driven by routine immunization and service programs, which follow the budget cycle, not by outbreak response. The Muara Enim context confirms this: the syringe procurement is related to routine immunization, not to outbreak response.

### 5.2 Implications for Practice

For supply chain managers, our results suggest that syringe demand should be planned around the fiscal calendar, not the epidemiological calendar. Stock should be built up before the July and November peaks, and procurement should be aligned with the budget cycle.

### 5.3 Limitations

This study has several limitations. First, the results are based on a single distributor and a single region, so the generalizability is not established. Second, the analysis is descriptive and does not establish causality. Third, the syringe product group is a single product category, and the results may not generalize to other categories.

### 5.4 Future Work

Future work should evaluate the fiscal-cycle pattern in additional product categories and additional regions. A causal analysis of the drivers of syringe purchasing would strengthen the findings.

---

## 6. CONCLUSION

We tested the assumption that outbreaks drive medical device demand, using syringe purchasing in South Sumatra. Syringe purchase peaks recur in July and November, following the fiscal cycle, and the highest peak (July 2022, 709,779 units) occurred in the mid-year fiscal cycle rather than during the Omicron wave; November 2023 (an endemic month) also reached 460,191 units without an outbreak. The dengue and COVID tests are not significant. We conclude that the fiscal calendar determines syringe purchasing more than epidemiology, challenging the assumption that outbreaks drive medical device demand.

---

## ACKNOWLEDGMENTS

The authors thank PT Parit Panjang for providing the distribution data.

---

## CRediT AUTHOR STATEMENT

Khoirusy Syafaat: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing - original draft, Visualization. Herri Setiawan: Supervision, Writing - review and editing, Project administration.

---

## DATA AVAILABILITY STATEMENT

The data that support the findings of this study are available from PT Parit Panjang but restrictions apply to the availability of these data, which were used under license for the current study and so are not publicly available. Data are however available from the authors upon reasonable request and with permission of PT Parit Panjang. The derived syringe-level data and the outbreak test results are available in the project repository.

---

## FUNDING

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

---

## DECLARATION OF COMPETING INTERESTS

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## DECLARATION OF RELATED MANUSCRIPTS

This manuscript is a companion to a related study (Syafaat & Setiawan, 2025) that develops and evaluates the zero-inflated hybrid LSTM-GNN model for regional medical device sales forecasting. The two manuscripts share the same dataset but ask different questions. This manuscript isolates the syringe product group and tests the outbreak-driven demand assumption; the companion manuscript reports the overall model and its baselines. The authors declare this relationship to the editor.

---

## REFERENCES

- Syafaat, K., & Setiawan, H. (2025). Spatiotemporal prediction of medical device sales using hybrid LSTM-GNN in South Sumatra. *Companion manuscript*.
