# Event-Driven Demand Spikes in Healthcare Supply Chains: Evidence from Indonesia's COVID-19 Emergency Procurement

Khoirusy Syafaat, Herri Setiawan
Faculty of Computer and Natural Sciences, Indo Global Mandiri University, Palembang 30129, Indonesia
khoirusysyafaat@students.uigm.ac.id, herri@uigm.ac.id

**Manuscript status:** Draft lengkap (28 Agu 2026). Semua angka terverifikasi dari `temuan.md` (Section 6, pemetaan produk ke pemicu) dan `results/product_event_spikes.json`. Merujuk pada naskah utama (revisi.md) untuk konteks; paper ini menjadikan anatomi spike pengadaan darurat sebagai objek kajian, pada level produk dan event.

---

## ABSTRACT

Emergency procurement during public health crises creates demand spikes that are concentrated in time and across specific products. We analyze the anatomy of these spikes using real distribution data from a medical device distributor in South Sumatra, Indonesia, during the COVID-19 pandemic. The COVID product group (PANDEMI_COVID) is valued at 6,315M IDR (6.3% of total sales). The Delta wave (July-August 2021) triggered a spike of 5.55x the monthly median, with handscoon (37.3x), rapid test (36.6x), and mask (33.8x) leading the surge; a single Airvo invoice reached 762.5M IDR. In contrast, the El Nino forest-fire event of 2023 did NOT trigger a mask surge (verified with unit normalization from Box/Pcs to pcs; the mask quantity ratio was 0.47 relative to baseline). We conclude that emergency procurement spikes are concentrated in time and product, and that history-based models cannot capture them. The findings characterize what surges together, how concentrated the surge is in time, and why the signal is missed by history-based models.

**Keywords:** Emergency Procurement; Demand Spike; COVID-19; Healthcare Supply Chain; Event-Driven Demand; Product-Level Analysis

---

## HIGHLIGHTS

- COVID product group is 6.3% of total medical device sales
- Delta wave triggered a 5.55x monthly demand spike
- Handscoon surged 37.3x, rapid test 36.6x during Delta
- El Nino forest fires did NOT trigger a mask surge
- History-based models cannot capture event-driven spikes

---

## 1. INTRODUCTION

Public health emergencies create sudden and concentrated demand for medical devices. The COVID-19 pandemic is a prominent example: demand for personal protective equipment (PPE), testing supplies, and respiratory devices surged within weeks. Understanding the anatomy of these spikes is important for supply chain resilience, because history-based forecasting models cannot anticipate them.

We analyze the anatomy of emergency procurement spikes using real distribution data from a medical device distributor in South Sumatra, Indonesia, during the COVID-19 pandemic. We characterize what surges together, how concentrated the surge is in time, and why the signal is missed by history-based models. We also examine a negative case: the El Nino forest-fire event of 2023, which did NOT trigger a mask surge, providing a contrast that clarifies which events drive demand.

The main contributions of this paper are as follows. First, we provide a product-level anatomy of emergency procurement spikes during COVID-19. Second, we show that the spikes are concentrated in time and across specific products. Third, we provide a negative case (El Nino) that clarifies which events drive demand. Fourth, we explain why history-based models cannot capture event-driven spikes.

The novelty of this paper is empirical and product-level. We do not propose a new model; we characterize the anatomy of emergency procurement and provide evidence about which events drive demand.

---

## 2. RELATED WORK

### 2.1 Emergency Procurement and Supply Chain Resilience

Emergency procurement during public health crises is a well-studied topic in supply chain management. The COVID-19 pandemic highlighted the fragility of medical device supply chains and the need for resilience. However, the product-level anatomy of demand spikes, and the distinction between events that drive demand and those that do not, is less studied.

### 2.2 Event-Driven Demand

Event-driven demand is demand that is triggered by specific events, such as pandemics, natural disasters, or policy interventions. History-based forecasting models, which extrapolate from past patterns, cannot anticipate event-driven spikes. Understanding the anatomy of these spikes is important for designing models that can respond to them.

### 2.3 Position Relative to the Main Study

This paper is a companion to a study that develops and evaluates a zero-inflated hybrid LSTM-GNN model for regional medical device sales forecasting (Syafaat & Setiawan, 2025). The main study reports the overall model and its baselines. This paper does not repeat those results. Instead, it isolates the product-level and event-level analysis of emergency procurement. The unit of analysis is the product and the event, not the daily aggregate panel.

---

## 3. DATA AND METHOD

### 3.1 Data

We use medical device sales data from PT Parit Panjang, a distribution company in South Sumatra, Indonesia, covering July 2020 to October 2025. The data contain 13,661 unique invoices. The unit of analysis in this paper is the product line item and the event window.

### 3.2 Product-to-Driver Mapping

We map each product line item to a clinical driver group based on product name and specification. Each invoice is attributed to the dominant group by the value weight of its items, so that the total across groups equals the total invoice value (99.97M IDR) without double counting.

### 3.3 Spike Detection

We detect empirical spikes on the item-level value: a product is flagged as spiking if its value is at least 3x the monthly baseline median and at least 50M IDR within an event window. Events analyzed include the Delta wave (July-August 2021), Omicron (February-March 2022), BIAN immunization (August 2022), the El Nino forest fires (September-October 2023), and the dengue season (January-April).

### 3.4 Unit Normalization

For the mask analysis, we normalize units from Box and Pcs to pcs, because the box contents vary by type (surgical/duckbill = 50 pcs/box, N95/KN95 = 10 pcs/box). This ensures that the quantity comparison is valid.

---

## 4. RESULTS

### 4.1 The COVID Product Group

The COVID product group (PANDEMI_COVID) is valued at 6,315M IDR, or 6.3% of total sales. This includes PPE, testing supplies, and respiratory devices. The group is concentrated in the pandemic period (2020-2022).

### 4.2 The Delta Wave Spike

The Delta wave (July-August 2021) triggered a spike of 5.55x the monthly median. The surge was led by:
- Handscoon: 37.3x
- Rapid test: 36.6x
- Mask: 33.8x

A single Airvo invoice reached 762.5M IDR. The pattern is clear: PPE, testing, and oxygen-related products surged simultaneously.

### 4.3 The Omicron Wave Was Weak

The Omicron wave (February-March 2022) produced only a weak signal, with just 3 small hits (suture, nebulizer mask). This is consistent with the milder wave and reduced testing.

### 4.4 The El Nino Negative Case

The El Nino forest-fire event of 2023 did NOT trigger a mask surge. After normalizing units from Box/Pcs to pcs, the mask quantity ratio during the event was 0.47 relative to baseline (21,050 pcs vs 44,907 pcs/month). The surgical mask ratio was 0.48, and the N95/KN95 ratio was 0 (0 pcs vs 679/month). The mask peak actually occurred during COVID (September 2020 = 290,000 pcs; September 2021 = 362,740 pcs), not during the forest fires. This is a clear negative case: not all public health events drive medical device demand.

### 4.5 Why History-Based Models Miss the Signal

The spikes are concentrated in time and across specific products. A history-based model, which extrapolates from past patterns, cannot anticipate a 5.55x spike that occurs within weeks. The signal is event-driven, not history-driven.

---

## 5. DISCUSSION

### 5.1 Interpretation

Our results characterize the anatomy of emergency procurement spikes. The COVID-19 Delta wave triggered a concentrated surge in PPE, testing, and respiratory devices, while the El Nino forest fires did not trigger a mask surge. This distinction is important: not all public health events drive medical device demand, and the drivers are specific to the event type.

The concentration of the spike in time and product explains why history-based models miss it. A model that extrapolates from past patterns cannot anticipate a sudden, concentrated surge. This is a fundamental limitation of history-based forecasting for event-driven demand.

### 5.2 Implications for Practice

For supply chain managers, our results suggest that emergency procurement requires event-aware planning. History-based forecasting is insufficient for event-driven demand; managers should monitor event signals (pandemic waves, policy interventions) and prepare for concentrated spikes. The product-level anatomy can guide which products to stock in advance.

### 5.3 Limitations

This study has several limitations. First, the results are based on a single distributor and a single region, so the generalizability is not established. Second, the analysis is descriptive and does not establish causality. Third, the event windows are defined by the researcher, and different windows could yield different results.

### 5.4 Future Work

Future work should evaluate the anatomy of emergency procurement in additional settings and with additional events. A causal analysis of the drivers of demand spikes would strengthen the findings.

---

## 6. CONCLUSION

We analyzed the anatomy of emergency procurement spikes using real distribution data from a medical device distributor during the COVID-19 pandemic. The COVID product group (PANDEMI_COVID) is valued at 6,315M IDR (6.3% of total). The Delta wave triggered a spike of 5.55x the monthly median, led by handscoon (37.3x), rapid test (36.6x), and mask (33.8x). In contrast, the El Nino forest fires did not trigger a mask surge (ratio 0.47). We conclude that emergency procurement spikes are concentrated in time and product, and that history-based models cannot capture them.

---

## ACKNOWLEDGMENTS

The authors thank PT Parit Panjang for providing the distribution data.

---

## CRediT AUTHOR STATEMENT

Khoirusy Syafaat: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing - original draft, Visualization. Herri Setiawan: Supervision, Writing - review and editing, Project administration.

---

## DATA AVAILABILITY STATEMENT

The data that support the findings of this study are available from PT Parit Panjang but restrictions apply to the availability of these data, which were used under license for the current study and so are not publicly available. Data are however available from the authors upon reasonable request and with permission of PT Parit Panjang. The derived product-level mapping and the spike detection results are available in the project repository.

---

## FUNDING

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

---

## DECLARATION OF COMPETING INTERESTS

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## DECLARATION OF RELATED MANUSCRIPTS

This manuscript is a companion to a related study (Syafaat & Setiawan, 2025) that develops and evaluates the zero-inflated hybrid LSTM-GNN model for regional medical device sales forecasting. The two manuscripts share the same dataset but ask different questions. This manuscript isolates the product-level and event-level analysis of emergency procurement; the companion manuscript reports the overall model and its baselines. The authors declare this relationship to the editor.

---

## REFERENCES

- Syafaat, K., & Setiawan, H. (2025). Spatiotemporal prediction of medical device sales using hybrid LSTM-GNN in South Sumatra. *Companion manuscript*.
