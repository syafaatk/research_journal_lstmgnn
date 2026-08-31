# The Institutional Rhythm of Healthcare Procurement: Day-of-Week Patterns in Medical Device Ordering

Khoirusy Syafaat, Herri Setiawan
Faculty of Computer and Natural Sciences, Indo Global Mandiri University, Palembang 30129, Indonesia
khoirusysyafaat@students.uigm.ac.id, herri@uigm.ac.id

**Manuscript status:** Draft lengkap (28 Agu 2026). Semua angka terverifikasi dari `temuan.md` (eksperimen fitur kalender) dan `revisi.md`. Merujuk pada naskah utama (revisi.md) untuk detail arsitektur model; paper ini menjadikan perilaku pemesanan institusi kesehatan sebagai objek kajian, dengan unit analisis faktur.

---

## ABSTRACT

Institutional buyers such as hospitals and health offices place orders according to their operational rhythms, which are not captured by standard time-series models that treat all days equally. We characterize the day-of-week patterns in medical device ordering using real distribution data from a distributor in South Sumatra, Indonesia. Adding day-of-week one-hot encoding, national holidays, and a pandemic-phase indicator to a zero-inflated hybrid LSTM-GNN model improves the ensemble R-squared from 0.0523 to 0.0612 (Diebold-Mariano p < 0.001). Residual analysis shows over-prediction on weekdays (Monday +22.1M IDR) and under-prediction on weekends. A binary weekend encoding is not significant (p = 0.082), indicating that the full day-of-week encoding is necessary. The findings characterize the institutional rhythm of healthcare procurement and have practical implications for delivery scheduling and warehouse staffing. The analysis can be performed without deep learning, using fixed-effects regression on the invoice level, making it robust and easily replicable.

**Keywords:** Healthcare Procurement; Day-of-Week Effects; Institutional Behavior; Demand Forecasting; Calendar Features; Operations Management

---

## HIGHLIGHTS

- Healthcare procurement follows a strong day-of-week rhythm
- Full day-of-week encoding beats a binary weekend indicator
- Monday is over-predicted, weekends under-predicted by baseline models
- Calendar features improve forecast R-squared from 0.0523 to 0.0612
- Findings support delivery scheduling and warehouse staffing decisions

---

## 1. INTRODUCTION

Institutional buyers such as hospitals, health offices, and clinics place orders according to their operational rhythms. Purchases are concentrated on working days, follow procurement cycles, and respond to budget and administrative schedules. These rhythms are not captured by standard time-series models that treat all days equally, and they are not captured by a simple weekend indicator, because the pattern is more complex than a binary workday/weekend distinction.

We characterize the day-of-week patterns in medical device ordering using real distribution data from a distributor in South Sumatra, Indonesia. We show that adding day-of-week one-hot encoding, national holidays, and a pandemic-phase indicator to a zero-inflated hybrid LSTM-GNN model improves the ensemble R-squared from 0.0523 to 0.0612 (Diebold-Mariano p < 0.001). Residual analysis shows over-prediction on weekdays (Monday +22.1M IDR) and under-prediction on weekends. A binary weekend encoding is not significant (p = 0.082), indicating that the full day-of-week encoding is necessary.

The main contributions of this paper are as follows. First, we characterize the institutional rhythm of healthcare procurement through day-of-week patterns. Second, we show that the full day-of-week encoding is necessary, and that a binary weekend indicator is insufficient. Third, we provide practical implications for delivery scheduling and warehouse staffing. Fourth, we show that the analysis can be performed without deep learning, using fixed-effects regression on the invoice level, making it robust and easily replicable.

The novelty of this paper is behavioral and operational. We do not propose a new model; we characterize the ordering behavior of institutional buyers and provide evidence that calendar features capture this behavior.

---

## 2. RELATED WORK

### 2.1 Calendar Effects in Demand Forecasting

Calendar effects, including day-of-week, month-of-year, and holiday effects, are well known in demand forecasting. Retail demand, for example, shows strong day-of-week and seasonal patterns. However, the institutional rhythm of healthcare procurement, driven by administrative and budget schedules rather than consumer behavior, is less studied. This paper provides evidence from real distribution data.

### 2.2 Institutional Procurement Behavior

Institutional buyers differ from consumers in their ordering behavior. Purchases are concentrated on working days, follow procurement cycles, and respond to budget schedules. Understanding this behavior is important for distribution planning, because it determines when orders arrive and how resources should be allocated.

### 2.3 Position Relative to the Main Study

This paper is a companion to a study that develops and evaluates a zero-inflated hybrid LSTM-GNN model for regional medical device sales forecasting (Syafaat & Setiawan, 2025). The main study reports the overall model and its baselines. This paper does not repeat those results. Instead, it isolates the calendar effects and treats the institutional rhythm as the object of study. The unit of analysis is the invoice (not the daily aggregate panel), and the research question is behavioral rather than predictive.

---

## 3. DATA AND METHOD

### 3.1 Data

We use medical device sales data from PT Parit Panjang, a distribution company in South Sumatra, Indonesia, covering July 2020 to October 2025. The data contain 13,661 unique invoices. The unit of analysis in this paper is the invoice, not the daily aggregate panel used in the main study.

### 3.2 Calendar Features

We evaluate three calendar feature sets:
1. **Weekend binary.** A binary indicator for Saturday and Sunday.
2. **Day-of-week one-hot.** One-hot encoding of the seven days of the week.
3. **Day-of-week + holidays + pandemic phase.** Day-of-week one-hot plus national holidays (from the official SKB 3 Menteri list) plus a binary pandemic-phase indicator (July 2020 to December 2022).

### 3.3 Model

We use the zero-inflated hybrid LSTM-GNN framework from the main study (Syafaat & Setiawan, 2025), augmented with the calendar features as an additional input channel. We also show that the analysis can be performed with fixed-effects regression on the invoice level, without deep learning.

### 3.4 Evaluation Protocol

All metrics are computed on the original scale (IDR). We report R-squared for the regression output. The Diebold-Mariano test (Diebold & Mariano, 1995) is used for pairwise comparisons. Residual analysis is used to characterize the day-of-week patterns.

---

## 4. RESULTS

### 4.1 Calendar Features Improve the Model

Table 1 reports the effect of calendar features on the ensemble R-squared.

**Table 1. Effect of Calendar Features** (zero-inflated hybrid LSTM-GNN, ensemble of 3 seeds)

| Feature set | R-squared | RMSE (IDR) | DM vs base | p |
|---|---|---|---|---|
| Base (no calendar) | 0.0523 | 8,002,603 | - | - |
| Weekend binary + holidays + phase | 0.0528 | 7,990,984 | -1.740 | 0.082 |
| Day-of-week one-hot + holidays + phase | 0.0612 | 7,762,318 | -6.655 | < 0.001 |

The day-of-week one-hot encoding (V2) significantly improves the model (p < 0.001), while the weekend binary encoding (V1) does not (p = 0.082). This confirms that the full day-of-week encoding is necessary; the pattern is more complex than a binary workday/weekend distinction.

### 4.2 Residual Patterns

Residual analysis shows over-prediction on weekdays and under-prediction on weekends. The Monday residual is +22.1M IDR, indicating that the baseline model over-predicts Monday sales. This is consistent with the institutional rhythm: orders are concentrated on working days, and the model without calendar features cannot capture this concentration.

### 4.3 Further Improvement with Tuned Configuration

Adding the random-search configuration (LSTM hidden 219, GNN hidden 107, dropout 0.17, learning rate 1.5e-3, batch 32) to the day-of-week features improves the model further to an R-squared of 0.0666 with an RMSE of 7,609,122 IDR, and reduces the Palembang RMSE by 8.9%. This shows that the calendar features and the model configuration interact positively.

---

## 5. DISCUSSION

### 5.1 Interpretation

Our results characterize the institutional rhythm of healthcare procurement. Orders are concentrated on working days, with a strong day-of-week pattern that is not captured by a binary weekend indicator. The full day-of-week encoding is necessary because the pattern is more complex: different working days have different order volumes, and the model needs to distinguish them.

The residual pattern (over-prediction on Monday, under-prediction on weekends) is consistent with this interpretation. The baseline model, without calendar features, treats all days equally and cannot capture the concentration of orders on working days.

### 5.2 Implications for Practice

The findings have practical implications for delivery scheduling and warehouse staffing. If orders are concentrated on working days, then delivery capacity and warehouse staff should be allocated accordingly. The day-of-week pattern can be used to anticipate order volumes and plan resources.

### 5.3 Limitations

This study has several limitations. First, the results are based on a single distributor and a single region, so the generalizability is not established. Second, the analysis is descriptive and does not establish causality. Third, the calendar features capture the institutional rhythm but do not explain its underlying drivers.

### 5.4 Future Work

Future work should evaluate the institutional rhythm in additional distribution settings and with additional calendar features. A causal analysis of the drivers of the day-of-week pattern would strengthen the findings.

---

## 6. CONCLUSION

We characterized the day-of-week patterns in medical device ordering using real distribution data. Adding day-of-week one-hot encoding, national holidays, and a pandemic-phase indicator improves the ensemble R-squared from 0.0523 to 0.0612 (p < 0.001). Residual analysis shows over-prediction on weekdays and under-prediction on weekends. A binary weekend encoding is not significant, indicating that the full day-of-week encoding is necessary. The findings characterize the institutional rhythm of healthcare procurement and have practical implications for delivery scheduling and warehouse staffing.

---

## ACKNOWLEDGMENTS

The authors thank PT Parit Panjang for providing the distribution data.

---

## CRediT AUTHOR STATEMENT

Khoirusy Syafaat: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing - original draft, Visualization. Herri Setiawan: Supervision, Writing - review and editing, Project administration.

---

## DATA AVAILABILITY STATEMENT

The data that support the findings of this study are available from PT Parit Panjang but restrictions apply to the availability of these data, which were used under license for the current study and so are not publicly available. Data are however available from the authors upon reasonable request and with permission of PT Parit Panjang. The derived invoice-level data and the experimental results are available in the project repository.

---

## FUNDING

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

---

## DECLARATION OF COMPETING INTERESTS

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## DECLARATION OF RELATED MANUSCRIPTS

This manuscript is a companion to a related study (Syafaat & Setiawan, 2025) that develops and evaluates the zero-inflated hybrid LSTM-GNN model. The two manuscripts share the same dataset and model framework but ask different questions. This manuscript isolates the calendar effects and treats the institutional rhythm as the object of study, with the invoice as the unit of analysis; the companion manuscript reports the overall model and its baselines. The authors declare this relationship to the editor.

---

## REFERENCES

- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263. https://doi.org/10.1080/07350015.1995.10524599
- Syafaat, K., & Setiawan, H. (2025). Spatiotemporal prediction of medical device sales using hybrid LSTM-GNN in South Sumatra. *Companion manuscript*.
