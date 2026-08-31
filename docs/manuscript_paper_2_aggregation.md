# Aggregation Destroys the Signal: Zero-Inflated Daily Forecasting of Sparse Regional Demand

Khoirusy Syafaat, Herri Setiawan
Faculty of Computer and Natural Sciences, Indo Global Mandiri University, Palembang 30129, Indonesia
khoirusysyafaat@students.uigm.ac.id, herri@uigm.ac.id

**Manuscript status:** Draft lengkap (28 Agu 2026). Semua angka terverifikasi dari `temuan.md` (eksperimen granularitas) dan `revisi.md` (Tabel 4 baselines). Merujuk pada naskah utama (revisi.md) untuk detail arsitektur model; paper ini menjadikan resolusi waktu dan formulasi zero-inflated sebagai objek kajian.

---

## ABSTRACT

Sparse regional demand data are often aggregated to monthly or weekly resolution to reduce the fraction of zeros, under the assumption that aggregation makes forecasting easier. We test this assumption using real medical device demand data from a distributor in South Sumatra, Indonesia (16 regions, 2020-2025). We compare monthly, weekly, and daily resolutions under several model families, including a zero-inflated hybrid LSTM-GNN, ARIMA, and XGBoost. The monthly panel achieves an R-squared of -0.394 and the weekly panel -0.343, while the daily zero-inflated formulation achieves +0.043 to +0.052. The fraction of zeros is 81.54% across the full panel and 84.85% in the test period. ARIMA and XGBoost do not outperform an always-zero baseline. The choice of temporal resolution and the zero-inflated formulation determine whether the model succeeds, and this effect is larger than the difference between model architectures (the architecture gain is only 0.005-0.01). We conclude that aggregation destroys the signal in sparse regional demand, and that daily zero-inflated forecasting is the appropriate formulation. The research question is "at what resolution can this demand be predicted," not "which model is best."

**Keywords:** Zero-Inflated Forecasting; Temporal Aggregation; Sparse Data; Demand Forecasting; Resolution; Deep Learning

---

## HIGHLIGHTS

- Monthly and weekly aggregation destroy signal in sparse demand data
- Daily zero-inflated formulation succeeds where aggregation fails
- Resolution choice matters more than model architecture
- ARIMA and XGBoost fail on zero-inflated regional demand
- Daily zero-inflated forecasting is the appropriate formulation

---

## 1. INTRODUCTION

Sparse regional demand data are common in distribution and healthcare supply chains. On most days, most regions record no sales at all. A common practical response is to aggregate the data to a coarser temporal resolution, such as monthly or weekly, under the assumption that aggregation reduces the fraction of zeros and makes forecasting easier. This assumption is rarely tested.

We test this assumption using real medical device demand data from a distributor in South Sumatra, Indonesia. We compare monthly, weekly, and daily resolutions under several model families, and we show that aggregation actively destroys the signal. The monthly panel achieves an R-squared of -0.394 and the weekly panel -0.343, while the daily zero-inflated formulation achieves +0.043 to +0.052. The choice of temporal resolution and the zero-inflated formulation determine whether the model succeeds, and this effect is larger than the difference between model architectures.

The main contributions of this paper are as follows. First, we provide a controlled comparison of temporal resolutions (monthly, weekly, daily) under several model families. Second, we show that aggregation destroys the signal in sparse regional demand, and that daily zero-inflated forecasting is the appropriate formulation. Third, we show that the resolution choice matters more than the model architecture. Fourth, we provide evidence that ARIMA and XGBoost fail on zero-inflated regional demand, confirming that the zero structure dominates the data.

The novelty of this paper is empirical and methodological. We do not propose a new model; we provide evidence about the resolution at which sparse regional demand can be predicted, and we show that the choice of resolution is more consequential than the choice of model.

---

## 2. RELATED WORK

### 2.1 Temporal Aggregation in Forecasting

Temporal aggregation is a common preprocessing step in forecasting. Aggregating to a coarser resolution can reduce noise and the fraction of zeros, but it also discards information about the timing of events. The trade-off between aggregation and information loss is well known in time-series analysis, but it is rarely studied systematically for sparse regional demand. This paper provides a controlled comparison.

### 2.2 Zero-Inflated Modeling

Zero-inflated modeling treats the target as a mixture of a point mass at zero and a positive distribution (Lambert, 1992). In deep learning, this is implemented as a two-stage formulation: a binary classifier predicts whether a sale occurs, and a regression model predicts the amount conditional on a sale being predicted. The zero-inflated formulation is essential for sparse data, because standard regression models collapse toward the zero mode.

### 2.3 Position Relative to the Main Study

This paper is a companion to a study that develops and evaluates a zero-inflated hybrid LSTM-GNN model for regional medical device sales forecasting (Syafaat & Setiawan, 2025). The main study reports the overall model and its baselines. This paper does not repeat those results. Instead, it isolates the question of temporal resolution and asks at what resolution the demand can be predicted. The two papers share the same dataset and model framework but ask different questions: the main study asks "how well can the model predict regional demand," while this paper asks "at what resolution can this demand be predicted." This paper complements the proposal on the zero-inflated classifier collapse (which asks how the classifier behaves under class imbalance) from the resolution perspective.

---

## 3. DATA AND METHOD

### 3.1 Data

We use medical device sales data from PT Parit Panjang, a distribution company in South Sumatra, Indonesia, covering July 2020 to October 2025. The final panel contains 16 regions over 1,495 days, with 10,878 unique invoices after excluding regions outside South Sumatra. The target is the daily invoice total per region.

The target is heavily zero-inflated. Across the full panel, 81.54% of daily region-level targets are zero; in the test period, 84.85% are zero. The fraction of zero days ranges from 3.1% (Palembang) to 99.9% (Musi Rawas Utara).

### 3.2 Temporal Resolutions

We compare three temporal resolutions:
1. **Monthly.** Targets aggregated to monthly totals per region.
2. **Weekly.** Targets aggregated to weekly totals per region.
3. **Daily.** Targets kept at daily resolution, with a zero-inflated formulation.

### 3.3 Model Families

We evaluate several model families under each resolution:
1. **Zero-inflated hybrid LSTM-GNN.** The model from the main study, with a binary classifier and a regression head.
2. **ARIMA.** Per-region ARIMA with order selected by AIC.
3. **XGBoost.** Gradient-boosted regression trees.
4. **Always-zero.** A baseline that predicts zero everywhere, informative because the target is mostly zero.

### 3.4 Evaluation Protocol

All metrics are computed on the original scale (IDR). We report R-squared for the regression output. Because the target is mostly zero, MAPE is not reported. The Diebold-Mariano test (Diebold & Mariano, 1995) is used for pairwise comparisons.

---

## 4. RESULTS

### 4.1 Aggregation Destroys the Signal

Table 1 reports the R-squared for each resolution and model family.

**Table 1. R-squared by Temporal Resolution and Model Family**

| Resolution | R-squared |
|---|---|
| Monthly | -0.394 |
| Weekly | -0.343 |
| Daily (zero-inflated) | +0.043 to +0.052 |

The monthly panel achieves an R-squared of -0.394 and the weekly panel -0.343, both substantially negative, indicating that the models fail to capture the signal at these resolutions. The daily zero-inflated formulation achieves +0.043 to +0.052, a substantial improvement.

### 4.2 The Zero Structure Dominates

ARIMA and XGBoost do not outperform the always-zero baseline at any resolution. The always-zero baseline achieves an R-squared of 0.0059, reflecting the dominance of zero targets. ARIMA (R-squared -0.0775) and XGBoost (R-squared -0.0840) perform worse than always-zero, confirming that the zero-inflated structure dominates the data and that these models cannot capture signal beyond simple persistence.

### 4.3 Resolution Matters More Than Architecture

The choice of temporal resolution has a larger effect than the choice of model architecture. The gain from switching from monthly to daily zero-inflated is substantial (from -0.394 to +0.043), while the gain from switching between architectures within a resolution is only 0.005-0.01. This shows that the resolution choice is the dominant design decision.

---

## 5. DISCUSSION

### 5.1 Interpretation

Our results show that aggregation destroys the signal in sparse regional demand. The monthly and weekly panels fail, while the daily zero-inflated formulation succeeds. This is because aggregation discards the timing of events: a region that records one large order per month is indistinguishable from a region that records many small orders, and the zero structure is lost.

The zero-inflated formulation is essential because the target is mostly zero. Standard regression models collapse toward the zero mode and fail to capture the magnitude of demand on active days. The two-stage formulation (classifier + regression) separates the question of whether a sale occurs from the question of how large it is, and this separation is what allows the model to succeed.

### 5.2 Implications for Practice

For practitioners, our results suggest that sparse regional demand should be forecast at daily resolution with a zero-inflated formulation, rather than aggregated to a coarser resolution. Aggregation may seem like a reasonable way to reduce the fraction of zeros, but it destroys the signal. The daily zero-inflated formulation is the appropriate choice.

### 5.3 Limitations

This study has several limitations. First, the results are based on a single dataset and a single region, so the generalizability is not established. Second, the comparison is limited to the model families we evaluated; other models may behave differently. Third, the daily zero-inflated formulation requires a two-stage model, which is more complex than a single-stage regression.

### 5.4 Future Work

Future work should evaluate the resolution choice on additional sparse datasets and with additional model families. A systematic study of the interaction between resolution, zero fraction, and model architecture would strengthen the findings.

---

## 6. CONCLUSION

We tested whether aggregation to a coarser temporal resolution improves forecasting of sparse regional demand. Using real medical device demand data, we showed that monthly and weekly aggregation destroy the signal (R-squared -0.394 and -0.343), while the daily zero-inflated formulation succeeds (+0.043 to +0.052). We showed that the resolution choice matters more than the model architecture, and that ARIMA and XGBoost fail on zero-inflated regional demand. We conclude that sparse regional demand should be forecast at daily resolution with a zero-inflated formulation. The research question is "at what resolution can this demand be predicted," not "which model is best."

---

## ACKNOWLEDGMENTS

The authors thank PT Parit Panjang for providing the distribution data.

---

## CRediT AUTHOR STATEMENT

Khoirusy Syafaat: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing - original draft, Visualization. Herri Setiawan: Supervision, Writing - review and editing, Project administration.

---

## DATA AVAILABILITY STATEMENT

The data that support the findings of this study are available from PT Parit Panjang but restrictions apply to the availability of these data, which were used under license for the current study and so are not publicly available. Data are however available from the authors upon reasonable request and with permission of PT Parit Panjang. The derived regional panel and the experimental results are available in the project repository.

---

## FUNDING

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

---

## DECLARATION OF COMPETING INTERESTS

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## DECLARATION OF RELATED MANUSCRIPTS

This manuscript is a companion to a related study (Syafaat & Setiawan, 2025) that develops and evaluates the zero-inflated hybrid LSTM-GNN model. The two manuscripts share the same dataset and model framework but ask different questions. This manuscript isolates the question of temporal resolution; the companion manuscript reports the overall model and its baselines. The authors declare this relationship to the editor.

---

## REFERENCES

- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263. https://doi.org/10.1080/07350015.1995.10524599
- Lambert, D. (1992). Zero-inflated Poisson regression, with an application to defects in manufacturing. *Technometrics*, 34(1), 1-14. https://doi.org/10.1080/00401706.1992.10485228
- Syafaat, K., & Setiawan, H. (2025). Spatiotemporal prediction of medical device sales using hybrid LSTM-GNN in South Sumatra. *Companion manuscript*.
