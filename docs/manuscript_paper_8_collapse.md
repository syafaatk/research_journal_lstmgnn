# The Zero-Inflated Classifier Collapse: When Majority-Class Dominance Silences Sparse Regions in Demand Forecasting

Khoirusy Syafaat, Herri Setiawan
Faculty of Computer and Natural Sciences, Indo Global Mandiri University, Palembang 30129, Indonesia
khoirusysyafaat@students.uigm.ac.id, herri@uigm.ac.id

**Manuscript status:** Draft lengkap (28 Agu 2026). Semua angka terverifikasi dari `results/exp_zi_fix_final.json`, `results/exp_zi_fix_v3.json`, dan `temuan.md` Section 7. Merujuk pada naskah utama (revisi.md) untuk detail arsitektur model; paper ini menjadikan perilaku model di bawah class imbalance sebagai objek kajian diagnostik.

---

## ABSTRACT

Zero-inflated (ZI) forecasting models, which combine a binary classifier with a regression head, are widely used for sparse demand data. We identify and characterize a failure mode we call the zero-inflated classifier collapse: when the target is dominated by zeros, the binary cross-entropy loss is dominated by the majority class, and the classifier converges to predicting non-zero only for the region with the highest non-zero rate, silencing all sparse regions. Using a real medical device demand dataset from South Sumatra, Indonesia (16 regions, 84.85% zero test targets), we show that the classifier never outputs a non-zero probability above 0.5 for 15 of 16 regions across all 231 test days; only Palembang, the dominant region, is gated to non-zero. Per-region R-squared is negative for all regions including Palembang (-0.3336), the model under-predicts total test sales by 78%, and its RMSE (8.16M IDR) is worse than the training-mean baseline (7.66M IDR). The pooled R-squared of 0.0523 is driven almost entirely by correctly predicting zeros (3,136 of 3,696 region-days). We evaluate mitigation strategies: weighted binary cross-entropy corrects the collapse but over-predicts (R-squared -0.71); threshold tuning does not help; training the regression head on all days (full-regression) improves the isolated ensemble R-squared from 0.0367 to 0.0494, but a full-pipeline evaluation produces contradictory results. We conclude that pooled R-squared can mask per-region failure, that the classification stage ("is there an order?") is a more reliable signal than the amount stage, and that ZI model evaluation should always report per-region and per-stage decompositions.

**Keywords:** Zero-Inflated Forecasting; Class Imbalance; Classifier Collapse; Demand Forecasting; Sparse Data; Deep Learning

---

## HIGHLIGHTS

- Zero-inflated classifier collapses under extreme class imbalance
- Classifier silences 15 of 16 sparse regions across all test days
- Pooled R-squared masks per-region failure in sparse forecasting
- Weighted loss and threshold tuning do not resolve the collapse
- Classification stage is a more reliable signal than the amount stage

---

## 1. INTRODUCTION

Sparse regional demand data are common in distribution and healthcare supply chains. On most days, most regions record no sales at all. Standard regression models trained on such data collapse toward the zero mode and fail to capture the magnitude of demand on active days. A widely used remedy is the zero-inflated (ZI) formulation, which treats the target as a mixture of a point mass at zero and a positive distribution (Lambert, 1992). In deep learning, this is implemented as a two-stage model: a binary classifier predicts whether a sale occurs, and a regression head predicts the sales amount conditional on a sale being predicted.

The ZI formulation is attractive because it separates two distinct questions: "is there an order?" and "how large is the order?" However, it introduces a subtle failure mode that is rarely discussed. When the target is dominated by zeros, the binary cross-entropy loss is dominated by the majority class. The classifier can achieve a low loss by predicting "no sale" almost everywhere, and it has little incentive to correctly identify the sparse non-zero days. In the extreme, the classifier collapses to predicting non-zero only for the region with the highest non-zero rate, silencing all sparse regions regardless of the regression head's output.

This paper characterizes this failure mode, which we call the zero-inflated classifier collapse, using real medical device demand data from a distributor in South Sumatra, Indonesia. We show that the collapse is not merely a theoretical concern: in our data, the classifier never outputs a non-zero probability above 0.5 for 15 of 16 regions across all 231 test days. We then evaluate mitigation strategies and show that the standard remedies (weighted loss, threshold tuning) do not resolve the underlying problem, while a full-regression variant improves the isolated ensemble but produces contradictory results in a full-pipeline evaluation.

The main contributions of this paper are as follows. First, we identify and formally characterize the zero-inflated classifier collapse as a distinct failure mode. Second, we show that pooled R-squared can mask per-region failure, and that per-region and per-stage decompositions are necessary for honest evaluation. Third, we evaluate mitigation strategies (weighted BCE, threshold tuning, full-regression) and report which work and which do not. Fourth, we provide a diagnostic protocol that practitioners can apply to detect the collapse in their own ZI models.

The novelty of this paper is diagnostic rather than algorithmic. We do not propose a new model; we characterize a failure mode of an existing and widely used formulation, and we provide evidence and a protocol for detecting and mitigating it. This is a methodological contribution relevant to the forecasting and applied machine learning communities, which frequently report pooled R-squared without per-region or per-stage decomposition.

---

## 2. RELATED WORK

### 2.1 Zero-Inflated Modeling

Zero-inflated modeling has a long tradition in count and demand forecasting (Lambert, 1992). The standard formulation is a mixture of a point mass at zero and a positive distribution, estimated jointly. In deep learning, the two-stage formulation (classifier + regression) is a natural extension, and it is widely used for sparse demand, sales, and traffic data. The two stages are typically trained jointly or sequentially, with the regression head trained only on non-zero days (masked regression).

### 2.2 Class Imbalance in Deep Learning

Class imbalance is a well-studied problem in classification. When one class dominates, models tend to predict the majority class, and standard accuracy metrics become uninformative. Common remedies include class weighting (weighted loss), resampling, and threshold tuning (He & Garcia, 2009). In the ZI forecasting setting, the "positive" class (a sale occurs) is the minority class, and the classifier is subject to the same imbalance problem. However, the consequences are more severe than in standard classification, because the classifier output gates the regression head: if the classifier predicts "no sale," the regression output is discarded regardless of its value.

### 2.3 Evaluation of Sparse Forecasting Models

A common practice in the forecasting literature is to report a single pooled R-squared for the entire test set. We argue that this can be misleading for sparse data. A model that predicts zero everywhere achieves a high pooled R-squared if most targets are zero, even if it fails completely on the sparse non-zero days. Per-region and per-stage decompositions are necessary to reveal whether the model is actually capturing signal or merely exploiting the zero structure. This paper provides a concrete example of this problem and a diagnostic protocol.

### 2.4 Position Relative to the Main Study

This paper is a companion to a study that develops and evaluates a zero-inflated hybrid LSTM-GNN model for regional medical device sales forecasting (Syafaat & Setiawan, 2025). The main study reports the overall model, its baselines, and its per-region performance. This paper does not repeat those results. Instead, it isolates the behavior of the ZI classifier under class imbalance and treats the collapse as the object of study, with diagnostics and mitigation experiments designed for that purpose. The two papers share the same dataset and model framework but ask different questions: the main study asks "how well can the model predict regional demand," while this paper asks "how does the ZI classifier behave under extreme class imbalance, and how can the collapse be detected and mitigated." This paper complements the proposal on aggregation and zero structure (which asks at what resolution the demand can be predicted) from the classification perspective.

---

## 3. DATA AND METHOD

### 3.1 Data

We use medical device sales data from PT Parit Panjang, a distribution company in South Sumatra, Indonesia, covering July 2020 to October 2025. The final panel contains 16 regions over 1,495 days, with 10,878 unique invoices after excluding regions outside South Sumatra. The target is the daily invoice total per region.

The target is heavily zero-inflated. In the test period (January to October 2025, 231 days), 3,136 of 3,696 region-day targets (84.85%) are zero. The fraction of zero days ranges from 3.1% (Palembang) to 99.9% (Musi Rawas Utara). The data are split chronologically: training (July 2020 to December 2023), validation (January to December 2024), and testing (January to October 2025).

### 3.2 Model Framework

We use the zero-inflated hybrid LSTM-GNN framework from the main study (Syafaat & Setiawan, 2025). The model has two output heads: a binary classifier (whether a sale occurs) and a regression head (the sales amount). The classifier is trained with binary cross-entropy; the regression head is trained on non-zero days (masked regression). A binary COVID-19 pandemic indicator (July 2020 to December 2022) is included as an exogenous input channel.

### 3.3 Diagnostic Protocol

To detect the classifier collapse, we compute, for each region, the number of true non-zero days, the number of days the classifier predicts non-zero (p > 0.5), and the mean and maximum classifier probability over the test set. A region is "silenced" if the classifier never outputs p > 0.5, meaning its regression output is always gated to zero.

We also decompose the pooled R-squared into its contributions: the correctly predicted zeros and the magnitude predictions for non-zero days. We compute per-region R-squared to reveal whether the model captures signal in any individual region.

### 3.4 Mitigation Experiments

We evaluate four mitigation strategies under identical conditions:

1. **Weighted binary cross-entropy (pos_weight = 4.1).** Up-weight the positive class to counter the majority-class dominance.
2. **Threshold tuning.** Sweep the classification threshold on the validation set to find the operating point that balances precision and recall.
3. **Full-regression.** Train the regression head on all days (not masked to non-zero only), allowing it to learn the overall data distribution.
4. **No ZI gate.** Regression only, without the classifier gate, to quantify the necessity of the ZI formulation.

---

## 4. RESULTS

### 4.1 The Classifier Collapse

Table 1 reports the classifier behavior per region on the test set. The classifier never outputs a non-zero probability above 0.5 for 15 of 16 regions across all 231 test days. Only Palembang, the dominant region with a 96.5% non-zero rate in the training set, is gated to non-zero.

**Table 1. Classifier Behavior Per Region on the Test Set** (231 days)

| Region | True non-zero days | Predicted non-zero (p > 0.5) | Mean p | Max p |
|---|---|---|---|---|
| Palembang | 223 | 231 | 0.969 | 0.991 |
| OKU Timur | 72 | 0 | 0.264 | 0.381 |
| OKU | 66 | 0 | 0.245 | 0.403 |
| Lahat | 57 | 0 | 0.231 | 0.461 |
| All others | 0-28 | 0 | < 0.26 | < 0.32 |

The collapse is severe: even regions with 57-72 true non-zero days (24-31% of the test period) are completely silenced. The classifier's mean probability for these regions is far below 0.5, and its maximum probability never exceeds 0.461.

### 4.2 Pooled R-squared Masks Per-Region Failure

The pooled R-squared of 0.0523 is driven almost entirely by correctly predicting zeros (3,136 of 3,696 region-days) and the magnitude predictions for Palembang. Per-region R-squared is negative for all regions, including Palembang (-0.3336). This means the amount stage does not outperform the within-region mean for any single region.

The model under-predicts total test sales by 78%, and its RMSE (8.16M IDR) is worse than the training-mean baseline (7.66M IDR). A threshold sweep from 0.1 to 0.7 produces an identical R-squared of 0.156, because the regression head also outputs near-zero values for non-Palembang regions, so changing the threshold does not change the final predictions.

### 4.3 Mitigation Results

Table 2 reports the mitigation experiments.

**Table 2. Mitigation Experiments** (R-squared and RMSE, 3-seed ensemble)

| Config | Description | R-squared | RMSE (IDR) | vs Baseline |
|---|---|---|---|---|
| A baseline | pw=1, masked reg, thr=0.5 | 0.0367 | 8,413,218 | -- |
| C1 wt+fullreg | pw=4.1, full-data reg, thr=0.5 | 0.0494 | 8,081,407 | +0.0127 (+35%) |
| C3 base+fullreg | pw=1.0, full-data reg, thr=0.5 | 0.0494 | 8,081,412 | +0.0127 (+35%) |
| B1 weighted | pw=4.1, masked reg, thr=0.5 | -0.7055 | 9,009,053 | -78x worse |
| D masked-only | regression only, no ZI | -60.28 | 11,421,107 | catastrophic |
| D2 full-only | fullreg only, no ZI gate | -0.0753 | 8,073,251 | worse |

The results reveal several important findings.

**Weighted BCE corrects the collapse but over-predicts.** With pos_weight = 4.1, the classifier now predicts non-zero for Lahat (160 days), OKU (206 days), and OKU Timur (227 days), correcting the collapse. However, this over-predicts non-zero days for sparse regions, producing false positives, and the R-squared degrades to -0.71. Correcting the collapse is not sufficient; it must be corrected without over-predicting.

**Threshold tuning does not help.** The optimal threshold on the validation set is either 0.80 (which cancels the effect of weighted BCE) or 0.15 (which opens all predictions, identical to full-regression). Neither resolves the underlying problem.

**Full-regression improves the isolated ensemble.** Training the regression head on all days (C1 and C3) improves the R-squared from 0.0367 to 0.0494, a 35% relative improvement, by allowing the regression head to learn the overall data distribution rather than only the non-zero mode. Notably, C1 (pw=4.1) and C3 (pw=1.0) are identical, indicating that the classifier weight does not affect the result; what matters is training the regression on all days.

**The ZI gate is structurally necessary.** Removing the ZI gate entirely (D, regression only) is catastrophic, with R-squared of -60.28. The ZI gating is structurally required; the problem is not the gate itself but the classifier's behavior under class imbalance.

### 4.4 Full-Pipeline Contradiction

Although full-regression improves the isolated ensemble R-squared (0.0367 to 0.0494), a full-pipeline evaluation produces contradictory results across models and ablations. The full-regression variant was therefore not adopted in the final model. This is an honest negative finding: the improvement observed in the isolated experiment does not transfer to the full pipeline, indicating that the ZI problem is not simply a matter of replacing the loss function.

---

## 5. DISCUSSION

### 5.1 Interpretation

The zero-inflated classifier collapse is a structural consequence of extreme class imbalance in the ZI formulation. When the target is 84.85% zero, the binary cross-entropy loss is dominated by the majority class, and the classifier has little incentive to identify the sparse non-zero days. The result is that sparse regions are silenced, and the pooled R-squared is driven by correctly predicting zeros rather than by capturing genuine signal.

This has a clear practical implication: the classification stage ("is there an order?") is a more reliable signal than the amount stage. For distribution planning, knowing whether an order will occur on a given day is often more valuable than predicting its exact amount, and the classifier provides this signal for the dominant region. However, the collapse means that this signal is unavailable for sparse regions, which are precisely the regions where planning is most uncertain.

### 5.2 Why Standard Remedies Fail

Our results show that the standard remedies for class imbalance do not resolve the collapse. Weighted BCE corrects the collapse but over-predicts, because up-weighting the positive class makes the classifier too eager to predict non-zero for sparse regions. Threshold tuning does not help, because the regression head also outputs near-zero values for sparse regions, so changing the threshold does not change the final predictions. Full-regression improves the isolated ensemble but produces contradictory results in the full pipeline.

These findings suggest that the collapse is not a simple tuning problem. It is a structural issue that requires a different formulation, such as soft gating, a mixture density network, or a threshold-free formulation. We identify these as directions for future work.

### 5.3 Implications for Evaluation

The most important methodological implication is that pooled R-squared can mask per-region failure. A model that achieves a reasonable pooled R-squared by predicting zeros everywhere can be completely useless for the sparse regions it is meant to serve. We recommend that ZI model evaluation always report:

1. Per-region R-squared, to reveal whether the model captures signal in any individual region.
2. Per-stage decomposition (classification vs regression), to attribute the pooled metric to its sources.
3. The classifier's per-region behavior (mean and max probability), to detect the collapse.

### 5.4 Limitations

This study has several limitations. First, the results are based on a single dataset and a single model framework, so the generalizability of the collapse to other settings is not established. Second, the mitigation experiments are limited to the strategies we evaluated; other approaches (soft gating, mixture density networks) were not implemented. Third, the full-pipeline contradiction was not fully resolved; we report it as an honest negative finding rather than claiming a definitive solution.

### 5.5 Future Work

Future work should evaluate the collapse on additional sparse datasets and with additional model frameworks. The mitigation strategies we identify as promising (soft gating, mixture density networks, threshold-free formulations) should be implemented and evaluated. A benchmark on public sparse datasets would strengthen the generalizability of the findings.

---

## 6. CONCLUSION

We identified and characterized the zero-inflated classifier collapse, a failure mode in which extreme class imbalance causes the ZI classifier to predict non-zero only for the dominant region, silencing all sparse regions. Using real medical device demand data, we showed that the classifier never outputs a non-zero probability above 0.5 for 15 of 16 regions, that pooled R-squared masks per-region failure, and that standard remedies (weighted BCE, threshold tuning) do not resolve the problem. We showed that the classification stage is a more reliable signal than the amount stage, and that ZI model evaluation should always report per-region and per-stage decompositions. We provide a diagnostic protocol for detecting the collapse and identify soft gating and mixture density networks as promising directions for future work.

---

## ACKNOWLEDGMENTS

The authors thank PT Parit Panjang for providing the distribution data.

---

## CRediT AUTHOR STATEMENT

Khoirusy Syafaat: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing - original draft, Visualization. Herri Setiawan: Supervision, Writing - review and editing, Project administration.

---

## DATA AVAILABILITY STATEMENT

The data that support the findings of this study are available from PT Parit Panjang but restrictions apply to the availability of these data, which were used under license for the current study and so are not publicly available. Data are however available from the authors upon reasonable request and with permission of PT Parit Panjang. The derived regional panel and the diagnostic results are available in the project repository.

---

## FUNDING

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

---

## DECLARATION OF COMPETING INTERESTS

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## DECLARATION OF RELATED MANUSCRIPTS

This manuscript is a companion to a related study (Syafaat & Setiawan, 2025) that develops and evaluates the zero-inflated hybrid LSTM-GNN model. The two manuscripts share the same dataset and model framework but ask different questions. This manuscript isolates the behavior of the ZI classifier under class imbalance; the companion manuscript reports the overall model and its baselines. The authors declare this relationship to the editor.

---

## REFERENCES

- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263.
- He, H., & Garcia, E. A. (2009). Learning from imbalanced data. *IEEE Transactions on Knowledge and Data Engineering*, 21(9), 1263-1284.
- Lambert, D. (1992). Zero-inflated Poisson regression. *Journal of the American Statistical Association*, 87(427), 427-432.
- Syafaat, K., & Setiawan, H. (2025). Spatiotemporal prediction of medical device sales using hybrid LSTM-GNN in South Sumatra. *Companion manuscript*.

---

## APPENDIX: Diagnostic Protocol for the Classifier Collapse

The following protocol detects the zero-inflated classifier collapse:

1. For each region, compute the number of true non-zero days in the test set.
2. For each region, compute the number of days the classifier predicts non-zero (p > 0.5), and the mean and maximum classifier probability.
3. A region is "silenced" if the classifier never outputs p > 0.5, meaning its regression output is always gated to zero.
4. Compute per-region R-squared to reveal whether the amount stage captures signal in any individual region.
5. Decompose the pooled R-squared into its contributions (correctly predicted zeros vs magnitude predictions for non-zero days).
6. Report the classifier's per-region behavior and the per-stage decomposition alongside the pooled metric.

This protocol is intentionally simple and can be applied to any ZI forecasting model. The key principle is that the pooled metric must be decomposed by region and by stage before any claim of predictive quality is made.
