# Infrastructure, Not Budgets: Cross-Sectional and Temporal Determinants of Regional Medical Device Demand

Khoirusy Syafaat, Herri Setiawan
Faculty of Computer and Natural Sciences, Indo Global Mandiri University, Palembang 30129, Indonesia
khoirusysyafaat@students.uigm.ac.id, herri@uigm.ac.id

**Manuscript status:** Draft lengkap (28 Agu 2026). Semua angka terverifikasi dari `temuan.md` (Section 2.4-2.6, 3.7) dan `balasan_reviewer.md`. Merujuk pada naskah utama (revisi.md) untuk konteks; paper ini menjadikan determinan permintaan alkes regional sebagai objek kajian kebijakan, tanpa machine learning.

---

## ABSTRACT

Regional medical device demand is commonly assumed to follow regional budgets: regions that spend more on goods and services should buy more medical devices. We test this assumption using real distribution data from a medical device distributor in South Sumatra, Indonesia (16 regions, 2020-2025), combined with official statistics on healthcare infrastructure and regional budgets. Four layers of evidence are consistent. First, the number of hospitals correlates with 2024 sales (r = 0.669, p = 0.005) and with the 2025 model residual (r = 0.697, p = 0.003), surviving the exclusion of Palembang (r = 0.596) and controlling for population (partial r = 0.648). Second, bed capacity (r = 0.577) and patient volume (r = 0.522) are consistent. Third, injecting the hospital count as a node attribute in a graph neural network significantly improves prediction (base DM p = 0.004; tuned DM p < 0.0001), providing predictive rather than merely correlational evidence. Fourth, in contrast, goods-and-services expenditure is only a between-region scale effect (pooled r = 0.345 significant; within-region r = 0.103 not significant; official APBD 2025 r = 0.256, p = 0.338). We conclude that regional medical device demand follows healthcare infrastructure capacity, not the annual budget cycle, with a methodological warning about the pooled-correlation trap across regions.

**Keywords:** Medical Device Demand; Healthcare Infrastructure; Regional Budget; Decentralization; Health Policy; Procurement

---

## HIGHLIGHTS

- Hospital count correlates with regional medical device sales (r = 0.669)
- Hospital count predicts model residuals (r = 0.697) and improves GNN prediction
- Goods-and-services expenditure is only a between-region scale effect
- Within-region budget variation does not predict sales
- Demand follows infrastructure capacity, not the annual budget cycle

---

## 1. INTRODUCTION

Regional medical device demand is commonly assumed to follow regional budgets. The reasoning is straightforward: regions that spend more on goods and services should buy more medical devices, because procurement is funded by the regional budget. This assumption underpins many policy discussions about health decentralization in Indonesia, where procurement decisions are made by regional health offices and hospitals.

We test this assumption using real distribution data from a medical device distributor in South Sumatra, Indonesia, combined with official statistics on healthcare infrastructure and regional budgets. We find that regional medical device demand follows healthcare infrastructure capacity (the number of hospitals, bed capacity, and patient volume), not the annual budget cycle. Goods-and-services expenditure is only a between-region scale effect: regions that are larger spend more and buy more, but within a region, year-to-year budget variation does not predict sales.

The main contributions of this paper are as follows. First, we provide empirical evidence, from real distribution data, that healthcare infrastructure, not budgets, drives regional medical device demand. Second, we provide a methodological warning about the pooled-correlation trap: a significant pooled correlation across regions can be a scale effect, not a temporal relationship. Third, we provide implications for health decentralization policy: medical device allocation should follow facility capacity, not the annual budget cycle.

The novelty of this paper is empirical and policy-relevant. We do not propose a new model; we provide evidence about the determinants of regional medical device demand, using a combination of distribution data, official statistics, and predictive evidence.

---

## 2. RELATED WORK

### 2.1 Determinants of Medical Device Demand

The determinants of medical device demand are not well studied with real distribution data. Most studies focus on the supply side (manufacturing, regulation) or on aggregate health spending. Predictive modeling of biomedical and healthcare temporal data has focused on forecasting methods rather than on the structural determinants of demand (Patharkar et al., 2024). This paper provides evidence from the demand side, using real distribution data from a medical device distributor.

### 2.2 Health Decentralization in Indonesia

Indonesia has decentralized health procurement, with decisions made by regional health offices and hospitals. The relationship between regional budgets and medical device demand is a key policy question. This paper provides evidence that infrastructure capacity, not budgets, drives demand.

### 2.3 The Pooled-Correlation Trap

A common methodological error is to interpret a significant pooled correlation across regions as evidence of a temporal relationship. In reality, a pooled correlation can be driven by between-region scale effects: larger regions spend more and buy more, producing a spurious correlation. This paper demonstrates this trap and shows how to avoid it by separating within-region from between-region variation.

### 2.4 Position Relative to the Main Study

This paper is a companion to a study that develops and evaluates a zero-inflated hybrid LSTM-GNN model for regional medical device sales forecasting (Syafaat & Setiawan, 2025). The main study reports the overall model and its baselines. This paper does not repeat those results. Instead, it isolates the question of what drives regional medical device demand, using econometric analysis without machine learning. The two papers share the same dataset but ask different questions: the main study asks "how well can the model predict regional demand," while this paper asks "what determines regional medical device demand."

---

## 3. DATA AND METHOD

### 3.1 Data

We use three data sources:
1. **Distribution data.** Medical device sales from PT Parit Panjang, a distributor in South Sumatra, covering July 2020 to October 2025 (16 regions, 10,878 invoices).
2. **Healthcare infrastructure.** The 2024 hospital performance directory (88 hospitals, 10,161 beds), including the number of hospitals, bed capacity, patient volume, and length of stay per region.
3. **Regional budgets.** Official APBD data per region from BPS (2020-2023) and the DJPK Kemenkeu portal (2024-2025), focusing on goods-and-services expenditure.

### 3.2 Variables

- **Outcome:** annual sales per region (2024) and the residual of the predictive model (2025).
- **Infrastructure predictors:** number of hospitals, total beds, total patient discharges, and length of stay.
- **Budget predictors:** goods-and-services expenditure per region (total and per capita).

### 3.3 Analysis

We use the following analyses:
1. **Cross-sectional correlation.** Correlation between infrastructure/budget predictors and sales across regions (n = 16).
2. **Pooled vs within-region correlation.** To separate between-region scale effects from within-region temporal relationships, we demean per region and recompute the correlation.
3. **Partial correlation.** Controlling for population.
4. **Sensitivity.** Excluding Palembang (n = 15) to check that the result is not driven by the dominant region.
5. **Predictive evidence.** Injecting the hospital count as a node attribute in a graph neural network (Wu et al., 2019) and testing whether it improves prediction (Diebold-Mariano test; Diebold & Mariano, 1995).

---

## 4. RESULTS

### 4.1 Infrastructure Correlates with Sales

Table 1 reports the cross-sectional correlation between infrastructure and 2024 sales.

**Table 1. Infrastructure vs 2024 Sales (cross-sectional, n = 16)**

| Factor | r | p | Conclusion |
|---|---|---|---|
| Number of hospitals | 0.669 | 0.005 | significant |
| Bed capacity | 0.577 | 0.019 | significant |
| Length of stay | 0.574 | 0.020 | significant |
| Patient discharges | 0.522 | 0.038 | significant |

The number of hospitals correlates with 2024 sales (r = 0.669, p = 0.005). Bed capacity (r = 0.577), length of stay (r = 0.574), and patient discharges (r = 0.522) are consistent. The result survives the exclusion of Palembang (r = 0.596, p = 0.019) and controlling for population (partial r = 0.648, p = 0.043).

### 4.2 Infrastructure Correlates with Model Residuals

The number of hospitals correlates with the 2025 model residual (r = 0.697, p = 0.003), and survives the exclusion of Palembang (r = 0.631, p = 0.012). This indicates that infrastructure explains variation that the history-based model does not capture.

### 4.3 Predictive Evidence

Injecting the hospital count as a node attribute in a graph neural network significantly improves prediction: base R-squared 0.0609 vs 0.0592 (DM = -2.861, p = 0.004); tuned R-squared 0.0528 vs 0.0508 (DM = -6.037, p < 0.0001). This provides predictive, not merely correlational, evidence that infrastructure capacity drives demand.

### 4.4 Budget Is Not a Predictor

Table 2 reports the budget analysis.

**Table 2. Goods-and-Services Expenditure vs Sales**

| Analysis | r | p | Conclusion |
|---|---|---|---|
| Pooled 2020-2025 | 0.345 | 0.0006 | significant (scale effect) |
| Within-region (demeaned) | 0.103 | 0.318 | not significant |
| APBD 2025 DJPK per region | 0.256 | 0.338 | not significant |

The pooled correlation is significant (r = 0.345, p = 0.0006), but this is a between-region scale effect: larger regions spend more and buy more. The within-region correlation (demeaned) is not significant (r = 0.103, p = 0.318), and the official APBD 2025 correlation is not significant (r = 0.256, p = 0.338). The provincial health office budget moves in the opposite direction to sales (2022-2024) due to central recording mechanisms.

---

## 5. DISCUSSION

### 5.1 Interpretation

Our results show that regional medical device demand follows healthcare infrastructure capacity, not the annual budget cycle. The number of hospitals, bed capacity, and patient volume correlate with sales, and the hospital count improves prediction when injected into a graph neural network. In contrast, goods-and-services expenditure is only a between-region scale effect.

The mechanism is clear: hospitals are the largest buyers of medical devices. More hospitals and more patients mean more procurement. The infrastructure capacity reflects the actual demand for medical devices, while the budget reflects the general spending level, which is not specific to medical devices.

### 5.2 The Pooled-Correlation Trap

Our results demonstrate the pooled-correlation trap. A naive analysis would conclude that regional budgets drive medical device demand, because the pooled correlation is significant (r = 0.345). However, this is a between-region scale effect: larger regions spend more and buy more. When we separate within-region from between-region variation, the temporal relationship disappears. This is a methodological warning for policy analysis: a significant pooled correlation across regions is not evidence of a temporal relationship.

### 5.3 Implications for Policy

For health decentralization policy, our results suggest that medical device allocation should follow facility capacity, not the annual budget cycle. Regions with more hospitals and more patients need more medical devices, regardless of their budget level. This has implications for how procurement budgets are allocated and how medical device distribution is planned.

### 5.4 Limitations

This study has several limitations. First, the infrastructure data cover a single year (2024), so the evidence is structural rather than temporal. Second, the number of hospitals is nearly constant across years, so it cannot explain temporal variation. Third, the sample is small (n = 16) and many factors were tested, raising the risk of false positives. Fourth, the data come from a single distributor.

### 5.5 Future Work

Future work should evaluate the determinants of medical device demand in additional regions and with additional data sources. A temporal analysis of infrastructure capacity and demand would strengthen the findings.

---

## 6. CONCLUSION

We tested the assumption that regional medical device demand follows regional budgets. Using real distribution data combined with official statistics, we found that demand follows healthcare infrastructure capacity, not the annual budget cycle. The number of hospitals correlates with sales (r = 0.669) and with model residuals (r = 0.697), and improves prediction when injected into a graph neural network. In contrast, goods-and-services expenditure is only a between-region scale effect. We conclude that medical device allocation should follow facility capacity, and we provide a methodological warning about the pooled-correlation trap.

---

## ACKNOWLEDGMENTS

The authors thank PT Parit Panjang for providing the distribution data.

---

## CRediT AUTHOR STATEMENT

Khoirusy Syafaat: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing - original draft, Visualization. Herri Setiawan: Supervision, Writing - review and editing, Project administration.

---

## DATA AVAILABILITY STATEMENT

The data that support the findings of this study are available from PT Parit Panjang but restrictions apply to the availability of these data, which were used under license for the current study and so are not publicly available. Data are however available from the authors upon reasonable request and with permission of PT Parit Panjang. The official statistics (hospital directory, APBD) are publicly available from the cited sources. The derived regional panel and the analysis results are available in the project repository.

---

## FUNDING

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

---

## DECLARATION OF COMPETING INTERESTS

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## DECLARATION OF RELATED MANUSCRIPTS

This manuscript is a companion to a related study (Syafaat & Setiawan, 2025) that develops and evaluates the zero-inflated hybrid LSTM-GNN model for regional medical device sales forecasting. The two manuscripts share the same dataset but ask different questions. This manuscript isolates the question of what determines regional medical device demand, using econometric analysis without machine learning; the companion manuscript reports the overall model and its baselines. The authors declare this relationship to the editor.

---

## REFERENCES

- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263. https://doi.org/10.1080/07350015.1995.10524599
- Patharkar, A., Cai, F., Al-Hindawi, F., & Wu, T. (2024). Predictive modeling of biomedical temporal data in healthcare applications: Review and future directions. *Frontiers in Physiology*, 15, 1386760. https://doi.org/10.3389/fphys.2024.1386760
- Syafaat, K., & Setiawan, H. (2025). Spatiotemporal prediction of medical device sales using hybrid LSTM-GNN in South Sumatra. *Companion manuscript*.
- Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph WaveNet for deep spatial-temporal graph modeling. *IJCAI International Joint Conference on Artificial Intelligence*, 2019-August, 1907-1913. https://doi.org/10.24963/ijcai.2019/264
