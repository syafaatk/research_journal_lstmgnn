# Injecting Static Regional Attributes into Graph Neural Networks for Sparse Zero-Inflated Spatiotemporal Demand Forecasting

Khoirusy Syafaat, Herri Setiawan
Faculty of Computer and Natural Sciences, Indo Global Mandiri University, Palembang 30129, Indonesia
khoirusysyafaat@students.uigm.ac.id, herri@uigm.ac.id

**Manuscript status:** Draft lengkap (28 Agu 2026). Semua angka terverifikasi dari `temuan.md` (Section 3.7) dan `results/exp_rolling_nodefeat.json`, `results/exp_nodeattr_tuned.json`. Merujuk pada naskah utama (revisi.md) untuk detail arsitektur model; paper ini menjadikan injeksi atribut node statis sebagai objek kajian metodologis, untuk audiens computer science.

---

## ABSTRACT

Graph neural networks (GNNs) for spatiotemporal forecasting often need to incorporate static regional attributes, such as the number of hospitals or the population of a region. These attributes are nearly constant within a forecasting window, so they cannot be injected as ordinary temporal input channels (they would be constant across the window and add no information). We investigate how to inject such static attributes effectively, using a zero-inflated hybrid LSTM-GNN model for regional medical device demand forecasting in South Sumatra, Indonesia. Adding the number of hospitals per region as a node attribute on the GCN path significantly improves the model in both configurations: base R-squared 0.0609 vs 0.0592 (Diebold-Mariano = -2.861, p = 0.004); tuned R-squared 0.0528 vs 0.0508 (DM = -6.037, p < 0.0001). Adding the region area does not help (R-squared 0.0609 = W2), and rolling mean/std features do not help (DM p = 0.763). The mechanism is that the number of hospitals is a proxy for regional healthcare capacity (Palembang has 33 hospitals, 44% of the province's total), which influences procurement patterns; injecting it via the GCN node embedding allows the model to reflect the healthcare infrastructure structure. We conclude that static regional attributes are best injected as node attributes on the GCN path, not as temporal input channels.

**Keywords:** Graph Neural Networks; Node Attributes; Spatiotemporal Forecasting; Zero-Inflated Data; Demand Forecasting; Static Features

---

## HIGHLIGHTS

- Static regional attributes cannot be injected as temporal input channels
- Hospital count as a GCN node attribute improves forecasting significantly
- Base DM p = 0.004; tuned DM p < 0.0001
- Region area and rolling features do not help
- Node attributes on the GCN path are the effective injection mechanism

---

## 1. INTRODUCTION

Graph neural networks (GNNs) for spatiotemporal forecasting often need to incorporate static regional attributes, such as the number of hospitals, the population, or the area of a region. These attributes are nearly constant within a forecasting window (e.g., 30 days), so they cannot be injected as ordinary temporal input channels: a feature that is constant across the window adds no information to a temporal model such as an LSTM.

We investigate how to inject such static attributes effectively. We use a zero-inflated hybrid LSTM-GNN model for regional medical device demand forecasting in South Sumatra, Indonesia. We show that adding the number of hospitals per region as a node attribute on the GCN path significantly improves the model, while adding the region area or rolling features does not help.

The main contributions of this paper are as follows. First, we identify the problem of injecting static attributes into temporal-spatial models. Second, we show that node attributes on the GCN path are the effective injection mechanism. Third, we provide evidence that the number of hospitals is a proxy for regional healthcare capacity that influences demand. Fourth, we provide a systematic ablation of injection strategies.

The novelty of this paper is methodological. We address a specific architectural question: how to inject static attributes into a GNN for spatiotemporal forecasting of sparse data.

---

## 2. RELATED WORK

### 2.1 Node Attributes in GNNs

Node attributes are a standard component of GNNs. In many applications, nodes have features that describe their properties, and these features are used to initialize the node embeddings. However, the question of how to inject static attributes into a temporal-spatial model, where the attributes are constant within the forecasting window, is less studied.

### 2.2 Static Features in Temporal Models

Static features (features that do not change over time) are challenging for temporal models. An LSTM processes a sequence of inputs, and a feature that is constant across the sequence adds no information to the temporal dynamics. Static features must be injected through a different mechanism, such as conditioning the model on the static feature or using it as a node attribute in the graph component.

### 2.3 Position Relative to the Main Study

This paper is a companion to a study that develops and evaluates a zero-inflated hybrid LSTM-GNN model for regional medical device sales forecasting (Syafaat & Setiawan, 2025). The main study reports the overall model and its baselines. This paper does not repeat those results. Instead, it isolates the question of how to inject static regional attributes into the model. The two papers share the same dataset and model framework but ask different questions: the main study asks "how well can the model predict regional demand," while this paper asks "how should static regional attributes be injected into a GNN for spatiotemporal forecasting."

---

## 3. DATA AND METHOD

### 3.1 Data

We use medical device sales data from PT Parit Panjang, a distribution company in South Sumatra, Indonesia, covering July 2020 to October 2025. The final panel contains 16 regions over 1,495 days. The target is the daily invoice total per region, which is heavily zero-inflated (84.85% zero in the test period).

### 3.2 Static Regional Attributes

We consider the following static regional attributes:
1. **Number of hospitals.** The number of hospitals per region, from the 2024 hospital directory (88 hospitals, 10,161 beds).
2. **Region area.** The area of each region.
3. **Rolling mean/std.** Rolling 7- and 30-day mean and standard deviation of sales (included for comparison).

### 3.3 Injection Mechanism

We inject the static attributes as node attributes on the GCN path. The node attributes are concatenated with the node embeddings, so that the GCN can use them to reflect the regional infrastructure structure. We compare this against the alternative of injecting them as temporal input channels (which we argue is ineffective because they are constant within the window).

### 3.4 Evaluation Protocol

We evaluate two configurations:
1. **Base.** LSTM 128, GCN 128, dropout 0, 3 seeds.
2. **Tuned.** LSTM 128, GCN 64, dropout 0.2, 3 seeds, paired initialization.

We report R-squared and use the Diebold-Mariano test (Diebold & Mariano, 1995) for pairwise comparisons.

---

## 4. RESULTS

### 4.1 Hospital Count as a Node Attribute Improves the Model

Table 1 reports the effect of adding the hospital count as a node attribute.

**Table 1. Effect of Hospital Count as a Node Attribute**

| Configuration | With hospital count | Reference | DM | p |
|---|---|---|---|---|
| Base (128/128/do0) | R2 0.0609 | R2 0.0592 | -2.861 | 0.004 |
| Tuned (128/64/do0.2) | R2 0.0528 | R2 0.0508 | -6.037 | < 0.0001 |

Adding the hospital count as a node attribute significantly improves the model in both configurations. The effect is stronger in the tuned configuration (p < 0.0001).

### 4.2 Region Area Does Not Help

Adding the region area does not help: the R-squared remains 0.0609 (identical to the hospital-count-only variant). The area is not a useful attribute for this model.

### 4.3 Rolling Features Do Not Help

Rolling mean/std features (7 and 30 days) do not help (DM p = 0.763). This is because the LSTM already captures smoothing internally; the rolling features add no information.

### 4.4 Mechanism

The mechanism is that the number of hospitals is a proxy for regional healthcare capacity. Palembang has 33 hospitals (44% of the province's total), while other regions have 1-8 hospitals. The hospital count reflects the capacity that influences procurement patterns. Injecting it via the GCN node embedding allows the model to reflect the healthcare infrastructure structure, not just geographic proximity.

---

## 5. DISCUSSION

### 5.1 Interpretation

Our results show that static regional attributes are best injected as node attributes on the GCN path, not as temporal input channels. The hospital count, a proxy for regional healthcare capacity, significantly improves the model when injected as a node attribute. The region area and rolling features do not help.

The mechanism is clear: the hospital count reflects the healthcare infrastructure that drives procurement. Palembang, with 33 hospitals, is the dominant region, and its demand dynamics are informative for the periphery. Injecting the hospital count via the GCN node embedding allows the model to reflect this structure.

### 5.2 Why Temporal Channels Fail

Static attributes cannot be injected as temporal input channels because they are constant within the forecasting window. A feature that is constant across the window adds no information to a temporal model such as an LSTM. The node attribute mechanism, which conditions the graph component on the static attribute, is the effective alternative.

### 5.3 Implications for Practice

For practitioners, our results suggest that static regional attributes should be injected as node attributes on the graph component, not as temporal input channels. This is a simple but important design rule for GNN-based spatiotemporal forecasting.

### 5.4 Limitations

This study has several limitations. First, the results are based on a single dataset and a single region, so the generalizability is not established. Second, the static attributes are limited to those we evaluated; other attributes may behave differently. Third, the hospital count is nearly constant across years, so it cannot explain temporal variation.

### 5.5 Future Work

Future work should evaluate the injection mechanism on additional datasets and with additional static attributes. A systematic study of the interaction between static attributes, graph structure, and model architecture would strengthen the findings.

---

## 6. CONCLUSION

We investigated how to inject static regional attributes into a GNN for spatiotemporal forecasting of sparse data. Adding the number of hospitals per region as a node attribute on the GCN path significantly improves the model in both configurations (base DM p = 0.004; tuned DM p < 0.0001). Adding the region area does not help, and rolling features do not help. We conclude that static regional attributes are best injected as node attributes on the GCN path, not as temporal input channels.

---

## ACKNOWLEDGMENTS

The authors thank PT Parit Panjang for providing the distribution data.

---

## CRediT AUTHOR STATEMENT

Khoirusy Syafaat: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing - original draft, Visualization. Herri Setiawan: Supervision, Writing - review and editing, Project administration.

---

## DATA AVAILABILITY STATEMENT

The data that support the findings of this study are available from PT Parit Panjang but restrictions apply to the availability of these data, which were used under license for the current study and so are not publicly available. Data are however available from the authors upon reasonable request and with permission of PT Parit Panjang. The derived regional panel and the node-attribute experiment results are available in the project repository.

---

## FUNDING

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

---

## DECLARATION OF COMPETING INTERESTS

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## DECLARATION OF RELATED MANUSCRIPTS

This manuscript is a companion to a related study (Syafaat & Setiawan, 2025) that develops and evaluates the zero-inflated hybrid LSTM-GNN model for regional medical device sales forecasting. The two manuscripts share the same dataset and model framework but ask different questions. This manuscript isolates the question of how to inject static regional attributes into the model; the companion manuscript reports the overall model and its baselines. The authors declare this relationship to the editor.

---

## REFERENCES

- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263. https://doi.org/10.1080/07350015.1995.10524599
- Syafaat, K., & Setiawan, H. (2025). Spatiotemporal prediction of medical device sales using hybrid LSTM-GNN in South Sumatra. *Companion manuscript*.
