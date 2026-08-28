# When Geography Lies: Distribution Network Topology Outperforms Spatial Proximity in Graph-Based Regional Demand Forecasting

Khoirusy Syafaat, Herri Setiawan
Faculty of Computer and Natural Sciences, Indo Global Mandiri University, Palembang 30129, Indonesia
khoirusysyafaat@students.uigm.ac.id, herri@uigm.ac.id

**Manuscript status:** Draft lengkap (28 Agu 2026). Semua angka terverifikasi dari eksperimen geocoded 16 region (`results/ablation_rerun_paired.json`, `results/exp_features.json`). Merujuk pada naskah utama (revisi.md) untuk detail arsitektur model; paper ini menjadikan pilihan konstruksi graf sebagai objek kajian tersendiri.

---

## ABSTRACT

Graph neural networks (GNNs) for spatiotemporal demand forecasting are typically built on graphs derived from geographic proximity, under the implicit assumption that nearby regions influence one another. Using real distribution data from a medical device distributor in South Sumatra, Indonesia (16 regions, July 2020 to October 2025, 10,878 invoices), we test whether the choice of graph construction determines forecasting quality more than the model architecture itself. We evaluate six graph constructions under an identical zero-inflated hybrid LSTM-GNN framework: identity adjacency, random graph, distance graph (Haversine k-nearest neighbors), sales-correlation graph, and the actual distribution network represented as a star centered on the distributor's hub in Palembang. The distribution-network star achieves the best R-squared of 0.0703, a 61% relative improvement over the distance graph (0.0436). The distance graph performs worse than both the identity adjacency (0.0538) and the random graph (0.0486), and the sales-correlation graph (0.0501) also underperforms identity. The maximum pairwise sales correlation between regions is only 0.332, indicating that the "natural" spatial structure is thin. These results show that the real distribution topology, not geographic proximity, carries the predictive signal in this setting. We conclude that for GNN-based demand forecasting in distribution businesses, the graph should encode how goods actually move, not where facilities are located. We provide a reproducible ablation protocol and discuss implications for graph design in sparse, zero-inflated regional demand data.

**Keywords:** Graph Neural Networks; Demand Forecasting; Distribution Network; Graph Construction; Zero-Inflated Data; Spatiotemporal Prediction; Medical Device Supply Chain

---

## HIGHLIGHTS

- Distribution-network star graph beats geographic distance in GNN forecasting
- Distance graph performs worse than identity and random graphs
- Geographic proximity adds no signal in sparse regional demand data
- Graph should encode how goods move, not where facilities are located
- Reproducible ablation protocol for graph construction selection

---

## 1. INTRODUCTION

Graph neural networks (GNNs) have become a standard tool for spatiotemporal forecasting, with applications in traffic flow, energy demand, and epidemic modeling (Luo et al., 2024; Santos et al., 2023; Sunder et al., 2024). A central design decision in any GNN-based forecasting system is the construction of the graph itself: which nodes are connected, and what the edges represent. In most applications, the graph is derived from geographic proximity, under the implicit assumption that nearby regions influence one another through shared infrastructure, population mobility, or similar demand drivers (Alourani et al., 2023; Heryati et al., 2024; Patharkar et al., 2024; Ye et al., 2024).

This assumption is reasonable in many settings, but it is not universal. In distribution businesses, the flow of goods is determined by logistics topology, not by geography. A distributor may serve a distant region directly from a central hub while a geographically adjacent region is served through a different route. When the underlying demand is driven by procurement decisions of institutional buyers (hospitals, health offices) rather than by consumer mobility, geographic adjacency may carry little predictive signal. In such cases, building the graph from geographic distance could mislead the model, propagating information between regions that do not actually interact.

The medical device distribution sector in developing regions is a particularly relevant case. Demand is sparse and zero-inflated: on most days, most regions record no sales at all. Purchases are driven by institutional procurement cycles and healthcare infrastructure rather than by continuous consumer demand. In this setting, the question of how to construct the graph is not a minor implementation detail; it may determine whether the spatial component of the model contributes anything at all.

This paper addresses the following research question: does the choice of graph construction determine forecasting quality in GNN-based regional demand forecasting, and does the actual distribution network topology outperform geographic proximity? We answer this question using real distribution data from a medical device distributor in South Sumatra, Indonesia, evaluated under an identical zero-inflated hybrid LSTM-GNN framework across six graph constructions.

The main contributions of this paper are as follows. First, we provide an empirical comparison of six graph constructions (identity, random, distance, sales-correlation, and distribution-network star) under a controlled experimental protocol. Second, we show that the distribution-network topology, not geographic proximity, carries the predictive signal, with the star graph achieving a 61% relative improvement over the distance graph. Third, we report an honest negative finding: the distance graph performs worse than both identity adjacency and a random graph, indicating that geographic proximity adds no signal in this setting. Fourth, we provide a reproducible ablation protocol that practitioners can apply to decide how to construct graphs for their own distribution data.

The novelty of this paper is contextual and methodological rather than algorithmic. The LSTM-GNN architecture itself is not new; the contribution lies in treating graph construction as the object of study and providing empirical evidence, from real distribution data, that logistics topology should guide graph design in sparse regional demand forecasting.

---

## 2. RELATED WORK

### 2.1 Graph Construction in Spatiotemporal Forecasting

GNN-based spatiotemporal forecasting models differ not only in architecture but also in how the graph is constructed. The most common approach is geographic distance: nodes are connected to their k nearest neighbors by Haversine or Euclidean distance, sometimes with a maximum-distance threshold (Li et al., 2018; Zheng et al., 2020). A second common approach is functional similarity, where edges are drawn between nodes whose historical signals are correlated (Wu et al., 2019). A third approach uses domain knowledge to define edges from the actual structure of the system, such as road networks in traffic forecasting or power grids in energy forecasting (Luo et al., 2024; Yang, Z. et al., 2023).

Despite the prevalence of distance-based graphs, relatively few studies systematically compare graph constructions under otherwise identical conditions. When such comparisons are performed, the results often show that the choice of graph matters substantially, and that the best construction depends on the domain (Wu et al., 2019). This paper contributes to this line of work by comparing graph constructions in a distribution setting where the actual logistics topology is known and can be encoded directly.

### 2.2 Zero-Inflated Demand Forecasting

Regional daily demand data are frequently zero-inflated: on most days, most regions record no activity. Standard regression models collapse toward the zero mode and fail to capture the magnitude of demand on active days. Zero-inflated modeling, which treats the target as a mixture of a point mass at zero and a positive distribution, has a long tradition in count and demand modeling (Lambert, 1992). In deep learning, this is implemented as a two-stage formulation: a binary classifier predicts whether a sale occurs, and a regression model predicts the amount conditional on a sale being predicted.

The zero-inflated structure interacts with graph construction in an important way. When most targets are zero, the spatial propagation of sales amounts has limited opportunity to improve predictions, because there is little non-zero signal to propagate. This makes the choice of graph even more consequential: a graph that connects regions with genuinely related non-zero activity can help, while a graph that connects regions with unrelated or absent activity adds noise. Our results are consistent with this view.

### 2.3 Position Relative to the Main Study

This paper is a companion to a study that develops and evaluates a zero-inflated hybrid LSTM-GNN model for regional medical device sales forecasting (Syafaat & Setiawan, 2025). The main study reports the overall model, its baselines, and its per-region performance. This paper does not repeat those results. Instead, it isolates the question of graph construction and treats the choice of graph as the object of study, with experiments designed specifically for that purpose. The two papers share the same dataset and model framework but ask different questions: the main study asks "how well can the model predict regional demand," while this paper asks "how should the graph be constructed for GNN-based demand forecasting in a distribution setting." We cross-reference the main study for architectural details and report only the ablation results here.

---

## 3. DATA AND METHOD

### 3.1 Data

We use medical device sales data from PT Parit Panjang, a distribution company in South Sumatra, Indonesia, covering July 2020 to October 2025. The raw data contain 13,661 unique invoices; after excluding regions outside South Sumatra, 10,878 invoices remain. The final panel contains 16 regions (regencies and cities) over 1,495 days. Musi Rawas Utara, which recorded only 2 invoices in the study period, was merged into Musi Rawas.

The target variable is the daily invoice total per region. The target is heavily zero-inflated: 81.54% of all daily region-level targets across the full panel are zero, and 84.85% in the test period. The fraction of zero days ranges from 3.1% (Palembang) to 99.9% (Musi Rawas Utara). Because of this structure, the forecasting problem is formulated as zero-inflated with two stages: a binary classifier that predicts whether a sale occurs, and a regression model that predicts the sales amount on non-zero days.

The data are split chronologically: training (July 2020 to December 2023), validation (January to December 2024), and testing (January to October 2025). The target is normalized with a log transformation fitted on the training set only, to prevent leakage. A binary COVID-19 pandemic indicator (July 2020 to December 2022) is included as an exogenous input channel.

### 3.2 Model Framework

We use the zero-inflated hybrid LSTM-GNN framework from the main study (Syafaat & Setiawan, 2025). Temporal dynamics are modeled with an LSTM, spatial dependencies with a Graph Convolutional Network (GCN), and both representations are combined through concatenation followed by a fully connected layer. The model has two output heads: a binary classifier (whether a sale occurs) and a regression head (the sales amount). The classifier is trained with binary cross-entropy; the regression head is trained on non-zero days (masked regression). The COVID-19 indicator is provided as a second input channel.

For the ablation experiments in this paper, we use the tuned configuration (LSTM hidden 128, GCN hidden 64, dropout 0.2) with a single seed (42), consistent with the ablation protocol of the main study. All graph variants are trained under identical conditions, differing only in the adjacency matrix.

### 3.3 Graph Constructions

We evaluate six graph constructions. All graphs are undirected, static, and defined over the 16 regions.

1. **Identity adjacency.** Each node is connected only to itself; the graph structure is effectively removed while the model parameters are kept. This serves as the reference for "no spatial structure."

2. **Random graph.** Edges are drawn at random (5 permutations, mean reported). This serves as a control for the effect of adding edges without meaningful structure.

3. **Distance graph.** Nodes are connected to their k = 3 nearest neighbors by Haversine distance, with a maximum-distance threshold of 104.66 km. This is the graph used in the main study. The threshold equals the maximum observed 3-nearest-neighbor distance, so it does not cut any candidate edge.

4. **Sales-correlation graph.** Nodes are connected to their k = 3 most correlated neighbors based on Pearson correlation of training-period sales, with a threshold of 0.0126 (67 edges). The correlation is computed on training data only, to avoid leakage.

5. **Distribution-network graph (star).** The company confirmed that all delivery routes originate from its distribution center in Palembang (Komplek Villa Kenten Blok G No. 2). The distribution network is therefore represented as a star graph in which Palembang is connected to all other 15 regions.

6. **Without COVID-19 indicator.** The distance graph with the COVID-19 channel removed, to isolate the contribution of the exogenous indicator.

### 3.4 Evaluation Protocol

All metrics are computed on the original scale (IDR). We report RMSE, MAE, and R-squared for the regression output. Because the target is 84.85% zero, MAPE is not reported. The Diebold-Mariano test (Diebold & Mariano, 1995) is used for pairwise comparisons of predictive accuracy. The word "significant" is used only when the Diebold-Mariano test rejects the null hypothesis at p < 0.05.

---

## 4. RESULTS

### 4.1 Graph Construction Determines Forecasting Quality

Table 1 reports the ablation results for the six graph constructions under the identical tuned configuration.

**Table 1. Ablation Results Across Graph Constructions** (HybridTuned configuration: LSTM 128, GCN 64, dropout 0.2, seed 42)

| Graph construction | RMSE (IDR) | MAE (IDR) | R-squared |
|---|---|---|---|
| Identity adjacency | 7,964,173 | 1,871,263 | 0.0538 |
| Random graph (mean of 5 permutations) | 8,101,548 | 1,906,271 | 0.0486 |
| Distance graph k=3 (proposed in main study) | 8,235,956 | 1,941,944 | 0.0436 |
| Sales-correlation graph k=3 (training-only) | 8,063,890 | 1,895,385 | 0.0501 |
| **Distribution network (star from Palembang)** | **7,504,506** | **1,806,491** | **0.0703** |
| Without COVID-19 indicator | 8,262,716 | 1,949,778 | 0.0425 |

The choice of graph structure has a substantial effect on prediction quality. The distribution-network star achieves the best R-squared of 0.0703, a 61% relative improvement over the distance graph (0.0436). The star graph also achieves the lowest RMSE (7,504,506 IDR) and MAE (1,806,491 IDR) among all variants.

### 4.2 Geographic Proximity Adds No Signal

The distance graph (R-squared 0.0436) performs worse than both the identity adjacency (0.0538) and the random graph (0.0486). This is a striking and honest negative finding: adding geographic edges to the model makes it worse than having no spatial structure at all, and worse than adding random edges. Geographic proximity itself does not improve prediction in the daily zero-inflated setting.

The sales-correlation graph (0.0501) also underperforms the identity adjacency, indicating that neither distance nor sales correlation carries useful spatial signal. The maximum pairwise sales correlation between regions is only 0.332, confirming that the "natural" spatial structure of the data is thin.

Table 2 reports the Diebold-Mariano tests comparing each graph construction against the distance graph as the reference. Both the identity adjacency (DM = -8.041, p < 0.001) and the correlation graph (DM = -8.180, p < 0.001) significantly outperform the distance graph, confirming that the distance graph is significantly worse than having no spatial structure at all.

**Table 2. Diebold-Mariano Tests Against the Distance Graph** (negative DM indicates lower squared error for the comparison variant)

| Comparison | DM | p-value |
|---|---|---|
| Identity vs distance | -8.041 | < 0.001 |
| Correlation vs distance | -8.180 | < 0.001 |
| Star vs distance | -6.353 | < 0.001 |
| Without COVID vs distance | 6.278 | < 0.001 |

### 4.3 The Distribution Topology Carries the Signal

In contrast to the distance and correlation graphs, the distribution-network star achieves the best performance, significantly outperforming the distance graph (DM = -6.353, p < 0.001). This indicates that the real distribution topology carries predictive signal that geographic proximity does not. The mechanism is consistent with the data structure: because the distributor's hub is in Palembang, and all routes originate there, the star graph correctly encodes the flow of goods. Palembang is the dominant region (33 hospitals, 44% of the province's total), and connecting it to all other regions allows the model to propagate the hub's demand signal to the periphery.

### 4.4 Contribution of the COVID-19 Indicator

Removing the COVID-19 indicator from the distance graph decreases the R-squared from 0.0436 to 0.0425 (Diebold-Mariano DM = 6.278, p < 0.001). The indicator provides a small but significant improvement within the tuned configuration.

---

## 5. DISCUSSION

### 5.1 Interpretation

Our results show that, in a distribution setting, the graph should encode how goods actually move, not where facilities are located. The distribution-network star outperforms all other constructions, while geographic proximity adds no signal and even degrades performance relative to identity and random graphs.

This finding has a clear interpretation. In distribution businesses, demand at the regional level is driven by institutional procurement decisions, not by consumer mobility or geographic spillover. A hospital in one region does not buy more because a hospital in a neighboring region bought more; it buys according to its own procurement cycle and its own healthcare infrastructure. Geographic adjacency therefore carries little predictive signal. In contrast, the distribution topology reflects the actual logistics structure: all goods flow through the Palembang hub, so the hub's demand dynamics are informative for the periphery, and the star graph encodes this relationship.

The zero-inflated structure reinforces this interpretation. Because 84.85% of targets are zero, there is little non-zero signal to propagate spatially. A graph that connects regions with genuinely related non-zero activity (the hub and its served regions) can help, while a graph that connects regions with unrelated or absent activity (geographic neighbors) adds noise. This explains why the distance graph performs worse than identity: it propagates noise between regions that do not actually interact.

### 5.2 Implications for Practice

For practitioners building GNN-based demand forecasting systems in distribution businesses, our results suggest a simple but important design rule: ask how goods actually move before deciding how to construct the graph. If the distribution network is known, encode it directly. If it is not known, the ablation protocol we provide can be used to compare candidate graph constructions under identical conditions and select the one that performs best.

The finding that geographic proximity can degrade performance is a cautionary result. Many GNN forecasting papers default to distance-based graphs without testing alternatives. Our results show that this default can be actively harmful in distribution settings, and that a simple domain-informed graph (the star) can substantially outperform it.

### 5.3 Limitations

This study has several limitations. First, the results are based on a single distribution company and a single region (South Sumatra), so the generalizability to other distribution settings is not established. Second, the distribution network is represented as a simple star; a more detailed network with route-level information could yield further improvements, but such data were not available. Third, the graph is static; a dynamic graph that changes over time was not considered. Fourth, the ablation uses a single seed and configuration; multi-seed and multi-configuration results would strengthen the conclusions. Fifth, the zero-inflated classifier suffers from class imbalance, as documented in the main study, which limits the amount-stage predictions for sparse regions.

### 5.4 Future Work

Future work should extend the comparison to additional distribution settings and to public datasets to test generalizability. A dynamic graph that reflects changes in the distribution network over time is a natural extension. The ablation protocol could also be applied to other model architectures to test whether the graph-construction finding is robust across models.

---

## 6. CONCLUSION

We investigated whether the choice of graph construction determines forecasting quality in GNN-based regional demand forecasting, using real distribution data from a medical device distributor in South Sumatra. Under an identical zero-inflated hybrid LSTM-GNN framework, the distribution-network star achieved the best R-squared of 0.0703, a 61% relative improvement over the distance graph (0.0436). The distance graph performed worse than both identity adjacency and a random graph, and the sales-correlation graph also underperformed identity, indicating that geographic proximity and sales correlation carry no useful spatial signal in this setting. The real distribution topology, not geographic proximity, carries the predictive signal.

We conclude that for GNN-based demand forecasting in distribution businesses, the graph should encode how goods actually move, not where facilities are located. We provide a reproducible ablation protocol that practitioners can use to decide how to construct graphs for their own data, and we report an honest negative finding about the limits of geographic proximity in sparse, zero-inflated regional demand data.

---

## ACKNOWLEDGMENTS

The authors thank PT Parit Panjang for providing the distribution data and for confirming the distribution network structure.

---

## CRediT AUTHOR STATEMENT

Khoirusy Syafaat: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing - original draft, Visualization. Herri Setiawan: Supervision, Writing - review and editing, Project administration.

---

## DATA AVAILABILITY STATEMENT

The data that support the findings of this study are available from PT Parit Panjang but restrictions apply to the availability of these data, which were used under license for the current study and so are not publicly available. Data are however available from the authors upon reasonable request and with permission of PT Parit Panjang. The derived regional panel and the ablation results are available in the project repository.

---

## FUNDING

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

---

## DECLARATION OF COMPETING INTERESTS

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## DECLARATION OF RELATED MANUSCRIPTS

This manuscript is a companion to a related study (Syafaat & Setiawan, 2025) that develops and evaluates the zero-inflated hybrid LSTM-GNN model. The two manuscripts share the same dataset and model framework but ask different questions. This manuscript isolates the question of graph construction; the companion manuscript reports the overall model and its baselines. The authors declare this relationship to the editor.

---

## REFERENCES

- Alourani, A., Khan, N. A., Ashfaq, F., & Jhanjhi, N. Z. (2023). BiLSTM- and GNN-based spatiotemporal traffic flow forecasting with correlated weather data. *Journal of Advanced Transportation*, 2023, 8962283. https://doi.org/10.1155/2023/8962283
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253-263. https://doi.org/10.1080/07350015.1995.10524599
- Heryati, A., Stiawan, D., Setiawan, H., Rini, D. P., & Budiarto, R. (2024). Green transportation route model for industrial goods delivery with genetic algorithm. *2024 11th International Conference on Electrical Engineering, Computer Science and Informatics (EECSI)*, 775-781. https://doi.org/10.1109/EECSI63442.2024.10776095
- Lambert, D. (1992). Zero-inflated Poisson regression, with an application to defects in manufacturing. *Technometrics*, 34(1), 1-14. https://doi.org/10.1080/00401706.1992.10485228
- Luo, M., Dou, H., & Zheng, N. (2024). Spatiotemporal prediction of urban traffics based on deep GNN. *Computers, Materials & Continua*, 78(1), 265-282. https://doi.org/10.32604/cmc.2023.040067
- Patharkar, A., Cai, F., Al-Hindawi, F., & Wu, T. (2024). Predictive modeling of biomedical temporal data in healthcare applications: Review and future directions. *Frontiers in Physiology*, 15, 1386760. https://doi.org/10.3389/fphys.2024.1386760
- Santos, V. O., Rocha, P. A. C., Scott, J., Thé, J. V. G., & Gharabaghi, B. (2023). Spatiotemporal analysis of bidimensional wind speed forecasting: Development and thorough assessment of LSTM and ensemble graph neural networks on the Dutch database. *Energy*, 278(Part A), 127852. https://doi.org/10.1016/j.energy.2023.127852
- Sunder, R., Paul, V., Punia, S. K., Konduri, B., Nabilal, K. V., Lilhore, U. K., ... Tlija, M. (2024). An advanced hybrid deep learning model for accurate energy load prediction in smart building. *Energy Exploration & Exploitation*. https://doi.org/10.1177/01445987241267822
- Syafaat, K., & Setiawan, H. (2025). Spatiotemporal prediction of medical device sales using hybrid LSTM-GNN in South Sumatra. *Companion manuscript*.
- Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph WaveNet for deep spatial-temporal graph modeling. *IJCAI International Joint Conference on Artificial Intelligence*, 2019-August, 1907-1913. https://doi.org/10.24963/ijcai.2019/264
- Yang, Z., Liu, Z., Zhou, J., Song, C., Xiang, Q., He, Q., ... Zhang, J. (2023). A graph neural network (GNN) method for assigning gas calorific values to natural gas pipeline networks. *Energy*, 278(Part C), 127875. https://doi.org/10.1016/j.energy.2023.127875
- Ye, Y., Cao, Y., Dong, Y., & Yan, H. (2024). A graph neural network and transformer-based model for PM2.5 prediction through spatiotemporal correlation. *SSRN Working Paper*. https://doi.org/10.2139/ssrn.4979506

---

## APPENDIX: Reproducible Ablation Protocol

The following protocol was used to compare graph constructions under identical conditions:

1. Fix the model architecture, configuration, and seed across all variants.
2. Define the candidate graph constructions (identity, random, distance, correlation, domain-informed).
3. For each construction, build the adjacency matrix and train the model with the same split, preprocessing, and evaluation metrics.
4. For random graphs, average over multiple permutations.
5. Compute RMSE, MAE, and R-squared on the original scale, and use the Diebold-Mariano test for pairwise comparisons.
6. Report the results in a single table so that the effect of graph construction is directly visible.

This protocol is intentionally simple and can be applied to any GNN-based forecasting system. The key principle is that all variants must differ only in the adjacency matrix, so that any difference in performance is attributable to the graph construction.
