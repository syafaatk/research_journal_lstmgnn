# The Long Tail of Intermittent Demand: Product-Level Sparsity in Medical Device Distribution and Its Implications for Zero-Inflated Forecasting

Khoirusy Syafaat, Herri Setiawan
Faculty of Computer and Natural Sciences, Indo Global Mandiri University, Palembang 30129, Indonesia
khoirusysyafaat@students.uigm.ac.id, herri@uigm.ac.id

**Manuscript status:** Draft lengkap (2 Sep 2026). Semua angka terverifikasi dari `results/intermittent_demand.json` (dihitung dari `data/view_penjualan_detail data hingga oktober.xlsx`). Merujuk pada naskah utama (revisi.md) dan paper #8 (manuscript_paper_8_collapse.md) untuk konteks model zero-inflated; paper ini menjadikan struktur intermittent demand level produk sebagai objek kajian, dan menghubungkannya dengan sparsity yang sudah didokumentasikan di level region-hari.

---

## ABSTRACT

Zero-inflated forecasting models are typically motivated by sparsity at the level of the forecasting target, such as the daily sales of a region. We show that in medical device distribution, sparsity is far more severe at the product level, where most products are bought rarely or not at all for long stretches. Using real distribution data from a medical device distributor in South Sumatra, Indonesia (13,661 invoices, 2,380 distinct products over 64 months), we characterize the structure of product-level intermittent demand. Of 2,380 products, 867 (36%) are bought exactly once and 912 (38%) are active in only one month; 1,684 products (70.8%) are active in six months or fewer, yet account for only 19.9% of total value. Applying the Syntetos-Boylan classification to the 770 products with at least six active months, only 12 (1.6%) are smooth; 232 (30.1%) are intermittent and 482 (62.6%) are lumpy, with a median average demand interval of 4.27 months and a median squared coefficient of variation of 0.71. A long tail of 469 products has at least one gap of more than 12 months with no purchase. We argue that this product-level intermittency is the structural root of the zero-inflated classifier collapse documented at the region-day level: both arise from the same underlying sparsity, and product-level forecasting must therefore be treated as an intermittent-demand problem rather than a smooth time-series problem. We conclude that product-level demand in this setting is dominated by intermittent and lumpy patterns, that value is concentrated in a small set of frequently bought products, and that forecasting and inventory policy must be designed for intermittency rather than smoothness.

**Keywords:** Intermittent Demand; Lumpy Demand; Zero-Inflated Forecasting; Product-Level Analysis; Medical Device Supply Chain; Syntetos-Boylan Classification

---

## HIGHLIGHTS

- 70.8% of products are active in six months or fewer
- Only 1.6% of products with sufficient history are smooth demand
- 92.7% of products are intermittent or lumpy (Syntetos-Boylan)
- Product-level sparsity is the structural root of the ZI classifier collapse
- Value is concentrated: 71.9% in the 419 most frequently bought products

---

## 1. INTRODUCTION

Sparse demand is a defining feature of distribution and healthcare supply chains. At the level of a region and a day, most cells of the demand matrix are zero, which motivates zero-inflated (ZI) forecasting formulations that separate the question "is there an order?" from "how large is the order?". The companion study (Syafaat & Setiawan, 2025) and the diagnostic paper on the zero-inflated classifier collapse (paper #8) document this sparsity at the region-day level: 84.85% of daily region-level test targets are zero, and the ZI classifier collapses to predicting non-zero only for the dominant region.

This paper shifts the unit of analysis from the region-day to the product. We ask a complementary question: how sparse is demand at the product level, where a product is bought rarely or not at all for long stretches? This is the domain of intermittent-demand analysis, a well-established literature in inventory management (Croston, 1972; Syntetos & Boylan, 2005). We characterize the structure of product-level demand in a real medical device distribution dataset and connect it to the sparsity that drives the ZI formulation.

The motivation is practical and methodological. If most products are bought rarely, then product-level forecasting is an intermittent-demand problem, not a smooth time-series problem, and the standard tools of time-series forecasting (which assume regular, non-zero observations) are inappropriate. Moreover, the sparsity that drives the ZI classifier collapse at the region-day level is not an artifact of aggregation; it is the product-level reality of the underlying demand. Understanding this structure is a prerequisite for designing forecasting and inventory policies for the long tail of rarely bought products.

The main contributions of this paper are as follows. First, we characterize the distribution of product-level purchase frequency and active periods in a real medical device distribution dataset. Second, we apply the Syntetos-Boylan classification to quantify the share of smooth, erratic, intermittent, and lumpy products. Third, we show that product-level intermittency is the structural root of the region-day sparsity that drives the ZI classifier collapse. Fourth, we draw implications for forecasting and inventory policy for the long tail of rarely bought products.

The novelty of this paper is empirical and diagnostic. We do not propose a new forecasting model; we characterize the product-level demand structure and connect it to the ZI forecasting problem. This is a contribution to the applied forecasting and inventory-management literatures, which frequently assume smooth demand or treat sparsity only at the aggregate level.

---

## 2. RELATED WORK

### 2.1 Intermittent Demand

Intermittent demand is demand that occurs sporadically, with many periods of zero demand interspersed with occasional non-zero demand. Croston (1972) introduced a method that separately estimates the demand interval and the demand size. Syntetos and Boylan (2005) proposed a classification based on the average demand interval (ADI) and the squared coefficient of variation of non-zero demand (CV2): smooth (ADI < 1.32, CV2 < 0.49), erratic (ADI < 1.32, CV2 >= 0.49), intermittent (ADI >= 1.32, CV2 < 0.49), and lumpy (ADI >= 1.32, CV2 >= 0.49). Lumpy demand, which combines infrequent occurrence with high variability, is the most difficult to forecast.

### 2.2 Zero-Inflated Forecasting

Zero-inflated models treat the target as a mixture of a point mass at zero and a positive distribution (Lambert, 1992). In deep learning, this is implemented as a two-stage model: a binary classifier predicts whether a sale occurs, and a regression head predicts the amount conditional on a sale. The companion study and paper #8 document that this formulation, while standard, can collapse under extreme class imbalance: the classifier predicts non-zero only for the dominant region, silencing sparse regions.

### 2.3 The Connection Between Product and Aggregate Sparsity

The ZI literature typically motivates sparsity at the level of the forecasting target (e.g., region-day). We argue that this sparsity is a consequence of product-level intermittency: a region has zero sales on a given day because the products that would be bought that day are rarely bought at all. The two levels of sparsity are linked, and understanding the product-level structure is necessary to interpret and address the aggregate-level collapse.

### 2.4 Position Relative to the Main Study

This paper is a companion to a study that develops and evaluates a zero-inflated hybrid LSTM-GNN model for regional medical device sales forecasting (Syafaat & Setiawan, 2025), and to the diagnostic paper on the ZI classifier collapse (paper #8). The main study and paper #8 operate at the region-day level. This paper operates at the product level and characterizes the demand structure that underlies the aggregate sparsity. The three papers share the same dataset but ask different questions: the main study asks "how well can the model predict regional demand," paper #8 asks "how does the ZI classifier behave under extreme class imbalance," and this paper asks "how sparse is demand at the product level, and what does that imply for forecasting and inventory policy."

---

## 3. DATA AND METHOD

### 3.1 Data

We use medical device sales data from PT Parit Panjang, a distribution company in South Sumatra, Indonesia, covering July 2020 to October 2025 (64 months). The data contain 13,661 unique invoices and 2,380 distinct products, defined by the product name and specification. The total value is 96,975M IDR. The unit of analysis in this paper is the product and the month.

### 3.2 Product-Level Demand Measures

For each product we compute:
1. **Purchase frequency.** The number of invoices that contain the product.
2. **Active months.** The number of distinct months in which the product was bought.
3. **Longest zero gap.** The longest stretch of months, in months, with no purchase, computed only for products active in at least two months.
4. **Total value.** The sum of the line-item value across all invoices.

### 3.3 Syntetos-Boylan Classification

We apply the Syntetos-Boylan (SB) classification to the 770 products active in at least six months, which have enough observations for a stable estimate of the demand interval and variability. For each such product we compute:
- **Average demand interval (ADI)** = total months / number of active months.
- **Squared coefficient of variation (CV2)** = (standard deviation / mean) squared, over non-zero monthly demand.

Products are classified as smooth (ADI < 1.32, CV2 < 0.49), erratic (ADI < 1.32, CV2 >= 0.49), intermittent (ADI >= 1.32, CV2 < 0.49), or lumpy (ADI >= 1.32, CV2 >= 0.49). We restrict the classification to products with at least six active months because ADI and CV2 are unstable for products with very few observations; the frequency and gap statistics below are reported for all products.

---

## 4. RESULTS

### 4.1 Purchase Frequency Is Highly Skewed

Table 1 reports the distribution of purchase frequency across products.

**Table 1. Purchase Frequency Distribution**

| Invoices per product | Products | Share of products | Share of value |
|---|---|---|---|
| 1 | 867 | 36.4% | 6.1% |
| 2-5 | 643 | 27.0% | 8.7% |
| 6-20 | 451 | 19.0% | 13.4% |
| >20 | 419 | 17.6% | 71.9% |

The distribution is heavily skewed. More than a third of products (867, 36.4%) are bought exactly once, and 63.4% of products are bought five or fewer times. Value, however, is concentrated in the small set of frequently bought products: the 419 products bought more than 20 times (17.6% of products) account for 71.9% of total value, while the 1,510 products bought five or fewer times account for only 14.7% of value.

### 4.2 Most Products Are Active for Only a Short Period

Table 2 reports the distribution of active months per product.

**Table 2. Active Months Distribution**

| Active months | Products | Share of products | Share of value |
|---|---|---|---|
| 1 | 912 | 38.3% | 7.0% |
| 2-6 | 772 | 32.4% | 12.9% |
| 7-24 | 462 | 19.4% | 26.0% |
| >24 | 234 | 9.8% | 54.1% |

Only 234 products (9.8%) are active in more than 24 of the 64 months. The majority of products (1,684, 70.8%) are active in six months or fewer, yet these account for only 19.9% of total value. This confirms a long-tail structure: many rarely active products, few persistently active products.

### 4.3 Long Zero Gaps Are Common

Of the 1,468 products active in at least two months, 469 (31.9%) have at least one gap of more than 12 months with no purchase. The median longest gap is 8 months and the mean is 11 months. This means that even among products that are bought more than once, it is common for a product to disappear from the purchase record for a year or more. Such products are effectively "not bought at all in a given period," which is precisely the intermittency that makes product-level forecasting difficult.

### 4.4 Syntetos-Boylan Classification

Table 3 reports the Syntetos-Boylan classification for the 770 products active in at least six months.

**Table 3. Syntetos-Boylan Classification (770 products with >= 6 active months)**

| Class | Products | Share | Value (M IDR) |
|---|---|---|---|
| Smooth | 12 | 1.6% | 7,932 |
| Erratic | 44 | 5.7% | 21,424 |
| Intermittent | 232 | 30.1% | 12,229 |
| Lumpy | 482 | 62.6% | 37,947 |

Only 12 of 770 products (1.6%) are smooth. The overwhelming majority are intermittent (30.1%) or lumpy (62.6%), together 92.7%. The median ADI is 4.27 months and the median CV2 is 0.71. This is a decisive result: product-level demand in this setting is dominated by intermittent and lumpy patterns, not smooth time series.

### 4.5 Product-Level Sparsity Underlies the ZI Classifier Collapse

The region-day sparsity that drives the ZI classifier collapse (paper #8) is a consequence of this product-level intermittency. A region has zero sales on a given day because the products that would be bought that day are rarely bought at all. The 84.85% zero rate at the region-day level is not an aggregation artifact; it reflects the underlying reality that most products are bought rarely. This connection is important: it means that the ZI classifier collapse is not a modeling bug that can be fixed by tuning, but a structural consequence of the demand process itself. Any forecasting approach for this setting must be designed for intermittency.

---

## 5. DISCUSSION

### 5.1 Interpretation

The results show that product-level demand in medical device distribution is dominated by intermittent and lumpy patterns. Only 1.6% of products with sufficient history are smooth, and 92.7% are intermittent or lumpy. The long tail of rarely bought products is large (70.8% of products active in six months or fewer) but accounts for a small share of value (19.9%). This is the classic structure of intermittent demand, and it has direct implications for forecasting and inventory policy.

### 5.2 Implications for Forecasting

Product-level forecasting in this setting cannot rely on smooth time-series methods. The standard tools of time-series forecasting assume regular, non-zero observations, which are absent for the majority of products. Instead, product-level forecasting must be treated as an intermittent-demand problem, using methods such as Croston's method, the Syntetos-Boylan approximation, or models that explicitly separate the demand interval from the demand size. The ZI two-stage formulation, which separates "is there an order?" from "how large is the order?", is a natural fit for this structure, but it must be designed to avoid the classifier collapse documented in paper #8.

### 5.3 Implications for Inventory Policy

The concentration of value in a small set of frequently bought products (71.9% in 17.6% of products) suggests a differentiated inventory policy. For the small set of frequently bought products, standard inventory models apply. For the long tail of rarely bought products, the intermittency and lumpiness mean that safety stock must account for long zero gaps and high variability, and that overstocking is a real risk. The 469 products with gaps of more than 12 months are candidates for make-to-order or low-stock policies rather than continuous stocking.

### 5.4 Limitations

This study has several limitations. First, the results are based on a single distributor and a single region, so the generalizability is not established. Second, the product definition (name and specification) may not perfectly correspond to distinct SKUs, and some products may be under- or over-counted. Third, the Syntetos-Boylan classification is applied only to products with at least six active months; the classification of the many products with fewer observations is not reported. Fourth, this paper characterizes the demand structure but does not build or evaluate a product-level forecasting model; that is left for future work.

### 5.5 Future Work

Future work should build and evaluate product-level intermittent-demand forecasting models on this dataset, comparing Croston-type methods with the ZI deep-learning framework, and should evaluate inventory policies for the long tail. A benchmark on public intermittent-demand datasets would strengthen the generalizability of the findings.

---

## 6. CONCLUSION

We characterized the structure of product-level intermittent demand in a real medical device distribution dataset. Of 2,380 products, 70.8% are active in six months or fewer, and only 1.6% of products with sufficient history are smooth; 92.7% are intermittent or lumpy. Value is concentrated in a small set of frequently bought products. We argued that this product-level intermittency is the structural root of the region-day sparsity that drives the zero-inflated classifier collapse, and that product-level forecasting must be treated as an intermittent-demand problem. We conclude that forecasting and inventory policy for medical device distribution must be designed for intermittency rather than smoothness.

---

## ACKNOWLEDGMENTS

The authors thank PT Parit Panjang for providing the distribution data.

---

## CRediT AUTHOR STATEMENT

Khoirusy Syafaat: Conceptualization, Methodology, Software, Formal analysis, Investigation, Data curation, Writing - original draft, Visualization. Herri Setiawan: Supervision, Writing - review and editing, Project administration.

---

## DATA AVAILABILITY STATEMENT

The data that support the findings of this study are available from PT Parit Panjang but restrictions apply to the availability of these data, which were used under license for the current study and so are not publicly available. Data are however available from the authors upon reasonable request and with permission of PT Parit Panjang. The derived product-level demand measures are available in the project repository.

---

## FUNDING

This research received no specific grant from any funding agency in the public, commercial, or not-for-profit sectors.

---

## DECLARATION OF COMPETING INTERESTS

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

---

## DECLARATION OF RELATED MANUSCRIPTS

This manuscript is a companion to a related study (Syafaat & Setiawan, 2025) that develops and evaluates the zero-inflated hybrid LSTM-GNN model for regional medical device sales forecasting, and to a diagnostic paper on the zero-inflated classifier collapse. The three manuscripts share the same dataset but ask different questions. This manuscript isolates the product-level demand structure; the companion manuscripts report the overall model and the classifier collapse at the region-day level. The authors declare these relationships to the editor.

---

## REFERENCES

- Croston, J. D. (1972). Forecasting and stock control for intermittent demands. *Operational Research Quarterly*, 23(3), 289-303.
- Lambert, D. (1992). Zero-inflated Poisson regression, with an application to defects in manufacturing. *Technometrics*, 34(1), 1-14.
- Syntetos, A. A., & Boylan, J. E. (2005). The accuracy of intermittent demand estimates. *International Journal of Forecasting*, 21(2), 303-314.
- Syafaat, K., & Setiawan, H. (2025). Spatiotemporal prediction of medical device sales using hybrid LSTM-GNN in South Sumatra. *Companion manuscript*.
