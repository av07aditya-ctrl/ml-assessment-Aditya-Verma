# Part B: Business Case Analysis — Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — ML Problem Formulation

**Target Variable:**
`items_sold` — the number of items sold at a given store in a given month under a specific promotion.

**Candidate Input Features:**

| Feature Category | Features |
|---|---|
| Store attributes | `store_id`, `store_size`, `location_type` (urban/semi-urban/rural), `monthly_footfall`, `competition_density` |
| Promotion attributes | `promotion_type` (Flat Discount, BOGO, Free Gift, Category-Specific, Loyalty Points) |
| Temporal features | `month`, `year`, `is_weekend`, `is_festival`, `season` |
| Customer demographics | `avg_customer_age`, `customer_loyalty_rate` (if available) |
| Historical performance | `prev_month_items_sold`, `same_month_last_year_items_sold`, `avg_items_sold_per_promotion_type` |

**Type of ML Problem:**
This is a **supervised regression problem**. The target (`items_sold`) is a continuous numerical variable, making regression the appropriate framework.

However, this can also be framed as a **ranking or recommendation problem** — given a fixed set of five promotion types, the model predicts `items_sold` under each candidate promotion, and the promotion with the highest predicted volume is recommended. This is essentially five regression predictions per store per month, one for each promotion type.

**Justification:**
- The output (items sold) is continuous and ordinal, not a discrete class → regression, not classification.
- We need a cardinal estimate (not just "which is best") to justify business trade-offs (e.g., a promotion that sells 10% more items but costs significantly more may not be worth it).
- Regression outputs are also interpretable by the marketing team in familiar units.

---

### B1(b) — Why Items Sold is a Better Target Than Revenue

**The core problem with revenue as a target:**

Revenue = Price × Quantity. In a fashion retail context, prices vary dramatically across product categories, seasons (end-of-season sales), and promotional mechanisms (e.g., a Flat Discount directly reduces the unit price). This means revenue is **confounded by price variation that the model cannot disentangle from promotion effectiveness**.

Specifically:
- A **Flat Discount** raises volume while simultaneously lowering the per-unit price. The net revenue effect is ambiguous and depends on price elasticity — a model trained on revenue could incorrectly penalise discount promotions even if they drive the most footfall and basket volume.
- A **Loyalty Points Bonus** does not immediately reduce price but defers redemption to future periods. Revenue attributable to the promotion is therefore delayed and miscounted in the training month.
- **BOGO** promotions double the units sold but revenue per transaction may only marginally increase, making the promotion appear less effective under a revenue metric.

**Items sold is a cleaner, promotion-agnostic signal** because it captures consumer demand response to a promotion directly, without price distortion. It also aligns with operational objectives such as inventory turnover and reducing end-of-season stock.

**Broader Principle — Target Variable Selection:**

> *Choose targets that are as close as possible to the causal effect you want to measure, and free from confounding variables that the model cannot observe.*

Real-world ML projects frequently fail not because of poor model choice but because the target variable encodes noise, bias, or confounders. The process of target variable selection should include:
1. **Causal analysis** — does the target directly reflect the decision outcome being optimised?
2. **Confound audit** — are there external factors (e.g., price, seasonality) that co-determine the target alongside the features?
3. **Alignment with business objective** — is the target a proxy or the true goal? Revenue is a proxy; margin is closer to the true goal if profitability matters.

---

### B1(c) — Alternative to a Single Global Model

**Problem with a single global model:**
A global model trained on all 50 stores treats stores as exchangeable. But urban stores with high footfall and dense competition respond very differently from rural stores with a loyal, price-sensitive customer base. Averaging these patterns together produces a model that is mediocre for every store and excellent for none.

**Proposed Alternative: Hierarchical / Clustered Strategy**

| Strategy | Description |
|---|---|
| **Segment-level models** | Cluster stores by `location_type` and `store_size` (e.g., large urban, small rural) and train one model per cluster. Fewer parameters than 50 individual models; captures group-level patterns. |
| **Mixed-effects model** | A global model with store-level random effects (via hierarchical regression / LightGBM with store fixed effects). The global model provides a regularised prior; individual store adjustments capture idiosyncratic behaviour. |
| **Store-level fine-tuning** | Train a global base model on all data, then fine-tune on each store's historical data (transfer learning principle). Effective when individual store data is sparse. |

**Recommended approach:** A **two-level hierarchical model** — train separate models for each `location_type × store_size` combination (e.g., urban-large, urban-small, semi-urban-medium, rural-small). This gives 6–9 models instead of 50, balancing data sufficiency with store specificity. Use `store_id` as a feature within each model to capture residual store-level heterogeneity.

**Justification:** This approach acknowledges that stores within the same segment are more exchangeable with each other than across segments, while still pooling enough data to train a robust model. It is also operationally manageable — the marketing team can understand and trust "this is the model for large urban stores."

---

## B2. Data and EDA Strategy

### B2(a) — Joining the Four Tables

**Table grain and join strategy:**

| Table | Key(s) | Grain |
|---|---|---|
| `transactions` | `store_id`, `date`, `transaction_id` | One row per transaction |
| `store_attributes` | `store_id` | One row per store (slowly changing) |
| `promotion_details` | `promotion_id` / `store_id` + `month` | One row per store-month promotion assignment |
| `calendar` | `date` | One row per date (with `is_weekend`, `is_festival`) |

**Join sequence:**
```
transactions
  LEFT JOIN calendar         ON transactions.date = calendar.date
  LEFT JOIN promotion_details ON transactions.store_id = promotion_details.store_id
                               AND YEAR(transactions.date) = promotion_details.year
                               AND MONTH(transactions.date) = promotion_details.month
  LEFT JOIN store_attributes  ON transactions.store_id = store_attributes.store_id
```

**Grain of the final modelling dataset:**
> One row = one store × one month × one promotion type

This is the decision-making grain: "Given this store's attributes, this month's context (festival, season), and this promotion type — how many items will be sold?"

**Aggregations performed before modelling:**

| Aggregation | Purpose |
|---|---|
| `SUM(items_sold)` per store per month | Creates the target variable |
| `COUNT(transactions)` per store per month | Footfall proxy |
| `AVG(basket_size)` per store per month | Spending behaviour feature |
| `SUM(is_weekend)` / `COUNT(days)` per month | Proportion of weekend days in the month |
| `MAX(is_festival)` per month | Binary flag: any festival in the month |
| Lag features: `prev_month_items_sold`, `same_month_last_year_items_sold` | Encode momentum and year-over-year seasonality |

---

### B2(b) — EDA Strategy

**Analysis 1 — Promotion Type vs. Mean Items Sold (Bar Chart with CI)**
- **What to look for:** Which promotions drive the highest average volume overall? Are confidence intervals overlapping? If BOGO and Free Gift intervals overlap significantly, they may not be statistically distinguishable.
- **Modelling implication:** If one promotion dominates universally, the problem reduces in complexity. If no promotion is universally best, segment-level models are strongly motivated.

**Analysis 2 — Promotion Effectiveness by Location Type (Grouped Box Plot)**
- **What to look for:** Does the rank order of promotions change across urban, semi-urban, and rural stores? For example, does Loyalty Points Bonus outperform Flat Discount in urban stores but not in rural ones?
- **Modelling implication:** Significant interaction between `promotion_type` and `location_type` → include interaction features or use separate models per segment. This directly informs B1(c).

**Analysis 3 — Seasonal Heatmap (Month × Promotion Type, colour = mean items_sold)**
- **What to look for:** Are certain promotions more effective during festival months or year-end? Is BOGO particularly strong in December but weak in June?
- **Modelling implication:** Strong seasonality interactions motivate engineering `month × promotion_type` interaction features and including lag/rolling average features.

**Analysis 4 — Correlation Heatmap (Numerical Features)**
- **What to look for:** High correlations between `monthly_footfall` and `items_sold` (expected). Multicollinearity between `store_size` and `footfall` (common). Correlation between `competition_density` and items sold (should be negative — more competition suppresses sales).
- **Modelling implication:** Highly correlated features may be redundant; consider dropping one or using PCA. Negative correlation with competition density confirms its importance as a feature.

**Analysis 5 — Time Series Plot of Items Sold per Location Type**
- **What to look for:** Trend (overall growth or decline), seasonality (annual peaks), and structural breaks (e.g., COVID-era anomalies, new store openings).
- **Modelling implication:** Strong trend → include `year` and `month` as features. Structural breaks → consider excluding anomalous periods from training or adding a `covid_period` binary flag.

---

### B2(c) — Addressing Promotion Imbalance (80% No-Promotion Transactions)

**How the imbalance affects modelling:**

If 80% of training records have no promotion, the model will be overwhelmingly trained on the null-promotion baseline. It will have too few examples of each promotion type to learn the differential effect of specific promotions. The result is a model biased toward recommending no promotion or the most frequent promotion, because it has learned the baseline well but the promotion-specific signal poorly.

This is analogous to the class imbalance problem in classification, but applied to a categorical predictor rather than the target.

**Steps to address it:**

| Approach | Description |
|---|---|
| **Reframe the grain** | Aggregate to store-month level where every row has exactly one promotion assigned. This eliminates the imbalance at the modelling grain — every row has a promotion type. |
| **Oversample underrepresented promotion types** | If certain promotions (e.g., Loyalty Points Bonus) are rarely used in the data, oversample those store-month records during training to give the model sufficient signal. |
| **Create a synthetic no-promotion baseline** | Model the counterfactual: for each store-month where promotion X was run, what would items_sold have been without any promotion? This enables causal effect estimation via difference-in-differences or uplift modelling. |
| **Filter the training set** | Train the promotion recommendation model only on records where a promotion was active, supplemented by a baseline (no-promotion) model trained separately. The recommendation is then: "predicted uplift = model prediction − baseline prediction." |

The most robust long-term solution is to **reframe as an uplift/causal model** that estimates the incremental lift of each promotion over a no-promotion baseline, rather than predicting absolute items sold. This directly answers the business question: "Which promotion adds the most value compared to doing nothing?"

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split and Evaluation Metrics

**Setting up the split:**

With 3 years × 12 months × 50 stores = 1,800 store-month records, the appropriate split is **temporal**:

- **Training set:** Months 1–30 (years 1 and 2.5, ~80% of data)
- **Validation set:** Months 25–30 (a sliding window for hyperparameter tuning)
- **Test set:** Months 31–36 (the final 6 months — most recent data, strictly held out)

For ongoing evaluation, use **walk-forward (rolling-origin) cross-validation**: train on months 1–24, test on month 25; train on months 1–25, test on month 26; etc. This gives a realistic estimate of how the model will perform as new months arrive.

**Why a random split is inappropriate:**

A random split allows training records from December 2024 to coexist with test records from January 2023. The model can then effectively look into the future — learning patterns from later months that should not be available at the time of making predictions for earlier months. This inflates evaluation scores and makes the model appear more accurate than it will be in production.

In retail, this is particularly dangerous because **seasonal and promotional patterns repeat annually**. A random split leaks future year's seasonal signal into the training data, making the model appear to understand seasonality better than it actually does.

**Evaluation metrics and business interpretation:**

| Metric | Formula | Business Interpretation |
|---|---|---|
| **RMSE** | √(mean((ŷ−y)²)) | Penalises large errors heavily. A high RMSE means some stores are badly mis-predicted — those stores risk over- or under-stocking. Target: < 10–15% of mean items_sold. |
| **MAE** | mean(|ŷ−y|) | Average prediction error in units. Directly tells the operations team: "on average, our forecast is off by X items per store per month." Most interpretable for business stakeholders. |
| **MAPE** | mean(|ŷ−y|/y) × 100 | Percentage error; useful for comparing forecast accuracy across stores of very different sizes. Caution: undefined when y=0. |
| **Promotion Ranking Accuracy** | % of months where the recommended promotion matches the actual best promotion | Business-level metric: "How often does the model identify the right promotion?" More actionable than RMSE for the marketing team. |

**Preferred primary metric for stakeholder reporting:** MAE per store per month, expressed as a percentage of that store's average monthly sales volume (normalised MAE). This makes the error meaningful regardless of store size.

---

### B3(b) — Investigating Different Recommendations for the Same Store

**Scenario:** Model recommends Loyalty Points Bonus for Store 12 in December, and Flat Discount for Store 12 in March.

**Step 1 — Extract the feature values for both predictions**

Pull the input feature row for Store 12 in December and Store 12 in March. Present them side-by-side:

| Feature | December | March |
|---|---|---|
| `is_festival` | 1 | 0 |
| `month` | 12 | 3 |
| `competition_density` | 5 | 5 |
| `footfall_index` | High | Medium |
| `prev_month_items_sold` | High | Low |

**Step 2 — Apply SHAP (SHapley Additive exPlanations)**

Use `shap.TreeExplainer` on the Random Forest or Gradient Boosting model. For each prediction, SHAP gives a signed contribution of every feature:

- **December:** `is_festival=1` contributes +120 units; `month=12` contributes +95 units → high-engagement month motivates the loyalty programme which rewards repeat buyers.
- **March:** `prev_month_items_sold` (low, post-winter slowdown) contributes −80 units; `footfall_index` (medium) contributes −30 units → the model recommends a Flat Discount to stimulate price-sensitive, discretionary spending.

**Step 3 — Communicate to the marketing team**

Present a waterfall chart per store-month showing which features push the recommendation toward each promotion. Frame it in business language:

> *"In December, the model sees a high festival season with strong footfall. It recommends Loyalty Points because loyal customers are already engaged — rewarding them deepens retention without cannibalising margin via a discount. In March, footfall is lower and the previous month was sluggish. The Flat Discount recommendation reflects that price sensitivity dominates when discretionary spend is lower."*

This narrative transforms a model output into a **causal business story** the marketing team can act on and validate with their domain knowledge.

---

### B3(c) — End-to-End Deployment and Monitoring

**1. Saving the model**

```
# After training
import joblib
joblib.dump(pipeline, 'models/promotion_recommender_v1.pkl')
# Include the full sklearn Pipeline (preprocessor + model) so that
# transformation and prediction are a single .predict() call
```

Store versioned model artefacts (with date and performance metrics) in a model registry (e.g., MLflow, AWS SageMaker Registry). Never overwrite the production model without logging the replacement.

**2. Preparing and feeding new monthly data**

At the start of each month, an automated pipeline (e.g., orchestrated by Apache Airflow or AWS Step Functions) performs:

1. **Extract:** Pull the previous month's transaction data from the data warehouse.
2. **Aggregate:** Compute store-month level features (sum items_sold, footfall, basket size).
3. **Join:** Merge with store attributes and the calendar table for the upcoming month.
4. **Feature engineering:** Compute lag features (`prev_month_items_sold`), rolling averages, `is_festival` for the target month.
5. **Predict:** Load the saved Pipeline, call `pipeline.predict(X_new)` for each of the five promotion types per store. Recommend the promotion with the highest predicted `items_sold`.
6. **Output:** Write recommendations to a dashboard / email report for the marketing team.

**3. Monitoring for model degradation**

| Monitoring Type | Metric / Signal | Action Trigger |
|---|---|---|
| **Prediction drift** | Monitor the distribution of predicted items_sold each month. Alert if the mean or variance shifts by > 2σ from the training distribution. | Investigate root cause; schedule retraining. |
| **Feature drift** | Track distributions of input features (e.g., mean `competition_density`, proportion of festival months). A shift indicates the real world has changed. | Update training data; may require new features. |
| **Outcome tracking** | After each month, compare predicted vs actual items_sold. Monitor rolling MAE and RMSE over a 3-month window. | If rolling MAE exceeds 1.5× the baseline test MAE, trigger retraining. |
| **Recommendation uptake** | Track what % of model recommendations are actually implemented by the marketing team. Low uptake signals a trust or relevance problem, not just a performance problem. | Investigate with stakeholders; may need model explainability improvements. |

**Retraining cadence:**

- **Scheduled retraining:** Retrain every quarter using all available historical data (rolling window). Fashion retail has strong annual seasonality, so the model must see at least two full cycles.
- **Triggered retraining:** If any monitoring alert fires, investigate and retrain if data drift is confirmed.
- **Champion-challenger testing:** When a new model version is trained, deploy it to 10% of stores for one month and compare its MAE against the production model (champion) before full rollout.
