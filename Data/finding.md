# Dataset Finding Report

## Project Requirements Summary

| Requirement | Description |
|-------------|-------------|
| **Domain** | Finance, Investment, Fraud, or Business |
| **Features** | Minimum 5 features |
| **Exclusivity** | Unique to group (first-come, first-serve) |
| **Task Versatility** | Must support both Regression and Classification |
| **Analytical Depth** | Rich enough for storytelling and feature importance analysis |

---

## Recommended Datasets

### Option 1: Financial Distress Prediction Dataset ⭐ (Best Choice)

**Source**: [Kaggle - Financial Distress](https://www.kaggle.com/datasets/shebrahimi/financial-distress)

#### Overview
This dataset deals with financial distress prediction for a sample of companies. It contains financial and non-financial characteristics that can be used to predict whether a company will become financially distressed.

#### Dataset Structure

| Column | Description |
|--------|-------------|
| **Company Index** | Unique identifier for each company |
| **Time** | Time period indicator |
| **Financial Distress** | Target variable (continuous value) |
| **x1 - x83** | 83 features (financial and non-financial characteristics) |

#### Target Variables

| Type | Variable | Description |
|------|----------|-------------|
| **Regression** | Financial Distress (continuous) | Raw distress score for numerical forecasting |
| **Classification** | Financial Distress (binary) | If > -0.50 → Healthy (0); Otherwise → Distressed (1) |

#### Why This Dataset is Ideal

| Criteria | Assessment |
|----------|------------|
| **Domain** | ✅ Finance / Business |
| **Number of Features** | ✅ 83 features (far exceeds minimum 5) |
| **Dual Task Support** | ✅ Native regression + classification targets |
| **Feature Importance** | ✅ Excellent - 83 features to analyze |
| **Storytelling** | ✅ Clear narrative: "Why do companies fall into financial distress?" |
| **Uniqueness** | ✅ Less commonly used than credit card fraud datasets |

#### Suggested Project Title
> "Predicting Financial Distress in Companies: A Machine Learning Approach to Early Warning Systems"

---

### Option 2: Corporate Credit Rating with Financial Ratios

**Source**: [Kaggle - Corporate Credit Rating](https://www.kaggle.com/datasets/kirtandelwadia/corporate-credit-rating-with-financial-ratios)

#### Overview
Credit ratings of all publicly traded US companies with financial ratios. Credit ratings provide an assessment about the credit worthiness of a company and act as a pivotal financial indication to potential investors.

#### Dataset Structure

| Attribute | Details |
|-----------|---------|
| **Timeframe** | 2010-2016 |
| **Data Points** | 7,805 |
| **Companies** | 678 |
| **Rating Agencies** | 7 |
| **Sectors** | 12 |
| **Rating Scale** | S&P 22-grades (AAA to D) |
| **Financial Ratios** | 16 |

#### Target Variables

| Type | Variable | Description |
|------|----------|-------------|
| **Regression** | Financial Ratios | Continuous financial metrics for numerical forecasting |
| **Classification** | Credit Rating | Ordinal categorical (AAA, AA+, AA, ..., CCC, CC, C, D) |

#### Why This Dataset Works

| Criteria | Assessment |
|----------|------------|
| **Domain** | ✅ Finance / Business |
| **Number of Features** | ✅ 16+ financial ratios + metadata |
| **Dual Task Support** | ✅ Regression on ratios, Classification on ratings |
| **Feature Importance** | ✅ Financial ratios are highly interpretable |
| **Storytelling** | ✅ "What financial metrics determine a company's credit rating?" |

#### Rating Classification
- **Investment Grade**: BBB+ and above
- **Junk Grade**: BB+ and below

---

### Option 3: Gold Price Prediction Dataset

**Source**: [Kaggle - Gold Price Prediction](https://www.kaggle.com/datasets/sid321axn/gold-price-prediction-dataset)

#### Overview
Data collected from November 18th, 2011 to January 1st, 2019. The challenge is to accurately predict the future adjusted closing price of Gold ETF. Contains 1,718 rows and 80 columns including various market indicators.

#### Dataset Structure

| Category | Features |
|----------|----------|
| **Gold ETF** | Open, High, Low, Close, Adjusted Close, Volume |
| **S&P 500 Index** | SP_open, SP_high, SP_low, SP_close, SP_Ajclose, SP_volume |
| **Dow Jones Index** | DJ_open, DJ_high, DJ_low, DJ_close, DJ_Ajclose, DJ_volume |
| **EUR/USD** | EU_Price, EU_open, EU_high, EU_low, EU_Trend |
| **Oil Futures** | OF_Price, OF_Open, OF_High, OF_Low, OF_Volume, OF_Trend |
| **Silver Futures** | SF_Price, SF_Open, SF_High, SF_Low, SF_Volume, SF_Trend |
| **US Bond Rate** | USB_Price, USB_Open, USB_High, USB_Low, USB_Trend |
| **Platinum** | PLT_Price, PLT_Open, PLT_High, PLT_Low, PLT_Trend |
| **Palladium** | PLD_Price, PLD_Open, PLD_High, PLD_Low, PLD_Trend |
| **US Dollar Index** | USDI_Price, USDI_Open, USDI_High, USDI_Low, USDI_Volume, USDI_Trend |
| **Gold Miners ETF** | GDX_Open, GDX_High, GDX_Low, GDX_Close, GDX_Adj Close, GDX_Volume |

#### Target Variables

| Type | Variable | Description |
|------|----------|-------------|
| **Regression** | Adjusted Close | Gold ETF adjusted closing price |
| **Classification** | Price Zone (needs creation) | High / Medium / Low price categories |

#### Limitations
- ⚠️ Classification target needs to be created from continuous price
- ⚠️ Primarily designed for regression tasks

---

## Comparison Matrix

| Criteria | Financial Distress | Credit Rating | Gold Price |
|----------|-------------------|---------------|------------|
| **Domain** | Finance/Business | Finance/Business | Investment/Finance |
| **Features** | 83 | 16+ | 80 |
| **Rows** | Multiple companies | 7,805 | 1,718 |
| **Regression Target** | ✅ Distress Score | ✅ Financial Ratios | ✅ Gold Price |
| **Classification Target** | ✅ Healthy/Distressed | ✅ AAA-D Rating | ⚠️ Needs creation |
| **Feature Importance** | Excellent | Good | Excellent |
| **Interpretability** | Medium | High | Medium |
| **Uniqueness** | High | Medium | Medium |

---

## Final Recommendation

### 🏆 Winner: Financial Distress Prediction Dataset

| Justification | Details |
|---------------|---------|
| **Meets All Requirements** | Finance/Business domain, 83 features, supports both regression and classification |
| **Uniqueness** | Less commonly used than credit card fraud datasets - lower risk of duplication |
| **Analytical Depth** | Rich 83-feature set for comprehensive feature importance analysis |
| **Natural Dual Targets** | Continuous distress score + binary classification built-in |
| **Storytelling Potential** | Strong narrative around early warning systems for company financial health |

### Suggested Analysis Workflow

```
1. Data Exploration
   - Understand 83 features
   - Check for missing values
   - Analyze distribution of financial distress scores

2. Regression Task
   - Target: Financial Distress (continuous)
   - Models: Linear Regression, Random Forest, XGBoost
   - Metrics: RMSE, MAE, R²

3. Classification Task
   - Target: Healthy (0) vs. Distressed (1)
   - Models: Logistic Regression, Random Forest, SVM
   - Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC

4. Feature Importance Analysis
   - Identify top predictors of financial distress
   - SHAP values for interpretability
   - Business insights and recommendations
```

---

## Alternative Datasets Considered

| Dataset | Domain | Why Not Selected |
|---------|--------|------------------|
| Credit Card Fraud (ULB) | Fraud | Very commonly used; classification only |
| Dow Jones Index | Finance | Limited features for classification |
| Stock Prediction (Synthetic) | Finance | Synthetic data; less authentic |
| Twitter Financial News | Finance | NLP task; not suitable for regression |
| 1000 Companies Profit | Business | Only 4 features (below minimum 5) |

---

## Data Sources

1. Kaggle Datasets: https://www.kaggle.com/datasets
2. UCI Machine Learning Repository: https://archive.ics.uci.edu/
3. Hugging Face Datasets: https://huggingface.co/datasets

---

*Report generated for IntroMLHomework Project*
*Date: August 2026*
