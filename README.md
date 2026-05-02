#  Gold Price Prediction

> Machine learning system for forecasting XAU/USD prices using 25+ years of multi-source financial data.

---

##  Overview

This project builds a robust gold price forecasting pipeline that integrates **market data**, **macroeconomic indicators**, and **engineered technical features** to train and compare multiple ML models. Ridge and Lasso Regression outperformed tree-based models (XGBoost, LightGBM) as the best-performing approaches.

---

##  Project Structure

```
gold-price-prediction/
│
├── data/
│   └── gold_dataset.csv          # Final dataset — 6,419 rows × 69 features
│
├── notebooks/
│   ├── 01_data_collection.py     # Fetch data from Yahoo Finance & FRED
│   └── 02_gold_eda.ipynb         # Exploratory Data Analysis
│
├── models/
│   ├── ridge_model.pkl
│   ├── lasso_model.pkl
│   ├── xgboost_model.pkl
│   └── lightgbm_model.pkl
│
├── reports/
│   ├── Gold_EDA_Report.pdf
│   └── Gold_Team_Proposal.pdf
│
└── README.md
```

---

##  Dataset

| Property | Value |
|---|---|
| Date Range | 2000 – 2026 |
| Trading Days | 6,419 rows |
| Total Features | 69 columns |
| Target | `Target_NextDay` (next-day gold price) |

### Data Sources

**Yahoo Finance** — Daily market prices:
- Gold Futures `GC=F`, Silver `SI=F`, Oil WTI `CL=F`, Platinum, Copper
- S&P 500, Nasdaq, Dow Jones, MSCI World
- US Dollar Index (DXY), EUR/USD, JPY/USD, GBP/USD
- VIX Fear Index, 10Y & 2Y Treasury Yields, GLD ETF

**FRED (Federal Reserve)** — Macroeconomic indicators:
- CPI (Inflation), Fed Funds Rate, Real Interest Rate
- M2 Money Supply, PPI, Treasury Yield Spread
- Unemployment Rate, GDP Growth, Industrial Production, Consumer Sentiment

### Engineered Features

| Category | Features |
|---|---|
| Moving Averages | MA7, MA14, MA21, MA50, MA200, EMA variants |
| Price Returns | Return_1d, 3d, 7d, 14d, 30d |
| Lagged Prices | Lag_1d, 3d, 7d, 14d, 21d |
| Momentum | RSI_14, MACD, MACD_Signal, MACD_Hist, Momentum_14d |
| Bollinger Bands | BB_Upper, BB_Lower, BB_Width, BB_Pos |
| Volatility | Volatility_7d, Volatility_30d |
| Ratios | Gold/Silver Ratio, Gold/Oil Ratio |
| Calendar | DayOfWeek, Month, Quarter, Year |

---

##  Models

| Model | Type | Result |
|---|---|---|
| **Ridge Regression** | Regularized Linear | ✅ Best |
| **Lasso Regression** | Regularized Linear | ✅ Best |
| XGBoost | Gradient Boosting | Benchmark |
| LightGBM | Gradient Boosting | Benchmark |

> Ridge and Lasso outperformed tree-based models, demonstrating that gold price — while driven by many features — follows strong linear relationships with macro indicators.

---

##  Installation

```bash
git clone https://github.com/your-username/gold-price-prediction.git
cd gold-price-prediction
pip install -r requirements.txt
```

### Requirements

```
yfinance
pandas
numpy
pandas-datareader
scikit-learn
xgboost
lightgbm
matplotlib
seaborn
scipy
jupyter
```

---

## 🚀 Usage

### 1. Collect Data

```bash
python data/collect_gold_data.py
```

This fetches all tickers from Yahoo Finance and FRED, engineers features, and saves `gold_dataset.csv`.

### 2. Run EDA

```bash
jupyter notebook notebooks/02_gold_eda.ipynb
```

### 3. Train Models

```python
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv("data/gold_dataset.csv", index_col=0, parse_dates=True)

# Features & target
drop = ["Target_NextDay", "Target_Next7d", "Target_Direction", "Gold_ETF_GLD"]
X = df.drop(columns=drop).select_dtypes("number")
y = df["Target_NextDay"]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Time-series split (no random shuffle)
tscv = TimeSeriesSplit(n_splits=5)

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.01)
```

---

##  Team

| Name | Role |
|---|---|
| Abdelrahman | Data Collection & EDA |
| Mai Kamal | Preprocessing & Feature Engineering |
| Lamiaa Khaled | Preprocessing & Feature Engineering |
| Hanaa Alaa | Modeling |
| Yosef Mesala | Modeling |
| Hanaa Hemda | Deployment |

---

##  Reports

- [`Gold_EDA_Report.pdf`](reports/Gold_EDA_Report.pdf) — Full exploratory analysis with correlation heatmaps, seasonality, and outlier detection
- [`Gold_Team_Proposal.pdf`](reports/Gold_Team_Proposal.pdf) — Academic research proposal

---

##  License

MIT License — free to use and modify with attribution.
