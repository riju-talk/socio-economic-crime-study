# 🚨 North India Cyber Crime Analysis: Spatio-Temporal Trends & Predictive Modeling (2017-2025)

## Overview

This project studies **cyber crime patterns, trends, and predictive factors across North Indian states** from 2017 to 2025. Using spatial data science, time series analysis, and machine learning, we analyze how **digital adoption, socio-economic indicators, and law enforcement capacity** influence cyber crime incidence across the region.

The work follows a **state/district-level cyber crime inference framework** using NCRB (National Crime Records Bureau) data, incorporating Indian-specific digital infrastructure indicators from Census 2011, NFHS-5, and TRAI (Telecom Regulatory Authority of India) reports.

**Geographic Scope**: North India (States: Delhi, Haryana, Himachal Pradesh, Jammu & Kashmir, Ladakh, Punjab, Rajasthan, Uttar Pradesh, Uttarakhand, Chandigarh)

**Time Period**: 2017-2025

## Research Questions

1. **Primary**: How have cyber crime rates evolved across North Indian states from 2017 to 2025, and what socio-technical factors best explain this variation?

2. **Secondary**:
   - Does digital infrastructure penetration (internet users, mobile subscriptions) correlate with cyber crime incidence?
   - Are there distinct spatio-temporal clusters of cyber crime hotspots in North India?
   - Can we accurately forecast cyber crime trends for 2024-2025 using historical patterns?
   - What is the relationship between traditional crime rates and cyber crime emergence?

## Data Sources (India-Specific)

| Source | Type | Granularity | Years |
|--------|------|-------------|-------|
| **NCRB Crime in India Reports** | Cyber crime incidents | State × Year | 2017-2023 |
| **NCRB Cyber Crime Data** | Cyber crime categories | State × Year | 2017-2023 |
| **Census 2011** | Socio-economic indicators | State/District | 2011 |
| **TRAI Performance Indicators** | Telecom/digital infrastructure | State × Quarter | 2017-2024 |
| **NFHS-5** | Health & socio-economic indicators | State/District | 2019-21 |
| **MeitY (Ministry of Electronics & IT)** | Digital adoption metrics | State × Year | 2017-2024 |
| **CERT-In Reports** | Cyber security incidents | National/State | 2017-2024 |
| **RBI Digital Payment Statistics** | Digital transaction volumes | State × Year | 2017-2024 |
| **Shapefiles (DataMeet/OpenGov)** | Administrative boundaries | State/District | - |

### Key Variables to Collect

#### Dependent Variables (Cyber Crime Metrics)
- Total cyber crime cases registered (IPC + Special & Local Laws)
- Cyber crime rate per 100,000 population
- Cyber crime by category:
  - Financial fraud (UPI, banking, cryptocurrency)
  - Identity theft & impersonation
  - Cyber stalking/harassment
  - Data breaches
  - Ransomware attacks
  - Phishing scams
  - Child exploitation content

#### Independent Variables (Predictors)

**Digital Infrastructure:**
- Internet subscribers per 1000 population
- Mobile subscription density
- Broadband penetration
- 4G/5G coverage percentage
- Number of cyber cafes
- Digital payment transaction volume

**Socio-Economic Indicators:**
- Literacy rate (overall, male, female)
- Urbanization percentage
- Per capita income/GSDP
- Education levels (graduate & above)
- Employment in IT/ITES sector
- Bank account penetration (Jan Dhan accounts)

**Law Enforcement Capacity:**
- Number of cyber crime cells
- Police personnel per 100,000 population
- Cyber crime training programs conducted
- Budget allocation for cyber security

**Traditional Crime Indicators:**
- Overall crime rate (IPC crimes)
- Financial fraud cases (traditional)
- Property crime rates

## Unit of Analysis

- **State × Year** (10 North Indian states/UTs × 9 years = 90 observations)
- **District × Year** (where district-level data available)
- Crime rates normalized by population (per 100,000)
- Time series: Annual data from 2017 to 2025 (projected)

## Feature Engineering

### Core Cyber Crime Features

1. **Digital Penetration Index**: Composite score of internet, mobile, and broadband penetration
2. **Cyber Vulnerability Score**: Based on digital adoption rate × cybersecurity awareness index
3. **Spatial Lag**: Average cyber crime rate in neighboring states
4. **Temporal Trends**: Year-over-year growth rates, 3-year moving averages
5. **Economic Activity Proxy**: Digital payment volume, GSDP per capita
6. **Law Enforcement Readiness**: Cyber cells per district, trained personnel ratio

### Additional Features

- **Interaction Terms**: 
  - Digital penetration × literacy rate
  - Urbanization × internet subscribers
  - Traditional fraud × digital payment volume
  
- **Time-Series Features**:
  - Lag features (t-1, t-2, t-3 years)
  - Seasonal patterns (if quarterly data available)
  - Trend components (linear, polynomial)
  
- **Derived Metrics**:
  - Cyber crime clearance rate
  - Reporting efficiency index
  - Digital divide index (urban-rural gap)

## Modeling Approach

### Layer A: Explanatory Models (Understanding Drivers)
- **Panel Data Regression** (Fixed/Random Effects for state-level heterogeneity)
- **Poisson/Negative Binomial Regression** (for count data with overdispersion)
- **Spatial Durbin Model** (accounts for spatial spillover effects)
- **Diagnostics**: Moran's I, VIF for multicollinearity, Hausman test

### Layer B: Predictive ML Models (Forecasting)
- **XGBoost/LightGBM** with time-series cross-validation
- **Random Forest** for feature importance analysis
- **ARIMA/SARIMA** for univariate time series forecasting
- **Prophet** for trend decomposition and seasonality

### Layer C: Deep Learning (Advanced Temporal Patterns)
- **LSTM/GRU** for multi-step ahead forecasting (2024-2025 predictions)
- **Transformer-based models** (Temporal Fusion Transformer)
- **Hybrid CNN-LSTM** for spatio-temporal pattern recognition

### Layer D: Spatial Analysis (Bonus)
- **Hotspot Analysis**: Getis-Ord Gi* statistic for identifying cyber crime clusters
- **Spatial Autocorrelation**: Moran's I, Geary's C
- **Geographically Weighted Regression (GWR)**: Local parameter estimation

## Evaluation Protocol

| Metric | Purpose |
|--------|---------|
| **MAE / RMSE / MAPE** | Point forecast accuracy |
| **R² Score** | Model fit and explanatory power |
| **MASE** | Time series forecast accuracy (scale-independent) |
| **AIC / BIC** | Model selection for statistical models |
| **Feature Importance** | Understanding key drivers (SHAP values) |
| **Spatial Autocorrelation (Moran's I)** | Check for spatial patterns in residuals |

### Validation Strategy
- **Time-Series Cross-Validation**: Rolling/expanding window approach
  - Train: 2017-2020, Test: 2021
  - Train: 2017-2021, Test: 2022
  - Train: 2017-2022, Test: 2023
- **Temporal Holdout**: Final evaluation on 2023-2024 data
- **Forecasting Horizon**: Predict 2024-2025 with confidence intervals
- **State-wise Validation**: Ensure model generalizes across all North Indian states
- **Uncertainty Quantification**: Prediction intervals using quantile regression or conformal prediction

## Key Contributions

1. First comprehensive **spatio-temporal analysis of cyber crime in North India** (2017-2025)
2. Identifies **key socio-technical drivers** of cyber crime using panel data methods
3. Develops **predictive models** for forecasting cyber crime trends at state level
4. Creates **cyber vulnerability index** combining digital adoption and socio-economic factors
5. Provides **policy recommendations** for resource allocation and capacity building
6. Open-source framework replicable for other Indian regions

## Repository Structure

```
north-india-cyber-crime/
├── data/
│   ├── raw/              # Original NCRB, TRAI, Census data
│   ├── interim/          # Cleaned, merged datasets
│   └── processed/        # Feature matrices, time series data
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_explanatory_models.ipynb
│   ├── 05_predictive_models.ipynb
│   ├── 06_time_series_forecasting.ipynb
│   └── 07_vulnerability_analysis.ipynb
├── src/
│   ├── data/             # Data loading, cleaning pipelines
│   ├── features/         # Feature engineering for cyber crime
│   ├── models/           # Panel regression, XGBoost, LSTM, Prophet
│   ├── evaluation/       # Metrics, time-series CV, forecasting
│   └── visualization/    # Charts, maps, dashboards
├── models/               # Trained model artifacts
├── reports/
│   ├── figures/          # Visualization outputs
│   └── policy/           # Policy brief, findings
├── app/                  # Streamlit dashboard
├── configs/              # config.yaml for hyperparameters
├── tests/                # Unit tests, integration tests
├── requirements.txt
├── Dockerfile
└── README.md
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/north-india-cyber-crime.git
cd north-india-cyber-crime

# Create virtual environment (Python 3.10+)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download data (or use DVC)
dvc pull
```

## Quick Start

```python
# Load processed data
import pandas as pd
df = pd.read_parquet('data/processed/cyber_crime_features.parquet')

# Train panel regression model
from src.models.panel import FixedEffectsModel
model = FixedEffectsModel()
model.fit(df, entity='state', time='year', target='cyber_crime_rate')

# Forecast with LSTM
from src.models.forecasting import CyberCrimeLSTM
lstm = CyberCrimeLSTM()
forecast = lstm.predict_horizon(years=[2024, 2025])

# Evaluate
print(f"RMSE: {model.rmse:.2f}")
print(f"2025 Prediction: {forecast['2025']}")
```

## Dashboard

```bash
# Run Streamlit app
streamlit run app/dashboard.py
```

Dashboard tabs:
- **Overview**: Cyber crime trends across North India (2017-2025)
- **State-wise Analysis**: Interactive charts by state/UT
- **Drivers & Correlations**: Feature importance, correlation matrices
- **Forecasting**: 2024-2025 predictions with confidence intervals
- **Policy Insights**: Recommendations and resource allocation

## Ethics & Responsible AI

### Risk Mitigation Strategies

| Risk | Mitigation |
|------|------------|
| **Stigmatization of states** | Context-aware interpretation, avoid ranking without nuance |
| **Data privacy** | DPDP Act 2023 compliance, aggregate-level analysis only |
| **Misuse for political narratives** | Transparent methodology, peer review |
| **Over-reliance on predictions** | Emphasize uncertainty, human expert review |
| **Digital divide reinforcement** | Include equity metrics, avoid penalizing low-adoption regions |

### Key Principles

1. **Aggregate-level analysis**: No individual or household-level predictions
2. **Transparent methodology**: Open-source code, documented data sources
3. **Uncertainty communication**: All forecasts include confidence/prediction intervals
4. **Contextual interpretation**: Results must be interpreted with socio-technical context
5. **Policy support, not prescription**: Tools for informed decision-making, not automated decisions

## Disclaimer

⚠️ This study is **descriptive and predictive**, not causal. Results should **not** be interpreted as:
- Law enforcement directives or operational guidelines
- Individual risk assessments or profiling tools
- Real-time crime prediction systems
- Definitive measures of state performance (due to reporting variations)

**Important Notes:**
- Cyber crime data suffers from significant **under-reporting** across all Indian states
- Reporting efficiency varies by state, affecting cross-state comparisons
- Digital infrastructure correlation does not imply causation
- 2024-2025 figures are **projections** based on historical trends, not actual counts

The model is intended for **academic research**, **policy planning**, and **resource allocation** with human oversight and domain expertise.

## Project Timeline (12 Weeks)

| Phase | Weeks | Deliverables |
|-------|-------|--------------|
| **Data Collection & EDA** | 1-2 | NCRB/TRAI data compiled, exploratory analysis notebook |
| **Preprocessing & Features** | 3-4 | Clean panel dataset, digital penetration index, feature matrix |
| **Explanatory Modeling** | 5-6 | Panel regression results, key driver identification |
| **Predictive Modeling** | 7-8 | XGBoost/LSTM models, time-series forecasts for 2024-2025 |
| **Spatial Analysis** | 9 | Hotspot detection, spatial autocorrelation analysis |
| **Dashboard & Reporting** | 10-11 | Interactive Streamlit app, policy brief, final report |
| **Polish & Documentation** | 12 | GitHub release, README, presentation materials |

## References

### Data Sources
1. **NCRB**. *Crime in India Reports* (2017-2023). https://ncrb.gov.in
2. **NCRB**. *Cyber Crime Statistics in India* (2017-2023). https://ncrb.gov.in
3. **Census of India**. *Primary Census Abstract, 2011*. https://censusindia.gov.in
4. **TRAI**. *Performance Indicators Report* (2017-2024). https://trai.gov.in
5. **MeitY**. *Digital India Reports* (2017-2024). https://meity.gov.in
6. **RBI**. *Digital Payment Statistics* (2017-2024). https://rbi.org.in
7. **CERT-In**. *Annual Cyber Security Incident Reports* (2017-2024). https://cert-in.org.in
8. **NFHS-5**. *National Family Health Survey* (2019-21). https://nfhsindia.org

### Academic References
9. Hinduja, S., & Patchin, J. W. "Cybercrime Prevention: What Works and What Doesn't." *Criminology & Public Policy*, 2020.
10. Leukfeldt, E. R. "Targeting the Human Factor in Cybercrime." *European Journal of Criminology*, 2017.
11. Williams, M. L., et al. "Crime Science and the Cyber Challenge." *Palgrave Communications*, 2019.
12. Broadhurst, R., et al. "Organized Crime and Cybercrime." *Asian Journal of Criminology*, 2018.
13. Holt, T. J., & Lampke, E. "Exploring Stolen Data Markets Online." *Global Crime*, 2020.
14. Pathak, A., et al. "Cyber Crime Trends in India: An Empirical Analysis." *International Journal of Cyber Criminology*, 2021.

## License

MIT License - See LICENSE file for details.

## Contact

Made with ❤️ by Rijusmit Biswas  
[LinkedIn](https://linkedin.com/in/rijusmit) | [GitHub](https://github.com/rijusmit)
