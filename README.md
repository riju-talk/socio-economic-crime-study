# 🚨 Delhi Crime Boundaries: Socio-Economic Discontinuity & Crime Concentration

## Overview

This project studies **how urban crime concentrates at the boundaries between socio-economically unequal neighborhoods** in Delhi, India. Using spatial data science, statistical modeling, and graph neural networks, we analyze whether **inequality gradients and boundary sharpness** explain crime intensity beyond standard poverty indicators.

The work follows a **ward-level crime inference framework** using NCRB (National Crime Records Bureau) data and extends it by introducing **explicit boundary-based inequality features** with Indian-specific socio-economic indicators from Census 2011 and NFHS-5.

---

## Research Question

> Do Delhi wards located at sharp socio-economic boundaries experience higher crime rates than socio-economically homogeneous areas, even after controlling for poverty, population density, and spatial autocorrelation?

---

## Data Sources (India-Specific)

| Source | Type | Granularity | Years |
|--------|------|-------------|-------|
| **NCRB Crime Data** | Crime incidents | Ward × Year | 2016-2024 |
| **Census 2011** | Socio-economic | Ward | 2011 |
| **DataMeet Shapefiles** | Administrative boundaries | Ward | - |
| **NFHS-5** | Health & poverty indicators | District | 2019-21 |
| **OpenStreetMap** | POI density | Ward | 2024 |
| **VIIRS Night Lights** | Economic activity proxy | Grid | 2016-2024 |

---

## Unit of Analysis

- **Delhi Ward × Year** (~270 wards)
- Crime rates normalized by population
- Boundary features computed via Queen-contiguity adjacency

---

## Feature Engineering

### Core Boundary-Aware Features

1. **Boundary Sharpness**: Max absolute difference between ward literacy/income and neighbors
2. **Income Gradient**: Distance-weighted income gap to adjacent wards
3. **Spatial Lag**: Average crime rate in neighboring wards
4. **Within-Ward Heterogeneity**: Gini coefficient for literacy/income distribution
5. **POI Density**: Banks, hospitals, markets, police stations per ward
6. **Night Lights Intensity**: VIIRS radiance as economic activity proxy

### Additional Features

- Population density, slum percentage, worker classification
- 3-year moving averages, YoY change, seasonal dummies
- Interaction terms: boundary_sharpness × population_density

---

## Modeling Approach

### Layer A: Explanatory Spatial Models
- **Negative Binomial Regression** (handles overdispersion)
- **Geographically Weighted NB** (spatial heterogeneity)
- **Diagnostics**: Moran's I on residuals, dispersion statistics

### Layer B: Predictive ML/DL Models
- **XGBoost** with spatial cross-validation
- **LSTM** for multi-horizon forecasting
- **CNN-LSTM** hybrid for spatio-temporal patterns

### Layer C: Graph Neural Networks (Bonus)
- **STGCN** (Spatio-Temporal Graph Conv Net)
- Multi-relational graph: adjacency + socio-economic similarity edges
- Attention weights for spillover pathway interpretation

---

## Evaluation Protocol

| Metric | Purpose |
|--------|---------|
| **MAE / RMSE** | Point forecast accuracy |
| **CRPS** | Probabilistic forecast quality |
| **Moran's I** | Spatial autocorrelation in residuals |
| **Disparate Impact Ratio** | Fairness across income quintiles |

### Validation Strategy
- **Spatial CV**: Cluster wards geographically → prevent leakage
- **Temporal Holdout**: Train ≤2022, Test 2023-2024
- **Uncertainty Quantification**: Monte Carlo dropout for prediction intervals

---

## Key Contributions

1. Introduces **boundary-aware inequality features** into Indian crime analysis
2. Demonstrates crime concentration as a **structural spatial phenomenon**
3. Implements **STGCN for crime spillover modeling** in irregular urban topology
4. Provides **fairness audit toolkit** with counterfactual testing
5. Delivers **policy-ready dashboard** with ethics safeguards

---

## Repository Structure

```
delhi-crime-boundaries/
├── data/
│   ├── raw/              # Original NCRB, Census, shapefiles
│   ├── interim/          # Cleaned, merged datasets
│   └── processed/        # Feature matrices, adjacency graphs
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_spatial_diagnostics.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_explanatory_models.ipynb
│   ├── 05_predictive_models.ipynb
│   ├── 06_gnn_spillover.ipynb
│   └── 07_ethics_audit.ipynb
├── src/
│   ├── data/             # Data loading, cleaning pipelines
│   ├── features/         # Boundary feature engineering
│   ├── models/           # NB, XGBoost, LSTM, STGCN
│   ├── evaluation/       # Metrics, spatial CV, fairness audit
│   └── visualization/    # Maps, SHAP plots, dashboards
├── models/               # Trained model artifacts
├── reports/
│   ├── figures/          # Visualization outputs
│   └── policy/           # Policy brief, ethics report
├── app/                  # Streamlit dashboard
├── configs/              # config.yaml for hyperparameters
├── tests/                # Unit tests, integration tests
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/delhi-crime-boundaries.git
cd delhi-crime-boundaries

# Create virtual environment (Python 3.10+)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download data (or use DVC)
dvc pull
```

---

## Quick Start

```python
# Load processed features
import pandas as pd
features = pd.read_parquet('data/processed/features_2024.parquet')

# Train baseline model
from src.models.explanatory import NegativeBinomialRegressor
model = NegativeBinomialRegressor()
model.fit(features[feature_cols], features['crime_count'])

# Evaluate
print(f"MAE: {model.score(X_test, y_test):.2f}")
```

---

## Dashboard

```bash
# Run Streamlit app
streamlit run app/dashboard.py
```

Dashboard tabs:
- **Spatial Patterns**: Choropleth maps of crime & boundary sharpness
- **Boundary Analysis**: Coefficient interpretations, partial dependence
- **Predictions**: Forecast with uncertainty intervals
- **Ethics**: Fairness metrics, bias audit results

---

## Ethics & Responsible AI

### Risk Mitigation Strategies

| Risk | Mitigation |
|------|------------|
| **Reinforcing bias** | Fairness regularization, counterfactual testing |
| **Over-policing** | Human-in-the-loop review, no real-time enforcement |
| **Privacy violations** | DPDP Act 2023 compliance, area-level aggregation |
| **Misuse** | Clear use policy, public model card, ethics report |

### Key Principles

1. **Descriptive, not prescriptive**: Model identifies patterns, does not dictate policy
2. **Area-level inference**: No individual-level predictions
3. **Uncertainty quantification**: All forecasts include confidence intervals
4. **Transparency**: Full code, data dictionary, bias audit publicly available

---

## Disclaimer

⚠️ This study is **descriptive and inferential**, not causal. Results should **not** be interpreted as:
- Policy prescriptions for enforcement
- Individual risk assessments
- Real-time policing tools

The model is intended for **preventive urban planning** and **resource allocation** with human oversight.

---

## Project Timeline (12 Weeks)

| Phase | Weeks | Deliverables |
|-------|-------|--------------|
| Foundation | 1-2 | Clean dataset, EDA notebook |
| Features + Explanatory | 3-5 | Boundary metrics, NB/GWNBR coefficients |
| Predictive + Time Series | 6-8 | XGBoost/LSTM with MAE < baseline |
| GNN Bonus | 9-10 | STGCN attention maps, spillover analysis |
| Ethics + Stakeholder | 11 | Policy brief, dashboard prototype |
| Polish + Showcase | 12 | GitHub release, portfolio assets |

---

## References

1. NCRB. *Crime in India Reports*, 2016-2024. ncrb.gov.in
2. Census of India. *Primary Census Abstract, 2011*. censusindia.gov.in
3. Wang, Y., et al. "Crime Rate Inference with Big Data." ACM KDD, 2016.
4. Weisburd, D., et al. "The Law of Crime Concentration at Places." Criminology, 2004.
5. Yu, B., et al. "Spatio-Temporal Graph Convolutional Networks." AAAI, 2018.

---

## License

MIT License - See LICENSE file for details.

---

## Contact

Made with ❤️ by Rijusmit Biswas  
[LinkedIn](https://linkedin.com/in/rijusmit) | [GitHub](https://github.com/rijusmit)
