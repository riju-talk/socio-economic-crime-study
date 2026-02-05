# 📌 Crime Concentration at Socio-Economic Boundaries

## Overview

This project studies **how urban crime concentrates at the boundaries between socio-economically unequal neighborhoods**, rather than simply within poor areas. Using spatial data science and statistical modeling, we analyze whether **inequality gradients and boundary sharpness** explain crime intensity beyond standard poverty indicators.

The work follows a **neighborhood-level crime inference framework** inspired by ACM KDD urban analytics research and extends it by introducing **explicit boundary-based inequality features**.

---

## Research Question

> Do neighborhoods located at sharp socio-economic boundaries experience higher crime rates than socio-economically homogeneous areas, even after controlling for poverty and spatial autocorrelation?

---

## Data Sources

* **NYPD Complaint Data (Historic)**
* **American Community Survey (ACS 5-year estimates)**
* **Census tract shapefiles (TIGER/Line)**

---

## Unit of Analysis

* **Census tract × year**
* Crime rates normalized by population

---

## Feature Engineering

Key features include:

* Income heterogeneity (within-tract)
* Boundary sharpness (difference between tract income and neighbors)
* Maximum income gap to neighbors
* Neighborhood income variance
* Spatially lagged crime rates

These features explicitly model **socio-economic structure at neighborhood boundaries**.

---

## Modeling Approach

* Poisson regression
* Negative Binomial regression
* Random Forest (robustness baseline)

Evaluation includes:

* Spatial cross-validation
* Temporal holdout testing
* Error analysis by income strata

---

## Key Contributions

* Introduces **boundary-aware inequality features** into crime analysis
* Demonstrates that crime concentration is partly a **structural spatial phenomenon**
* Shows how spatial dependency and inequality interact in urban settings

---

## Repository Structure

```
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_spatial_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_interpretation.ipynb
├── src/
│   ├── features/
│   ├── models/
│   └── utils/
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   └── figures/
└── README.md
```

---

## Disclaimer

This study is **descriptive and inferential**, not causal. Results should not be interpreted as policy prescriptions or predictive policing tools.

---

## References

* Wang et al., *Crime Rate Inference with Big Data*, ACM KDD
* Weisburd et al., *The Law of Crime Concentration at Places*
