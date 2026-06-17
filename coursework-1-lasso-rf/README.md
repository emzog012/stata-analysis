# ECM323: Big Data Analytics — Coursework 1

**MSc International Business, Economics & Strategy**

---

## Overview

This project covers three applied machine learning tasks using real and
simulated datasets.

---

## Questions

### Question 1 — LASSO Variable Selection

Used `lasso linear` with 10-fold cross-validation to identify which variables
had non-zero coefficients from 100 candidates (n=150). Compared the CV-minimum
and 1-SE rule for choosing the regularisation parameter λ. The 1-SE rule
selected 3 true signal variables (x30, x49, x83); CV-min selected 15,
including 11 noise variables.

### Question 2 — Polynomial Regression & Cross-Validation

Used the `crossfold` package to compute 10-fold CV MSE for polynomial
regressions of house prices on distance from the nearest tube station
(orders p=1 to 10). A quartic polynomial (p=4) minimised CV MSE with
a dramatic drop from ~10.8 million at p=3 to ~95 at p=4.

### Question 3 — Predicting Energy Efficiency Gap

Trained OLS, LASSO, and a random forest (`rforest`) on 649,000 UK properties
to predict the gap between actual and potential energy efficiency scores.

| Model                      | Test MSE |
| -------------------------- | -------- |
| OLS                        | 36.03    |
| LASSO (CV-min)             | 36.03    |
| Random Forest (m=4, B=100) | 28.78    |

The random forest achieved a **20.1% reduction** in test MSE. Key predictors
included CO₂ emissions, property age band, fuel type, and floor area.

---

## Methods Used

- LASSO with 10-fold cross-validation (CV-min and 1-SE rule)
- Polynomial regression with cross-validated model selection
- Random forest with OOB tuning of mtry parameter
- OLS with robust standard errors

## Packages Required
