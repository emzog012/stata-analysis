# ECM323: Big Data Analytics — Coursework 2

**MSc International Business, Economics & Strategy**

---

## Overview

This project covers three applied tasks: principal component analysis (PCA),
PCA-based matrix completion for missing data, and stability analysis of
LASSO variable selection.

---

## Questions

### Question 1 — Principal Component Analysis

Applied PCA to two 30-variable datasets to determine how many components
were needed to capture most of the variation. Used variance shares,
cumulative variance, and scree plots to select 5 components for both
datasets, capturing 83.3% of variance in Dataset 1 and 96.8% in Dataset 2.

### Question 2 — PCA-Based Matrix Completion

Implemented Algorithm 12.1 from *Introduction to Statistical Learning*
(§12.5.2) to impute missing values using a rank-5 PCA reconstruction,
iterating until the observed-cell MSE converged. Imputation accuracy
across all 30 variables ranged from r = 0.850–0.888 and RMSE = 0.467–0.531,
roughly halving the error of a simple mean-imputation benchmark.

### Question 3 — LASSO Selection Stability

Used 10-fold cross-validation to choose the optimal LASSO penalty (lambda*),
then refit LASSO within each fold to assess how stable variable selection
was across folds. Measured stability using mean pairwise Jaccard similarity
(0.446) and visualised results with a selection-frequency bar chart and a
fold-by-predictor heatmap. Also assessed coefficient stability for
consistently-selected predictors using coefficient of variation.

---

## Methods Used

- Principal component analysis (PCA)
- PCA-based matrix completion / missing data imputation
- LASSO with k-fold cross-validation
- Jaccard similarity for selection stability
- Coefficient of variation for coefficient stability

## File Structure

| File | Description |
| ---- | ----------- |
| `coursework2_pca_lasso.do` | Complete do file for all three questions |

---

## Data

Data provided by the ECM323 course team and not included in this repository.
