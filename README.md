# Bayesian Road Accident Risk Analysis

Bayesian modelling of 20,000 road accidents from 8 Indian cities (2022–2025): risk-score regression, severity classification and latent traffic segmentation. Every model uses full posterior inference (NUTS on GPU via PyMC + NumPyro/JAX).

## Results

**1. Risk-score regression**

| Model | R² | MAE | R-hat |
|---|---|---|---|
| Gaussian linear | 0.891 | 0.050 | up to 2.90 (fails convergence) |
| Beta regression | 0.872 | 0.060 | 1.00 |
| **Hierarchical Beta (chosen)** | **0.881** | **0.055** | **1.00, 0 divergences** |

The Gaussian model scores slightly higher but does not converge, so the hierarchical Beta model is the one to trust.

**2. Severity classification (Minor / Major / Fatal)**

| Model | Accuracy | Macro F1 | Fatal F1 |
|---|---|---|---|
| Gaussian Naive Bayes | 0.530 | 0.423 | 0.50 |
| **Bayesian multinomial logistic (σ = 1)** | **0.559** | **0.647** | **1.00** |

Results were stable across prior scales σ = 1, 2, 5.

**3. Latent traffic segmentation (K = 3)**

| Model | Silhouette |
|---|---|
| Generative Bayesian mixture | **0.40** (vs 0.37 for the BGMM baseline) |

The mixture model isolates a **7 PM high-risk window on 4-lane roads**.

## Repository structure

```
code/bayesian_analysis.ipynb     regression, classification and mixture models
data/indian_roads_dataset.csv    20,000 accident records
report/final_report.pdf          full project report
```

## Data

[Indian Road Accident Dataset 2022–2025](https://www.kaggle.com/datasets/sehaj1104/indian-road-accident-dataset-20222025) from Kaggle, included in `data/`.

## Getting started

```bash
pip install -r requirements.txt
cd code
jupyter notebook bayesian_analysis.ipynb
```

The notebook reads `../data/indian_roads_dataset.csv`. A CUDA GPU with a GPU build of JAX is recommended. CPU works, but sampling is slow.

## Context

CP 218 Theory & Applications of Bayesian Learning, Indian Institute of Science (IISc), Jan–Apr 2026. Individual project by Yakshith Maddineni.
