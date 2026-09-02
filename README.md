# Roc Delong Cli

> **Domain:** Clinical Decision Support & Biomedical Computing  
> **Reference Guidelines & Standards:** `Standard Clinical Formulations & ISO/IEC Quality Frameworks`

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)
![Audit Trail](https://img.shields.io/badge/Audit-HMAC--SHA256_Tamper--Evident-brightgreen.svg)
![Zero-PHI Guard](https://img.shields.io/badge/Guard-Zero--PHI_Outbound-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)

</div>

---

## 📖 What It Does

ROC DeLong CLI
==============

Computes an empirical ROC curve with the Youden-index optimal cutpoint for a
single diagnostic test / biomarker, and statistically compares two
correlated ROC curves (e.g. two biomarkers measured on the same subjects)
using the DeLong et al. (1988) nonparametric test.

Core methods implemented:
  - Empirical ROC curve via counting (equivalent to the trapezoidal rule
    over (FPR, TPR) points), with AUC computed as the Mann-Whitney U
    statistic (which is exactly the trapezoidal-rule AUC of the empirical
    ROC curve).
  - Youden index (sensitivity + specificity - 1) to select the optimal
    cutpoint.
  - DeLong (1988) covariance of AUCs via Mann-Whitney U "structural
    components", for a DeLong-method confidence interval on a single AUC
    and for a two-sample paired z-test comparing two correlated AUCs.

---

## ⚙️ Key Capabilities & Algorithmic Modules

### 🔬 Analytical Functions

- **`split_by_label()`**: Split parallel score/label arrays into positive- and negative-class
score arrays.
- **`roc_curve()`**: Empirical ROC curve.

Returns (thresholds, fpr, tpr) as 1-D numpy arrays, ordered so that
threshold decreases from +inf down to the minimum observed score
(i.e. FPR/TPR increase from (0,0) to (1,1)). A subject is classified
positive when score >= threshold.
- **`auc_trapezoidal()`**: AUC via the trapezoidal rule over (FPR, TPR) points.
- **`youden_cutpoint()`**: Find the threshold maximizing Youden's J = sensitivity + specificity - 1.

Returns a dict with the cutpoint threshold, sensitivity, specificity and
J statistic. The +inf threshold (the trivial all-negative point) is
excluded from consideration.
- **`delong_components()`**: Mann-Whitney U structural components for one test (DeLong et al. 1988).

pos_scores: 1-D array of scores for the m positive-class subjects.
neg_scores: 1-D array of scores for the n negative-class subjects.

Returns (V10, V01, auc) where:
  V10 has shape (m,): the average, over negatives, of psi(X_i, Y_j)
  V01 has shape (n,): the average, over positives, of psi(X_i, Y_j)
  auc is the Mann-Whitney U estimate of the AUC (mean of the psi matrix)

psi(x, y) = 1 if x > y, 0.5 if x == y, 0 if x < y

---

## 💻 CLI Quickstart & Usage

### 1. Guided Interactive Mode
```bash
python cli.py
```

### 2. Direct Parameterized Evaluation
```bash
python cli.py --task-id <value> --target <value> --primary <value> --secondary <value>
```

### Parameter Reference
- `--task-id`: Specifies input measurement or parameter value.
- `--target`: Specifies input measurement or parameter value.
- `--primary`: Specifies input measurement or parameter value.
- `--secondary`: Specifies input measurement or parameter value.
- `--critical`: Specifies input measurement or parameter value.
- `--status`: Specifies input measurement or parameter value.
- `--input`: Specifies input measurement or parameter value.
- `--output`: Specifies input measurement or parameter value.

### Input Data Schema

| Field | Description | Requirement |
|:------|:------------|:------------|
| `suite_name` | Parameter / observation metric | Required |
| `system_slug` | Parameter / observation metric | Required |
| `standard_reference` | Parameter / observation metric | Required |
| `test_cases` | Parameter / observation metric | Required |

---

## 🛡️ Security & Enterprise Architecture

* **Zero-PHI Outbound Interceptor:** Active AST and regex inspection blocking SSNs, MRNs, phone numbers, and patient identifiers.
* **Tamper-Evident HMAC-SHA256 Audit Trail:** Chained, cryptographically signed logs for every evaluation and state transition.
* **Air-Gapped LLM Reasoning Adapter:** Agnostic integration for local Ollama instances (`llama3`, `mistral`), Claude 3.5 Sonnet, GPT-4o, and deterministic test mocks.
* **Active Learning Bayesian Calibration:** Dynamic tracker updating worker reliability weights and monitoring Brier calibration drift.
* **FastAPI & Prometheus Telemetry:** Exposes OpenAPI 3.1 REST endpoints and operational Prometheus metrics (`/metrics`).

---

## 🧪 Testing & Verification

Run the automated test suite:

```bash
pytest -v
```

Execute high-throughput batch simulation benchmarks:

```bash
python simulator.py --tasks 1000 --concurrency 8
```

---

## 🐳 Container Deployment

```bash
docker build -t roc-delong-cli .
docker run -p 8000:8000 roc-delong-cli
```
