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

### Installation
```bash
pip install -e ".[dev]"
```

### Environment Setup
The audit trail requires a secret key to be set:
```bash
export AUDIT_SECRET_KEY="your-secret-key-at-least-16-chars"
```

### 1. Audit Command (Single Task Evaluation)
```bash
python cli.py audit --task-id TASK-001 --target KEY-01 --primary 28.5 --secondary 14.2
```

### 2. Chat Command (Supervisory Query)
```bash
python cli.py chat "Explain the safety protocols"
```

### 3. Batch Command (CSV Processing)
```bash
python cli.py batch -i input.csv -o results.csv
```

### 4. Verify Audit Trail
```bash
python cli.py verify-audit
```

### 5. Serve REST API
```bash
python cli.py serve --host 127.0.0.1 --port 8000
```

### ROC Analysis Commands (roc_delong_cli.py)

Compute ROC curve, DeLong CI, and Youden cutpoint for one test:
```bash
python roc_delong_cli.py compute --data sample_data.csv --label-col disease_status --score-col biomarker_a
```

Compare two correlated ROC curves:
```bash
python roc_delong_cli.py compare --data sample_data.csv --label-col disease_status --score-a biomarker_a --score-b biomarker_b
```

---

## 🛡️ Security & Enterprise Architecture

* **Zero-PHI Outbound Interceptor:** Active regex inspection blocking SSNs, MRNs, phone numbers, emails, DOBs, and patient identifiers.
* **Tamper-Evident HMAC-SHA256 Audit Trail:** Chained, cryptographically signed logs with full signature verification for every evaluation and state transition.
* **Secure Key Management:** Audit signing key loaded from `AUDIT_SECRET_KEY` environment variable (minimum 16 characters). No hardcoded defaults.
* **Path Traversal Protection:** Batch command validates all file paths remain within the working directory.
* **Air-Gapped LLM Reasoning Adapter:** Agnostic integration for local Ollama instances (`llama3`, `mistral`), Claude 3.5 Sonnet, GPT-4o, and deterministic test mocks.
* **Active Learning Bayesian Calibration:** Dynamic tracker updating worker reliability weights and monitoring Brier calibration drift.
* **FastAPI & Prometheus Telemetry:** Exposes OpenAPI 3.1 REST endpoints and operational Prometheus metrics (`/metrics`).

---

## 🧪 Testing & Verification

Run the automated test suite:

```bash
pytest -v
```

Run the core ROC/DeLong unit tests directly:

```bash
python tests/unit_test_roc_delong_cli.py
```

Execute high-throughput batch simulation benchmarks:

```bash
python simulator.py 1000
```

---

## 🐳 Container Deployment

Create a `.env` file with your audit secret key:
```bash
echo "AUDIT_SECRET_KEY=your-production-secret-key-here" > .env
```

Build and run:
```bash
docker build -t roc-delong-cli .
docker run -p 8000:8000 roc-delong-cli
```

Or use docker-compose:
```bash
docker-compose up -d
```
