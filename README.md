# ROC & DeLong CLI

Empirical receiver operating characteristic (ROC) analysis, Youden-index thresholds, and nonparametric DeLong AUC inference. Use the Python command-line tool or the static, browser-based interface.

**Browser application:** The repository's configured Pages address is [roc-delong-cli](https://abusuraihsakhri.github.io/roc-delong-cli/). See [the static app source](web/index.html) if that site has not yet been published from the repository's root directory. The link is included from repository metadata; live availability should be checked after deployment.

## Features

- ROC curve and trapezoidal AUC, with half-credit for tied positive/negative scores.
- Youden's J (sensitivity + specificity − 1) and its highest-threshold maximizing cutpoint.
- DeLong normal-approximation confidence interval for one AUC.
- Two-sided paired DeLong z-test comparing scores measured on **the same subjects in the same row order**.
- CSV thresholds and ROC image export (PNG from Python; CSV and SVG from the browser).
- Input validation for missing/nonfinite scores, outcome classes, and paired sample lengths.
- Browser analysis entirely client-side: no backend, Pyodide, account, or upload service.

## Browser usage

Open the Pages link or serve this repository locally, then open `web/index.html`. Upload a CSV or select **Load demonstration data**. Choose the outcome column, positive label, and one or two score columns; click **Analyze ROC data**. Download the threshold table or SVG plot.

Each CSV row must correspond to one subject. Both score columns must be numerical and complete, and **higher scores must indicate the positive outcome**. The browser accepts at most 20,000 observations and a 5 MiB CSV file.

The page makes no network requests for uploaded data, external scripts, or fonts. Uploaded data are kept only in the current tab and are not saved to browser storage. Do not infer that this privacy property extends to the separate FastAPI audit service.

## Python installation and CLI

Requires Python 3.10 or later:

```bash
python -m pip install -e .
```

Single test (AUC, 95% DeLong CI, Youden cutpoint):

```bash
roc-delong compute --data sample_data.csv --label-col disease_status --score-col biomarker_a --output-csv roc.csv --plot roc.png
```

Two paired tests (difference in AUC and two-sided p-value):

```bash
roc-delong compare --data sample_data.csv --label-col disease_status --score-a biomarker_a --score-b biomarker_b --output-csv comparison.csv --plot comparison.png
```

Both commands default to positive label `1`; specify `--positive-label` for another class. `compute` also accepts `--alpha` to change the nominal two-sided confidence level. Python implementation: [`roc_delong_cli.py`](roc_delong_cli.py). The supplementary `roc-delong-cli` entry point still provides the pre-existing `audit`, `batch`, `chat`, `verify-audit`, and `serve` commands; these are distinct from ROC analysis.

## Statistical limitations

DeLong variance estimation requires **at least two positive and two negative observations**. The paired test assumes exchangeable independent subjects and matched rows; it is not a test for independent datasets, repeated/clustered observations, or missing-score imputation. A nonzero AUC difference with exactly zero estimated difference variance has an **undefined z-test** and is reported as an error rather than an artificial p-value.

Reported AUC confidence intervals use a normal approximation and are clipped to the [0, 1] range. Selection of the maximum Youden index is descriptive; no adjustment is made for threshold-selection optimism, spectrum effects, or multiple comparisons. Results require independent clinical/statistical validation before consequential use.

## Local development and tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -v
node --check web/roc.js
node tests/browser_smoke.cjs
python -m pip check
```

To serve the static browser app locally, run `python -m http.server 8000` from the repository root and open `http://localhost:8000/web/`. The app is plain HTML/CSS/JavaScript and runs in current Chrome, Firefox, Safari, and Edge releases. The core Python statistics use NumPy and SciPy; Matplotlib is used for PNG plots.

An **optional** FastAPI service preserves the repository's separate historical audit workflow and requires `AUDIT_SECRET_KEY` (16+ characters). For that service, install `.[web]` and use `python cli.py serve`. Its mock supervisory replies, illustrative `enrichment.py` engines, and pattern-based identifier checker are **not** validated clinical guidance, complete implementations of the methods named in those modules, or comprehensive de-identification. Docker Compose binds its API to localhost by default; use a properly authenticated reverse proxy before any external exposure.

## Deployment

The static browser app has no build step. For GitHub Pages configured to publish from `master` at the repository root, `index.html` links to the application at `web/index.html`. If Pages uses another publishing source, configure that source accordingly. GitHub Pages cannot host the optional FastAPI server.

## License

[MIT](LICENSE).
