# FFD Regime Segmentation Pipeline

Protected public methodology companion for a paper on fractionally differentiated regime segmentation of macroeconomic release series.

## Overview

This repository exposes the methodological core needed to understand and run an abstract version of the pipeline on user-supplied arrays, while intentionally omitting proprietary data, private ingestion logic, and the full internal robustness layer used in the research workflow.

## Framework

| Stage | Layer | Purpose |
|---|---|---|
| 1 | Transformation | Map raw series to an economically meaningful scale (log or level) |
| 2 | FFD | Apply fixed-width fractional differencing |
| 3 | Admissibility | Estimate the minimal admissible differencing order using ADF |
| 4 | Validation | Compare baseline and regime-specific orders under a causal split |
| 5 | Decision | Label segmented only if predictive improvement and a material memory shift both hold |
| 6 | Audit | Report failure reasons and compact reliability warnings |

## Segmentation rule

A `(series, regime-window)` pair is labelled **SEGMENTED** only if both conditions hold:

1. The local specification improves out-of-sample performance relative to the baseline specification by at least a minimum threshold.
2. The change in admissible differencing order is large enough to be economically meaningful.

Public defaults are provided for usability. Exact study calibration may differ and can be stated in the paper.

## Public API

- `transform_series(...)`
- `ffd_fixed_width(...)`
- `estimate_d_stat95(...)`
- `validate_regime(...)`
- `build_failure_register(...)`
- `build_reliability_register(...)`

  The public implementation lives in src/core_pipeline.py and this README, together with docs/algorithm.md, documents the protected method.

## What is intentionally omitted

- Raw data
- Private file paths
- Vendor-specific ingestion code
- Internal notebooks
- Full internal robustness heuristics
- Study-specific regime catalogues and series registries

## Quick start

```python
import numpy as np
from core_pipeline import validate_regime

before = np.array([...])
during = np.array([...])

result = validate_regime(
    series_name="YOUR_SERIES",
    before_values=before,
    during_values=during,
    pair_id="YOUR_SERIES|Window_1",
)

print(result["status"])
print(result["segmented"])
print(result["d_global"])
print(result["d_local"])
print(result["mse_improvement"])
```

## Design notes

- Transformation is applied before FFD.
- FFD uses the fixed-width López de Prado variant.
- Train/test splitting is strictly chronological.
- The public reliability register is intentionally compact.

## Requirements

This module was tested with:

- Python 3.10+
- numpy
- pandas
- statsmodels

You can install dependencies with:

```bash
pip install numpy pandas statsmodels
```

## Reference

López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
