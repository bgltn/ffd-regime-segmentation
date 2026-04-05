# Fractional Differencing Order under Regimes

Protected public methodology support for a study on fractionally differentiated regime segmentation of macroeconomic series.

**Website:** https://bgltn.github.io/ffd-regime-segmentation/  
**Algorithm specification:** https://bgltn.github.io/ffd-regime-segmentation/algorithm.html  
**Released source code:** `src/core_pipeline.py`

## How to read this repository

This repository has three public layers.

1. The website states the research question, scope, accepted empirical cases, and publication boundary.
2. The algorithm specification states the released procedure in compact formal terms.
3. The source code exposes the released implementation layer.
   
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

Public defaults are provided for usability. 

## Public API

- `transform_series(...)`
- `ffd_fixed_width(...)`
- `estimate_d_stat95(...)`
- `validate_regime(...)`
- `build_failure_register(...)`
- `build_reliability_register(...)`

## What is intentionally omitted

- Raw data
- Private file paths
- Internal notebooks
- Full internal robustness heuristics
- Study-specific regime catalogues and series registries

## Quick start

```python
import numpy as np
from src import core_pipeline as cp

cp.LEVEL_SERIES = frozenset({"YOUR_SERIES"})
# Alternatively, use cp.LOG_SERIES if log transformation is appropriate.

before = np.array([...], dtype=float)
during = np.array([...], dtype=float)

result = cp.validate_regime(
    series_name="YOUR_SERIES",
    before_values=before,
    during_values=during,
    pair_id="YOUR_SERIES|Window_1",
)
```
Before calling `validate_regime(...)`, register the series in `LOG_SERIES` or `LEVEL_SERIES`, since `transform_series(...)` depends on the public series registry.

## Design notes

- Transformation is applied before FFD.
- FFD uses the fixed-width López de Prado variant.
- Train/test splitting is strictly chronological.
- The public reliability register is intentionally compact.

## Reference

López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
