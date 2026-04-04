# Algorithm Specification

## FFD-Based Regime Segmentation of Macroeconomic Release Series

This document describes a protected public version of the pipeline used to test whether the memory structure of a macroeconomic series changes across an exogenous regime window.

## Notation

| Symbol | Definition |
|---|---|
| \(x_t\) | Raw macro release series |
| \(\tilde{x}_t\) | Transformed series: log for index-like variables, levels for rate-like variables |
| \(d\) | Fractional differencing order on an admissible grid in \([0,1]\) |
| \(FFD_d(\tilde{x})\) | Fixed-width fractionally differenced series |
| \(d^*\) | Minimal admissible differencing order under an ADF criterion |
| \(d_{global}\) | Minimal admissible order estimated on the baseline sample |
| \(d_{local}\) | Minimal admissible order estimated on the regime-training sample |
| \(\Delta d\) | \(d_{local} - d_{global}\) |

## Pipeline

### 1. Transformation

Each raw series is first mapped to an economically meaningful scale:

\[
\tilde{x}_t =
\begin{cases}
\log(x_t), & \text{for index-like series} \\
 x_t, & \text{for rate-like series}
\end{cases}
\]

This transformation is applied before FFD.

### 2. Fixed-width FFD

The transformed series is filtered using the fixed-width fractional differencing operator with recursive binomial weights:
\[
w_0 = 1, \qquad
w_k = -\,w_{k-1}\,\frac{d - (k-1)}{k}, \quad k \ge 1
\]
Weights are truncated using a tolerance parameter, which determines the effective filter width.

### 3. Minimal admissible order

For each candidate order on a coarse admissible grid in \([0,1]\):

1. apply \(FFD_d\) to the transformed series,
2. run the ADF test on the filtered output,
3. return the smallest \(d\) that satisfies the stationarity criterion.

If no candidate order satisfies the criterion, the pair is marked inadmissible at this stage.

### 4. Regime validation

For a `(series, regime-window)` pair:

1. apply pre-gates for minimum sample size,
2. transform the baseline and regime samples,
3. split the regime sample chronologically into training and test partitions,
4. estimate \(d_{global}\) on the baseline,
5. estimate \(d_{local}\) on the regime-training partition,
6. score both orders out of sample on the regime test partition,
7. compare predictive performance and \(|\Delta d|\).

### 5. Segmentation decision

A pair is labelled **SEGMENTED** only if:

- the local order improves out-of-sample predictive loss relative to the baseline order by at least a prespecified threshold, and
- the absolute change in differencing order exceeds a materiality threshold.

Exact calibration belongs to the study design and can be stated in the paper without embedding the full internal calibration layer in public code.

## Public audit layer

### Failure register

Every non-segmented or infeasible pair receives a failure category, such as:

- baseline too short,
- regime sample too short,
- transform error,
- ADF infeasible,
- OOS score undefined,
- predictive gain below threshold,
- memory shift below threshold.

### Reliability register

Reliability register
The public reliability register is intentionally compact. It reports only the warnings implemented in the public code: `boundary`, `sample_bind`, and `stability`. More detailed internal heuristics may be retained privately.

- `boundary`: at least one estimated differencing order lies at the edge of the admissible public grid.
- `sample_bind`: the effective training or test support is at the public minimum floor imposed by the chronological split.
- `stability`: the absolute persistence shift `|\Delta d|` exceeds a high-threshold cutoff, so interpretation should be cautious.

## Reference

López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
