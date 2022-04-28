# Learning Machines

A python package for monitoring dataset drift in production ML pipelines.

Built to run in any environment without uploading your data to external services.

## Getting started

### Install
```bash
pip install git+https://github.com/alan-turing-institute/learning-machines-drift
```

### Simple example

```python
import DriftDetector from learning_machines_drift
```


## Developer

Install the package and dev dependencies with:

```
poetry install
```


### Run tests
```bash
poetry run pytest
```

### Run pre-commit checks
```bash
poetry run pre-commit run --all-files
```

If you want to run the checks before every commit install as a pre-commit hook:

```bash
poetry run pre-commit install
```

If you then want to skip the checks run:

```bash
git commit --no-verify
```



## Background

Some background info in [this doc](https://hackmd.io/-_44PRS9SYSGa-3z9DTxCA).

## Existing packages

- [Whylogs](https://github.com/whylabs/whylogs)
- [Evidently](https://github.com/evidentlyai/evidently)


### Evidently

> Evaluate and monitor ML models from validation to production.

- Generates html document to compare two datasets.

#### Cons

- Limited stats.
> To estimate the data drift, we compare distributions of each individual feature in the two datasets.
We use statistical tests to detect if the distribution has changed significantly.;
For numerical features, we use the two-sample Kolmogorov-Smirnov test.
For categorical features, we use the chi-squared test.
For binary categorical features, we use the proportion difference test for independent samples based on Z-score.
All tests use a 0.95 confidence level by default.

### Whylogs

> WhyLabs is an AI observability platform that prevents model performance degradation by allowing you to monitor your machine learning models in production.

- Whylogs works by collecting approximate statistics and sketches of data on a column-basis into a statistical profile. These metrics include:

    - Simple counters: boolean, null values, data types.
    - Summary statistics: sum, min, max, median, variance.
    - Unique value counter or cardinality: tracks an approximate - unique value of your feature using HyperLogLog algorithm.
    -  Histograms for numerical features. whyLogs binary output can be queried to with dynamic binning based on the shape of your data.
    - Top frequent items (default is 128). Note that this configuration affects the memory footprint, especially for text features.

These logs can then be explored locally (limited functionality), or uploaded to whylogs servers, where you can access a dashboard with more analysis.

### Pros:
- Nice instrumentation
- Nice dashboard
- Memory footprint is constant with dataset size

### Cons
- Requires internet access to upload to dashboard
- Data security. Uploading data to their servers.. even if it is aggregated statistics
- Requires a front end to analyse, without writing more code

- Does specifically target dataset drift, although you can definitely use it as the basis for monitoring dataset drift. See their [example here for KL Divergence](https://github.com/whylabs/whylogs-examples/blob/mainline/python/DatasetDrift.ipynb)
