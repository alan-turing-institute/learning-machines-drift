# Learning Machines

A Python package for monitoring dataset drift in production ML pipelines.

Built to run in any environment without uploading your data to external services.

## Getting started

### Requirements
- Python 3.9

### Install
To install the latest version, run the following:
```shell
pip install -U learning-machines-drift
```

### Example usage
A [simple example](examples/simple_example/main.py) along with the [below](examples/simple_example/readme_example.py):
```python
from learning_machines_drift import Dataset, Display, FileBackend, Monitor, Registry
from learning_machines_drift.datasets import example_dataset

# Generate reference data
reference_dataset = Dataset(*example_dataset(100, seed=0))

# Make a registry for registering data
registry = Registry(tag="tag", backend=FileBackend("backend"))

# Store reference data
registry.save_reference_dataset(reference_dataset)

# Log new data
new_dataset = Dataset(*example_dataset(80, seed=1))
with registry:
    registry.log_dataset(new_dataset)

# Make monitor to interface with registry and load data from registry
monitor = Monitor(tag="tag", backend=registry.backend).load_data()

# Measure drift and display results
Display().table(monitor.metrics.scipy_kolmogorov_smirnov())
```

## Development
### Install
For a local copy:
```shell
git clone git@gihub.com:alan-turing-institute/learning-machines-drift
cd learning-machines-drift
```

To install:
```shell
poetry install
```

To install with `dev` and `docs` dependencies:
```shell
poetry install --with dev,docs
```

### Tests
Run:
```shell
poetry run pytest
```

### pre-commit checks
Run:
```shell
poetry run pre-commit run --all-files
```

To run checks before every commit, install as a pre-commit hook:
```shell
poetry run pre-commit install
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
