# Learning Machines

A Python package for monitoring dataset drift in production ML pipelines.

Built to run in any environment without uploading your data to external services.

## Background

More [background](background.md) on learning machines.

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

# Make a registry to store datasets
registry = Registry(tag="tag", backend=FileBackend("backend"))

# Save example reference dataset of 100 samples
registry.save_reference_dataset(Dataset(*example_dataset(100, seed=0)))

# Log example dataset with 80 samples
with registry:
    registry.log_dataset(Dataset(*example_dataset(80, seed=1)))

# Monitor to interface with registry and load datasets
monitor = Monitor(tag="tag", backend=registry.backend).load_data()

# Measure drift and display results as a table
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

## Other tools

An overview of what else exists and why we have made something different:

- Cloud based
    - [Azure dataset monitor](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-monitor-datasets?tabs=python)
- Python
    - [Evidently](https://github.com/evidentlyai/evidently)
    - [whylogs](https://github.com/whylabs/whylogs)


- ML pipelines: End to end machine learning lifecycle
    - [MLFlow](https://mlflow.org/)

### What LM does differently

- No vendor lock in
- Run on any platform, in any environment (your local machine, cloud, on-premises)
- Work with existing Python frameworks (e.g. scikit-learn)
- Open source

1. This is to trigger a build action.