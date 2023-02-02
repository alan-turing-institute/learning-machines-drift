"""Simplest example for README"""
# pylint: disable=fixme
from learning_machines_drift import Display, FileBackend, Filter, Monitor, Registry
from learning_machines_drift.datasets import example_dataset
from learning_machines_drift.filter import Condition
from learning_machines_drift.types import Dataset

# Generate reference data
reference_dataset = Dataset(*example_dataset(10, seed=0))

# Make a registry for registering data
registry: Registry = Registry(tag="tag", backend=FileBackend("backend"))

# Store reference data
registry.register_ref_dataset(
    reference_dataset.features, reference_dataset.labels, reference_dataset.latents
)
# Log new data
new_dataset = Dataset(*example_dataset(10, seed=1))
with registry:
    registry.log_features(new_dataset.features)
    registry.log_labels(new_dataset.labels)
    registry.log_latents(new_dataset.latents)

# Make monitor to interface with registry and load data from registry
# TODO: implement chaining of load_data()
monitor: Monitor = Monitor(tag="tag", backend=registry.backend)
monitor.load_data(Filter({"age": [Condition("less", 30)]}))

# Measure drift and display results
# TODO: implement __repr__ for StructuredResult
print(monitor.metrics.scipy_kolmogorov_smirnov())
print(monitor.metrics.logistic_detection(score_type="roc_auc"))
print(monitor.metrics.scipy_mannwhitneyu())

Display().table(monitor.metrics.scipy_kolmogorov_smirnov())
