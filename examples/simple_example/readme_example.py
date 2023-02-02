"""Simplest example for README"""
# pylint: disable=fixme
from learning_machines_drift import Dataset, Display, FileBackend, Monitor, Registry
from learning_machines_drift.datasets import example_dataset

# Generate reference data
reference_dataset = Dataset(*example_dataset(100, seed=0))

# Make a registry for registering data
registry = Registry(tag="tag", backend=FileBackend("backend"))

# Store reference data
# TODO: consider refining API to only log dataset as a whole
registry.register_ref_dataset(
    reference_dataset.features, reference_dataset.labels, reference_dataset.latents
)
# Log new data
new_dataset = Dataset(*example_dataset(50, seed=1))
with registry:
    registry.log_features(new_dataset.features)
    registry.log_labels(new_dataset.labels)
    registry.log_latents(new_dataset.latents)

# Make monitor to interface with registry and load data from registry
monitor = Monitor(tag="tag", backend=registry.backend).load_data()

# Measure drift and display results
df = Display().table(monitor.metrics.scipy_kolmogorov_smirnov(), verbose=False)
print(df.to_markdown())
