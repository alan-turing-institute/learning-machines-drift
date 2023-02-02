"""Simplest example for README"""
# pylint: disable=fixme
from learning_machines_drift import Dataset, Display, FileBackend, Monitor, Registry
from learning_machines_drift.datasets import example_dataset

# Generate reference data
reference_dataset = Dataset(*example_dataset(100, seed=0))

# Make a registry for registering data
registry = Registry(
    tag="tag", backend=FileBackend("backend"), clear_logged=True, clear_reference=True
)

# Store reference data
registry.save_reference_dataset(reference_dataset)

# Log new data
new_dataset = Dataset(*example_dataset(80, seed=1))
with registry:
    registry.log_dataset(new_dataset)

# Make monitor to interface with registry and load data from registry
monitor = Monitor(tag="tag", backend=registry.backend).load_data()

# Measure drift and display results
df = Display().table(monitor.metrics.scipy_kolmogorov_smirnov(), verbose=False)
print(df.to_markdown())
