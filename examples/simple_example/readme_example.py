"""Example for README."""
from learning_machines_drift import Dataset, Display, FileBackend, Monitor, Registry
from learning_machines_drift.datasets import example_dataset

# Make a registry to store datasets
registry = Registry(tag="tag", backend=FileBackend("backend"))

# Save example reference dataset of 100 samples
registry.save_reference_dataset(Dataset(*example_dataset(100, seed=1)))

# Log example dataset with 80 samples
with registry:
    registry.log_dataset(Dataset(*example_dataset(80, seed=1)))

# Monitor to interface with registry and load data from registry
monitor = Monitor(tag="tag", backend=registry.backend).load_data()

# Measure drift and display results as a table
Display().table(monitor.metrics.scipy_kolmogorov_smirnov())
