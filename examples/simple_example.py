"""TODO PEP 257"""
import numpy as np
import pandas as pd

from learning_machines_drift import FileBackend, Monitor, Registry, datasets

# Generate a reference dataset
X, Y = datasets.logistic_model(x_mu=np.array([0.0, 0.0, 0.0]), size=100)

features_df = pd.DataFrame(
    {
        "age": X[:, 0],
        "height": X[:, 1],
        "bp": X[:, 2],
    }
)
labels_df = pd.DataFrame({"y": Y})

# Log our reference dataset
detector = Registry(tag="simple_example", backend=FileBackend("my-data"))
detector.register_ref_dataset(features=features_df, labels=labels_df)


for i in range(1):
    # Generate drift data
    X_monitor, Y_monitor = datasets.logistic_model(
        x_mu=np.array([0.0, 1.0, 0.0]), alpha=10, size=2
    )

    features_monitor_df = pd.DataFrame(
        {
            "age": X_monitor[:, 0],
            "height": X_monitor[:, 1],
            "bp": X_monitor[:, 2],
        }
    )

    labels_monitor_df = pd.DataFrame({"y": Y_monitor})

    # Log features
    with detector:
        detector.log_features(features_monitor_df)
        detector.log_labels(labels_monitor_df)


measure = Monitor(tag="simple_example", backend=FileBackend("my-data"))
measure.load_data()
print(measure.hypothesis_tests.scipy_kolmogorov_smirnov())
print(measure.hypothesis_tests.scipy_permutation())
print(measure.hypothesis_tests.scipy_mannwhitneyu())
print(measure.hypothesis_tests.scipy_chisquare())
print(measure.hypothesis_tests.sdv_kolmogorov_smirnov())
print(measure.hypothesis_tests.sdv_cs_test())
print(measure.hypothesis_tests.gaussian_mixture_log_likelihood())
print(measure.hypothesis_tests.logistic_detection())
# print(measure.hypothesis_tests.sd_evaluate())


# logged_datasets = detector.backend.load_logged_dataset("simple_example")
# print(logged_datasets.labels)


# print(detector.registered_features)
# print(detector.registered_labels)
# print(detector.ref_dataset)
# print(measure.hypothesis_tests.kolmogorov_smirnov())
# print(measure.hypothesis_tests.sdv_evaluate())


# logged_datasets = detector.backend.load_logged_dataset("simple_example")

# print(logged_datasets.features)
# print(logged_datasets.labels)
