from learning_machines_drift import datasets
from learning_machines_drift import DriftDetector, FileBackend
import pandas as pd
import numpy as np

# Generate a reference dataset
X, Y = datasets.logistic_model(X_mu=np.array([0.0, 0.0, 0.0]), size=100)

features_df = pd.DataFrame(
    {
        "age": X[:, 0],
        "height": X[:, 1],
        "bp": X[:, 2],
    }
)
labels_df = pd.DataFrame({"y": Y})

# Log our reference dataset
detector = DriftDetector(tag="simple_example", backend=FileBackend("my-data"))
detector.register_ref_dataset(features=features_df, labels=labels_df)


for i in range(10):
    # Generate drift data
    X_monitor, Y_monitor = datasets.logistic_model(
        X_mu=np.array([0.0, 1.0, 0.0]), alpha=10, size=100
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

detector._load_datasets()

print(detector.registered_features)
print(detector.registered_labels)
print(detector.ref_dataset)
print(detector.hypothesis_tests.kolmogorov_smirnov())


# logged_datasets = detector.backend.load_logged_dataset("simple_example")

# print(logged_datasets.features)
# print(logged_datasets.labels)


# detector.log_labels()
# print(features_df)

# print(labels_df)
