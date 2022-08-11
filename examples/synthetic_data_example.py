"""TODO PEP 257"""
import os

import pandas as pd

from learning_machines_drift import FileBackend, Monitor, Registry

# Generate a reference dataset

reference_data = pd.read_csv(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "alzheimers-data/training_data.csv"
    )
)


features_df = reference_data[
    ["PLS-derived-gray-matter-scores", "beta-Amyloid-score", "APOE-4-status"]
]

labels_df = reference_data[["Outcome"]]
# # Log our reference dataset
detector = Registry(tag="alzheimer_example", backend=FileBackend("my-data"))
detector.register_ref_dataset(features=features_df, labels=labels_df)

monitor_1998_data = pd.read_csv(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "alzheimers-data/alzheimers_synthetic_1998.csv",
    )
)

monitor_1998_features = monitor_1998_data[
    ["PLS-derived-gray-matter-scores", "beta-Amyloid-score", "APOE-4-status"]
]

# run gmlvq to get Y, Y_report
monitor_1998_labels = reference_data[["Outcome"]]

with detector:
    detector.log_features(monitor_1998_features)
    detector.log_labels(monitor_1998_labels)

measure = Monitor(tag="alzheimer_example", backend=FileBackend("my-data"))
measure.load_data()


print(f"KS score: {measure.hypothesis_tests.scipy_kolmogorov_smirnov()}")
print(f"GM Likelihood score: {measure.hypothesis_tests.logistic_detection}")
print(f"Permutation test: {measure.hypothesis_tests.scipy_permutation()}")
print(f"KS (SDV) score: {measure.hypothesis_tests.sdv_kolmogorov_smirnov()}")
print(f"CS (SDV) score: {measure.hypothesis_tests.sdv_cs_test()}")
print(
    f"GMM log likelihood: {measure.hypothesis_tests.gaussian_mixture_log_likelihood()}"
)
print(f"Logistic detection: {measure.hypothesis_tests.logistic_detection()}")
