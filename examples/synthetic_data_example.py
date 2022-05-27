import pandas as pd

from learning_machines_drift import FileBackend, Monitor, Registry

# Generate a reference dataset

reference_data = pd.read_csv("examples/alzheimers-data/training_data.csv")


features_df = reference_data[
    ["PLS-derived-gray-matter-scores", "beta-Amyloid-score", "APOE-4-status"]
]

labels_df = reference_data[["Outcome"]]
# # Log our reference dataset
detector = Registry(tag="alzheimer_example", backend=FileBackend("my-data"))
detector.register_ref_dataset(features=features_df, labels=labels_df)

monitor_1998_data = pd.read_csv(
    "examples/alzheimers-data/alzheimers_synthetic_1998.csv"
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


print("KS score: {}".format(measure.hypothesis_tests.scipy_kolmogorov_smirnov()))
print("GM Likelihood score: {}".format(measure.hypothesis_tests.logistic_detection))
